package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/logs"
	"github.com/nchapman/lleme/internal/proxy"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var (
	serverHost      string
	serverPort      int
	serverMaxModels int
	serverDetach    bool
)

var serverCmd = &cobra.Command{
	Use:     "server",
	Short:   "Manage the proxy server",
	GroupID: "server",
	Long: `Manage the lleme proxy server that routes requests to llama.cpp backends.

The proxy server:
  - Routes requests to the appropriate model backend
  - Automatically loads models on demand
  - Manages LRU eviction when max models is reached
  - Unloads idle models after the configured timeout

Examples:
  lleme server start          # Start in foreground
  lleme server start -d       # Start in background (detached)
  lleme server stop           # Stop the server
  lleme server restart        # Restart the server (always in background)`,
}

// validateServerFlags validates CLI overrides for host/port/max-models.
// Zero values mean "use config default" and are allowed.
func validateServerFlags() error {
	if serverPort != 0 && (serverPort < 1 || serverPort > 65535) {
		return fmt.Errorf("--port %d is out of range (must be 1-65535)", serverPort)
	}
	if serverMaxModels < 0 {
		return fmt.Errorf("--max-models %d must be non-negative (0 means unlimited)", serverMaxModels)
	}
	return nil
}

var serverStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the proxy server",
	Long: `Start the lleme proxy server in the foreground (or detached with -d).

The server listens for OpenAI- and Anthropic-compatible API requests and routes
them to the appropriate llama.cpp backend, loading models on demand.

Use --host, --port, and --max-models to override the values from config.`,
	PreRun: func(cmd *cobra.Command, args []string) {
		if err := validateServerFlags(); err != nil {
			ui.Fatal("%v", err)
		}
		if err := ensureBackends(backendsForLocalModels()); err != nil {
			ui.Fatal("%v", err)
		}
	},
	Run: func(cmd *cobra.Command, args []string) {
		// Check if already running
		if existingState := proxy.GetRunningProxyState(); existingState != nil {
			ui.PrintError("Server already running on http://%s:%d (PID %d)",
				existingState.Host, existingState.Port, existingState.PID)
			fmt.Println("Use 'lleme server stop' to stop the existing server first")
			ui.ExitFunc(1)
			return
		}

		if serverDetach {
			// Detached mode: spawn daemon with CLI overrides, daemon handles everything else
			startServerDetached()
			return
		}

		// Foreground mode: we ARE the server
		startServerForeground()
	},
}

var serverStopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop the proxy server",
	Long: `Stop the running lleme proxy server.

Unloads all backends gracefully (SIGTERM with a 5s timeout before SIGKILL) and
clears the on-disk state file.`,
	Run: func(cmd *cobra.Command, args []string) {
		stopped, err := stopServer()
		if err != nil {
			fmt.Printf("%s %v\n", ui.ErrorMsg("Error:"), err)
			return
		}
		if !stopped {
			fmt.Println(ui.Muted("Server is not running"))
			return
		}
		fmt.Println("Server stopped")
	},
}

var serverRestartCmd = &cobra.Command{
	Use:   "restart",
	Short: "Restart the proxy server",
	Long: `Stop the running proxy (if any) and start a fresh instance in the background.

Use --host, --port, and --max-models to change the server's binding on restart.`,
	PreRun: func(cmd *cobra.Command, args []string) {
		if err := validateServerFlags(); err != nil {
			ui.Fatal("%v", err)
		}
	},
	Run: func(cmd *cobra.Command, args []string) {
		stopped, _ := stopServer()
		if stopped {
			fmt.Println("Stopped server")
		}

		fmt.Println("Starting server...")

		startServerDetached()
	},
}

// stopServer stops the running server and returns whether it was stopped.
func stopServer() (bool, error) {
	state := proxy.GetRunningProxyState()
	if state == nil {
		// No state file found - try to find process by port (for servers started by older versions)
		port := 11313 // Default port as fallback
		if cfg, err := config.Load(); err == nil {
			port = cfg.Server.Port
		}
		if serverPort != 0 {
			port = serverPort
		}
		return stopServerByPort(port)
	}

	process, err := os.FindProcess(state.PID)
	if err != nil {
		proxy.ClearProxyState()
		return false, fmt.Errorf("could not find server process: %w", err)
	}

	if err := process.Signal(syscall.SIGTERM); err != nil {
		proxy.ClearProxyState()
		return false, fmt.Errorf("could not signal server process: %w", err)
	}

	for range 40 { // 4 seconds max
		time.Sleep(100 * time.Millisecond)
		if err := process.Signal(syscall.Signal(0)); err != nil {
			proxy.ClearProxyState()
			return true, nil
		}
	}

	process.Kill()
	proxy.ClearProxyState()
	return true, nil
}

// stopServerByPort finds and stops a process listening on the given port.
// Used as a fallback when no state file exists (e.g., server started by older version).
func stopServerByPort(port int) (bool, error) {
	pid := findProcessOnPort(port)
	if pid == 0 {
		return false, nil
	}

	// Verify it's a lleme process before killing
	if !isLlemeProcess(pid) {
		return false, fmt.Errorf("process on port %d (PID %d) is not a lleme server", port, pid)
	}

	process, err := os.FindProcess(pid)
	if err != nil {
		return false, fmt.Errorf("could not find process %d: %w", pid, err)
	}

	if err := process.Signal(syscall.SIGTERM); err != nil {
		return false, fmt.Errorf("could not signal process %d: %w", pid, err)
	}

	for range 40 { // 4 seconds max
		time.Sleep(100 * time.Millisecond)
		if err := process.Signal(syscall.Signal(0)); err != nil {
			return true, nil
		}
	}

	if err := process.Kill(); err != nil {
		return false, fmt.Errorf("failed to kill process %d: %w", pid, err)
	}
	return true, nil
}

// findProcessOnPort uses lsof to find the PID of a process listening on a port.
func findProcessOnPort(port int) int {
	cmd := exec.Command("lsof", "-ti", fmt.Sprintf("tcp:%d", port))
	output, err := cmd.Output()
	if err != nil {
		return 0
	}

	// lsof may return multiple PIDs (parent/child); take the first one
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 0 {
		return 0
	}
	var pid int
	if _, err := fmt.Sscanf(lines[0], "%d", &pid); err != nil {
		return 0
	}
	return pid
}

// isLlemeProcess checks if the given PID is a lleme process.
func isLlemeProcess(pid int) bool {
	cmd := exec.Command("ps", "-p", fmt.Sprintf("%d", pid), "-o", "args=")
	output, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(output), "lleme")
}

func startServerForeground() {
	// Load config
	cfg, err := config.Load()
	if err != nil {
		ui.Fatal("Failed to load config: %v", err)
	}

	// Build proxy config from app config + CLI overrides
	proxyCfg := proxy.ConfigFromAppConfig(cfg.Server)
	if serverHost != "" {
		proxyCfg.Host = serverHost
	}
	if serverPort != 0 {
		proxyCfg.Port = serverPort
	}
	if serverMaxModels != 0 {
		proxyCfg.MaxModels = serverMaxModels
	}

	// Create and start server (handles orphan cleanup internally)
	server := proxy.NewServer(proxyCfg, cfg)
	if err := server.Start(); err != nil {
		ui.Fatal("Failed to start server: %v", err)
	}

	// Print startup info
	fmt.Printf("Server started on http://%s:%d\n", proxyCfg.Host, proxyCfg.Port)
	fmt.Println()
	fmt.Printf("  %-14s %d\n", "Max models", proxyCfg.MaxModels)
	fmt.Printf("  %-14s %v\n", "Idle timeout", proxyCfg.IdleTimeout)
	fmt.Printf("  %-14s %d-%d\n", "Backend ports", proxyCfg.BackendPortMin, proxyCfg.BackendPortMax)
	fmt.Println()
	fmt.Println(ui.Header("Endpoints"))
	fmt.Printf("  %-12s %s %s\n", "Web UI", ui.Muted("GET"), "/")
	fmt.Printf("  %-12s %s %s\n", "Chat", ui.Muted("POST"), "/v1/chat/completions")
	fmt.Printf("  %-12s %s %s\n", "Completions", ui.Muted("POST"), "/v1/completions")
	fmt.Printf("  %-12s %s %s\n", "Messages", ui.Muted("POST"), "/v1/messages")
	fmt.Printf("  %-12s %s %s\n", "Models", ui.Muted("GET"), "/v1/models")
	fmt.Printf("  %-12s %s %s\n", "Status", ui.Muted("GET"), "/api/status")
	fmt.Println()

	installed, _ := llama.GetInstalledVersion()
	if installed != nil {
		fmt.Println(ui.LlamaCppCredit(installed.TagName))
	}
	fmt.Println(ui.Muted("Press Ctrl+C to stop"))

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	fmt.Println()
	fmt.Println("Shutting down...")

	if err := server.Stop(); err != nil {
		fmt.Printf("%s Failed to stop server cleanly: %v\n", ui.ErrorMsg("Warning:"), err)
	}

	proxy.ClearProxyState()
	fmt.Println("Server stopped")
}

func startServerDetached() {
	executable, err := os.Executable()
	if err != nil {
		ui.Fatal("Failed to get executable path: %v", err)
	}

	// Build args: only pass CLI overrides that were explicitly set
	args := []string{"internal-serve"}
	if serverHost != "" {
		args = append(args, "--host", serverHost)
	}
	if serverPort != 0 {
		args = append(args, "--port", fmt.Sprintf("%d", serverPort))
	}
	if serverMaxModels != 0 {
		args = append(args, "--max-models", fmt.Sprintf("%d", serverMaxModels))
	}

	// Spawn daemon - it handles its own logging, config loading, etc.
	cmd := exec.Command(executable, args...)
	cmd.Env = os.Environ()
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid: true,
	}

	if err := cmd.Start(); err != nil {
		ui.Fatal("Failed to start server in background: %v", err)
	}

	// Poll for state file (up to 5 seconds)
	logPath := logs.ProxyLogPath()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if state := proxy.GetRunningProxyState(); state != nil {
			fmt.Printf("Server started in background on http://%s:%d (PID %d)\n", state.Host, state.Port, state.PID)
			fmt.Printf("Web UI available at http://%s:%d\n", state.Host, state.Port)
			fmt.Printf("Logs: %s\n", ui.Muted(logPath))
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Printf("%s Server may have failed to start. Check logs: %s\n", ui.ErrorMsg("Warning:"), logPath)
}

// internalServeCmd is the daemon process for background serving.
// Fully self-contained: handles its own logging, config, lifecycle, and state.
var internalServeCmd = &cobra.Command{
	Use:    "internal-serve",
	Hidden: true,
	PreRun: func(cmd *cobra.Command, args []string) {
		if err := validateServerFlags(); err != nil {
			ui.Fatal("%v", err)
		}
	},
	Run: func(cmd *cobra.Command, args []string) {
		// Set up logging - daemon owns its log file
		logFile, err := logs.NewRotatingWriter(logs.ProxyLogPath())
		if err != nil {
			ui.Fatal("Failed to open proxy log: %v", err)
		}
		defer logFile.Close()
		logs.InitLogger(logFile, false)

		// Load config
		cfg, err := config.Load()
		if err != nil {
			logs.Warn("Failed to load config", "error", err)
			ui.Fatal("Failed to load config: %v", err)
		}

		// Build proxy config from app config + CLI overrides
		proxyCfg := proxy.ConfigFromAppConfig(cfg.Server)
		if serverHost != "" {
			proxyCfg.Host = serverHost
		}
		if serverPort != 0 {
			proxyCfg.Port = serverPort
		}
		if serverMaxModels != 0 {
			proxyCfg.MaxModels = serverMaxModels
		}

		// Create and start server (handles orphan cleanup and state persistence internally)
		server := proxy.NewServer(proxyCfg, cfg)
		if err := server.Start(); err != nil {
			logs.Warn("Failed to start server", "error", err)
			ui.Fatal("Failed to start server: %v", err)
		}

		// Wait for shutdown signal
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		server.Stop()
		proxy.ClearProxyState()
	},
}

func init() {
	rootCmd.AddCommand(serverCmd)
	rootCmd.AddCommand(internalServeCmd)

	serverCmd.AddCommand(serverStartCmd)
	serverCmd.AddCommand(serverStopCmd)
	serverCmd.AddCommand(serverRestartCmd)

	serverStartCmd.Flags().StringVarP(&serverHost, "host", "H", "", "Server host (default from config)")
	serverStartCmd.Flags().IntVarP(&serverPort, "port", "p", 0, "Server port (default from config)")
	serverStartCmd.Flags().IntVar(&serverMaxModels, "max-models", 0, "Maximum concurrent models (default from config)")
	serverStartCmd.Flags().BoolVarP(&serverDetach, "detach", "d", false, "Run server in background")

	serverRestartCmd.Flags().StringVarP(&serverHost, "host", "H", "", "Server host (default from config)")
	serverRestartCmd.Flags().IntVarP(&serverPort, "port", "p", 0, "Server port (default from config)")
	serverRestartCmd.Flags().IntVar(&serverMaxModels, "max-models", 0, "Maximum concurrent models (default from config)")

	internalServeCmd.Flags().StringVar(&serverHost, "host", "", "")
	internalServeCmd.Flags().IntVar(&serverPort, "port", 0, "")
	internalServeCmd.Flags().IntVar(&serverMaxModels, "max-models", 0, "")
}
