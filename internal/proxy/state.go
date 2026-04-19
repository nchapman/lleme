package proxy

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/fileutil"
	"github.com/nchapman/lleme/internal/logs"
)

const proxyStateFile = "proxy-state.json"

// BackendState persists backend process info for orphan cleanup
type BackendState struct {
	ModelName string    `json:"model_name"`
	PID       int       `json:"pid"`
	Port      int       `json:"port"`
	StartedAt time.Time `json:"started_at"`
}

// ProxyState persists proxy metadata for CLI commands to discover
type ProxyState struct {
	PID       int            `json:"pid"`
	Host      string         `json:"host"`
	Port      int            `json:"port"`
	StartedAt time.Time      `json:"started_at"`
	Backends  []BackendState `json:"backends,omitempty"`
}

// ProxyStatePath returns the path to the proxy state file
func ProxyStatePath() string {
	return filepath.Join(config.PidsPath(), proxyStateFile)
}

// SaveProxyState saves the proxy state to disk using atomic writes
func SaveProxyState(state *ProxyState) error {
	if err := os.MkdirAll(filepath.Dir(ProxyStatePath()), 0755); err != nil {
		return fmt.Errorf("failed to create state directory: %w", err)
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	if err := fileutil.AtomicWriteFile(ProxyStatePath(), data, 0644); err != nil {
		return fmt.Errorf("failed to write state: %w", err)
	}

	return nil
}

// LoadProxyState loads the proxy state from disk
func LoadProxyState() (*ProxyState, error) {
	data, err := os.ReadFile(ProxyStatePath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read state: %w", err)
	}

	var state ProxyState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to parse state: %w", err)
	}

	return &state, nil
}

// ClearProxyState removes the proxy state file
func ClearProxyState() error {
	if err := os.Remove(ProxyStatePath()); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// IsProxyRunning checks if the proxy is running based on saved state
func IsProxyRunning() bool {
	state, err := LoadProxyState()
	if err != nil || state == nil {
		return false
	}

	return isProcessRunning(state.PID)
}

// GetRunningProxyState returns the proxy state if the proxy is running
func GetRunningProxyState() *ProxyState {
	state, err := LoadProxyState()
	if err != nil || state == nil {
		return nil
	}

	if !isProcessRunning(state.PID) {
		// Proxy not running - don't clear state here as CleanupOrphanedBackends
		// needs it to find orphaned backends
		return nil
	}

	return state
}

// isProcessRunning checks if a process with the given PID is running
func isProcessRunning(pid int) bool {
	if pid <= 0 {
		return false
	}
	// On Unix, FindProcess always succeeds - the signal check does the real work
	process, _ := os.FindProcess(pid)
	return process.Signal(syscall.Signal(0)) == nil
}

// CleanupOrphanedBackends kills any orphaned llama-server processes from a previous
// proxy instance that crashed. Returns the number of processes killed.
func CleanupOrphanedBackends() int {
	state, err := LoadProxyState()
	if err != nil || state == nil {
		return 0
	}

	// If the proxy is still running, don't touch the backends
	if isProcessRunning(state.PID) {
		return 0
	}

	killed := 0
	for _, backend := range state.Backends {
		if backend.PID <= 0 {
			continue
		}

		if !isProcessRunning(backend.PID) {
			continue
		}

		// Verify this is actually a llama-server process
		if !isLlamaServerProcess(backend.PID) {
			continue
		}

		// Kill the orphaned backend
		if killProcess(backend.PID) {
			logs.Info("Cleaned up orphaned backend", "model", backend.ModelName, "pid", backend.PID)
			killed++
		}
	}

	// Clean up stale state file since proxy is dead
	ClearProxyState()

	return killed
}

// isLlamaServerProcess checks if the given PID is a llama-server process.
// Uses ps command which works on both Linux and macOS.
func isLlamaServerProcess(pid int) bool {
	cmd := exec.Command("ps", "-p", fmt.Sprintf("%d", pid), "-o", "comm=")
	output, err := cmd.Output()
	if err != nil {
		return false
	}
	return containsLlamaServer(string(output))
}

// containsLlamaServer checks if a command line contains llama-server
func containsLlamaServer(cmdline string) bool {
	return strings.Contains(cmdline, "llama-server") || strings.Contains(cmdline, "llama_server")
}

// killProcess sends SIGTERM, waits briefly, then SIGKILL if needed
func killProcess(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}

	// Try graceful termination
	if err := process.Signal(syscall.SIGTERM); err != nil {
		return false
	}

	// Wait up to 2 seconds for graceful exit
	for range 20 {
		time.Sleep(100 * time.Millisecond)
		if !isProcessRunning(pid) {
			return true
		}
	}

	// Force kill
	process.Kill()
	return true
}
