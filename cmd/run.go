package cmd

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/logs"
	"github.com/nchapman/lleme/internal/presets"
	"github.com/nchapman/lleme/internal/proxy"
	"github.com/nchapman/lleme/internal/server"
	"github.com/nchapman/lleme/internal/tui/chat"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var (
	tokens           int
	temperature      float64
	topP             float64
	topK             int
	minP             float64
	repeatPenalty    float64
	presencePenalty  float64
	frequencyPenalty float64
	systemPrompt     string

	// Server options (require model reload)
	ctxSize   int
	gpuLayers int
	threads   int
)

var runCmd = &cobra.Command{
	Use:     "run <model|persona> [prompt]",
	Short:   "Run inference with a model or persona",
	GroupID: "model",
	Long: `Run inference with a model or persona. The first argument can be:

Models:
  - Full name: bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
  - Partial name: llama (matches if unique)
  - Repo name: Llama-2-7B-GGUF

Personas:
  - Name of a saved persona (see 'lleme persona list')
  - Personas provide saved system prompts and options

The proxy server will be auto-started if not running.
Models are loaded on-demand and unloaded after idle timeout.`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := config.Load()
		if err != nil {
			ui.Fatal("Failed to load config: %v", err)
		}

		// Backend install is deferred until after model resolution so we
		// only download the runtime we actually need (GGUF → llama.cpp,
		// MLX → SwiftLM). See Step 2 below.

		modelQuery := args[0]
		promptStartIdx := 1 // Where prompt args begin (shifts if persona has no model)
		personaName := ""   // Track persona name for display

		// Check if this is a persona (personas take precedence over model names)
		var activePersona *config.Persona
		if config.PersonaExists(modelQuery) {
			personaName = modelQuery // Save persona name before modelQuery changes
			persona, err := config.LoadPersona(modelQuery)
			if err != nil {
				ui.Fatal("Failed to load persona: %v", err)
			}
			activePersona = persona

			// Get model from persona or second argument
			if persona.Model != "" {
				modelQuery = persona.Model
			} else if len(args) > 1 {
				modelQuery = args[1]
				promptStartIdx = 2 // Prompt starts after persona and model args
			} else {
				ui.PrintError("Persona '%s' has no model. Specify one:", args[0])
				fmt.Printf("  lleme run %s <model> [prompt]\n", args[0])
				ui.ExitFunc(1)
				return
			}

			// Apply persona system prompt if not overridden by flag
			if systemPrompt == "" && persona.System != "" {
				systemPrompt = persona.System
			}
		}

		// Step 2: Validate model exists (or offer to pull)
		resolvedModel, err := validateModel(modelQuery, cfg)
		if err != nil {
			ui.Fatal("%v", err)
		}

		// Install the backend this specific model needs. validateModel may
		// have just pulled it, so metadata is fresh and BackendKindForModelName
		// returns the right kind.
		kind := hf.BackendKindForModelName(resolvedModel.FullName)
		need := neededBackends{
			Llama:   kind == hf.BackendGGUF,
			SwiftLM: kind == hf.BackendMLX,
		}
		if err := ensureBackends(need); err != nil {
			ui.Fatal("%v", err)
		}

		// Step 3: Ensure proxy is running
		proxyURL, err := ensureProxyRunning(cfg)
		if err != nil {
			ui.Fatal("Failed to start proxy: %v", err)
		}

		// Create API client pointing to proxy
		api := server.NewAPIClientFromURL(proxyURL)

		// Check health
		if err := api.Health(); err != nil {
			ui.Fatal("Proxy health check failed: %v", err)
		}

		// Use the resolved full model name
		modelName := resolvedModel.FullName

		// Look up built-in preset for this model family.
		activePreset, matchedPattern := presets.Find(modelName)
		if activePreset != nil {
			fmt.Println(ui.Muted("Using preset: " + activePreset.Name + " (matched " + matchedPattern + ")"))
			logs.Info("Applied preset", "name", activePreset.Name, "pattern", matchedPattern, "model", modelName)
		} else {
			logs.Debug("No preset matched", "model", modelName)
		}

		// Track which server options were explicitly set
		ctxSizeSet := cmd.Flags().Changed("ctx-size")
		gpuLayersSet := cmd.Flags().Changed("gpu-layers")
		threadsSet := cmd.Flags().Changed("threads")

		promptArg := ""
		if len(args) > promptStartIdx {
			promptArg = strings.Join(args[promptStartIdx:], " ")
		}

		// Check if input is piped (non-interactive)
		stat, _ := os.Stdin.Stat()
		isPiped := (stat.Mode() & os.ModeCharDevice) == 0

		// Read piped input if present
		if isPiped {
			input, err := io.ReadAll(os.Stdin)
			if err != nil {
				ui.Fatal("Failed to read stdin: %v", err)
			}
			stdinContent := strings.TrimSpace(string(input))
			if stdinContent != "" {
				if promptArg != "" {
					promptArg = promptArg + "\n" + stdinContent
				} else {
					promptArg = stdinContent
				}
			}
		}

		// One-shot mode for CLI prompts or piped input
		if promptArg != "" {
			// Resolve the backend kind so preset/persona backend-specific
			// sections land in the right runtime. Errors fall back to
			// gguf inside the helper, which is the safe default.
			backendKind := hf.BackendKindForModelName(modelName)

			// Preload model with options (sync - user is blocked waiting for output anyway)
			var personaOpts map[string]any
			if activePersona != nil {
				personaOpts = activePersona.GetServerOptions(backendKind)
			}
			mergedOpts := presets.MergeServerOptions(activePreset, personaOpts, backendKind)
			if ctxSizeSet || gpuLayersSet || threadsSet || mergedOpts != nil {
				opts := &server.RunOptions{
					Options: mergedOpts,
				}
				if ctxSizeSet {
					opts.CtxSize = server.IntPtr(ctxSize)
				}
				if gpuLayersSet {
					opts.GpuLayers = server.IntPtr(gpuLayers)
				}
				if threadsSet {
					opts.Threads = server.IntPtr(threads)
				}
				if err := api.Run(modelName, opts); err != nil {
					ui.Fatal("Failed to load model: %v", err)
				}
			}

			session := NewChatSession(api, modelName, cfg, activePersona, activePreset)
			session.SetSystemPrompt(systemPrompt)
			session.SetSamplingOptions(temperature, topP, minP, repeatPenalty, presencePenalty, frequencyPenalty, topK, tokens)
			if err := session.Run(promptArg); err != nil {
				ui.Fatal("Chat failed: %v", err)
			}
			return
		}

		// Launch TUI for interactive mode
		m := chat.New(api, modelName, cfg, activePersona, activePreset, personaName)
		m.SetInitialServerOptions(ctxSize, gpuLayers, threads, ctxSizeSet, gpuLayersSet, threadsSet)
		m.SetSamplingOptions(temperature, topP, minP, repeatPenalty, presencePenalty, frequencyPenalty, topK, tokens)
		m.SetSystemPrompt(systemPrompt)

		p := tea.NewProgram(m, tea.WithAltScreen())
		m.SetProgram(p)

		if _, err := p.Run(); err != nil {
			ui.Fatal("TUI error: %v", err)
		}
	},
}

// ensureLlamaInstalled installs llama.cpp if not present
func ensureLlamaInstalled() error {
	fmt.Println("Installing llama.cpp...")
	fmt.Println()
	_, err := llama.InstallLatest(func(msg string) { fmt.Println(msg) })
	if err != nil {
		return fmt.Errorf("failed to install llama.cpp: %w", err)
	}

	fmt.Println()
	return nil
}

// validateModel checks if a model exists, offering to pull it if not found
func validateModel(query string, cfg *config.Config) (*proxy.DownloadedModel, error) {
	resolver := proxy.NewModelResolver()
	result, err := resolver.Resolve(query)
	if err != nil {
		return nil, err
	}

	// Model resolved successfully
	if result.Model != nil {
		return result.Model, nil
	}

	// Ambiguous match - user needs to be more specific
	if len(result.Matches) > 1 {
		var b strings.Builder
		fmt.Fprintf(&b, "'%s' matches multiple models:\n\n", query)
		for _, m := range result.Matches {
			fmt.Fprintf(&b, "  %s\n", m.FullName)
		}
		b.WriteString("\nSpecify the full model name to continue")
		return nil, fmt.Errorf("%s", b.String())
	}

	// Model not found locally - check if it looks like a HuggingFace ref
	user, repo, quant, parseErr := parseModelRef(query)
	if parseErr != nil {
		// Not a valid model ref format, show suggestions
		return nil, modelNotFoundError(query, result.Suggestions)
	}

	// Try to pull from HuggingFace
	pulledModel, err := offerToPull(cfg, user, repo, quant)
	if err != nil {
		return nil, err
	}

	return pulledModel, nil
}

// modelNotFoundError returns a helpful error for models that aren't found
func modelNotFoundError(query string, suggestions []proxy.DownloadedModel) error {
	var b strings.Builder
	fmt.Fprintf(&b, "no model matches '%s'", query)
	if len(suggestions) > 0 {
		b.WriteString("\n\nDid you mean:\n")
		for _, s := range suggestions {
			fmt.Fprintf(&b, "  %s\n", s.FullName)
		}
	} else {
		b.WriteString("\n\n  Use 'lleme list' to see downloaded models\n  Use 'lleme search <query>' to find models")
	}
	return fmt.Errorf("%s", b.String())
}

// selectQuant returns the best matching quantization from the available list.
// If quant is empty, it picks the best available; otherwise it validates the
// requested quantization exists.
func selectQuant(quants []hf.Quantization, quant string) (hf.Quantization, error) {
	if quant == "" {
		name := hf.GetBestQuantization(quants)
		q, found := hf.FindQuantization(quants, name)
		if !found {
			return hf.Quantization{}, fmt.Errorf("internal error: best quantization %q not found in list", name)
		}
		return q, nil
	}
	if q, found := hf.FindQuantization(quants, quant); found {
		return q, nil
	}
	var b strings.Builder
	fmt.Fprintf(&b, "quantization '%s' not found\n\nAvailable:\n", quant)
	for _, q := range hf.SortQuantizations(quants) {
		fmt.Fprintf(&b, "  %s (%s)\n", q.Name, ui.FormatBytes(q.Size))
	}
	return hf.Quantization{}, fmt.Errorf("%s", b.String())
}

// offerToPull checks HuggingFace and offers to download a model
func offerToPull(cfg *config.Config, user, repo, quant string) (*proxy.DownloadedModel, error) {
	client := hf.NewClient(cfg)

	// Check if model exists on HuggingFace
	modelInfo, err := client.GetModel(user, repo)
	if err != nil {
		if strings.Contains(err.Error(), "404") {
			return nil, fmt.Errorf("model '%s/%s' not found on Hugging Face\n\n  Use 'lleme search <query>' to find models", user, repo)
		}
		return nil, fmt.Errorf("failed to check model: %w", err)
	}

	// Check for gated models
	if bool(modelInfo.Gated) && !hf.HasToken(cfg) {
		return nil, fmt.Errorf("model '%s/%s' requires authentication\n\n  Get a token at https://huggingface.co/settings/tokens\n  Then run: hf auth login", user, repo)
	}

	// Get available quantizations
	files, err := client.ListFiles(user, repo, "main")
	if err != nil {
		return nil, fmt.Errorf("failed to list model files: %w", err)
	}

	quants := hf.ExtractQuantizations(files)
	if len(quants) == 0 {
		return nil, fmt.Errorf("'%s/%s' contains no GGUF files", user, repo)
	}

	selectedQuant, err := selectQuant(quants, quant)
	if err != nil {
		return nil, err
	}

	// Get manifest info for display (also returns manifest to pass to PullModel)
	info, manifest, manifestJSON, err := hf.GetManifestInfo(client, user, repo, selectedQuant)
	if err != nil {
		return nil, err
	}

	// Download the model
	modelName := hf.FormatModelName(user, repo, selectedQuant.Name)
	if info.IsVision {
		fmt.Printf("Downloading %s (%s + %s mmproj)...\n",
			modelName, ui.FormatBytes(info.GGUFSize), ui.FormatBytes(info.MMProjSize))
	} else {
		fmt.Printf("Downloading %s (%s)...\n", modelName, ui.FormatBytes(info.GGUFSize))
	}

	// Download the model using shared download logic
	fmt.Println()
	opts := &hf.PullOptions{
		Manifest:     manifest,
		ManifestJSON: manifestJSON,
	}

	result, err := hf.PullModelWithProgressFactory(client, user, repo, selectedQuant, opts, func() hf.ProgressDisplay {
		return ui.NewProgressBar()
	})
	if err != nil {
		return nil, err
	}

	if result.IsVision {
		fmt.Printf("Pulled %s (vision model)\n", modelName)
	} else {
		fmt.Printf("Pulled %s\n", modelName)
	}
	fmt.Println()

	// Return the downloaded model info
	return &proxy.DownloadedModel{
		User:      user,
		Repo:      repo,
		Quant:     selectedQuant.Name,
		FullName:  modelName,
		ModelPath: result.ModelPath,
	}, nil
}

// ensureProxyRunning starts the proxy if not already running and returns its URL
func ensureProxyRunning(cfg *config.Config) (string, error) {
	// Check if proxy is already running
	if state := proxy.GetRunningProxyState(); state != nil {
		return fmt.Sprintf("http://%s:%d", state.Host, state.Port), nil
	}

	// Need to start proxy
	fmt.Println(ui.Muted("Starting proxy..."))

	executable, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("failed to get executable path: %w", err)
	}

	// Use config values for proxy
	host := cfg.Server.Host
	if host == "" {
		host = "127.0.0.1"
	}
	port := cfg.Server.Port
	if port == 0 {
		port = 11313
	}

	args := []string{
		"internal-serve",
		"--host", host,
		"--port", fmt.Sprintf("%d", port),
	}

	cmd := exec.Command(executable, args...)
	cmd.Env = os.Environ()

	// Redirect output to log file
	logFile := logs.ProxyLogPath()
	log, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return "", fmt.Errorf("failed to open log file: %w", err)
	}
	cmd.Stdout = log
	cmd.Stderr = log

	// Detach from parent process
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid: true,
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start proxy: %w", err)
	}

	// Wait for proxy to become ready
	proxyURL := fmt.Sprintf("http://%s:%d", host, port)
	client := &http.Client{Timeout: 2 * time.Second}

	for range 30 {
		time.Sleep(200 * time.Millisecond)
		resp, err := client.Get(proxyURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return proxyURL, nil
			}
		}
	}

	return "", fmt.Errorf("proxy did not become ready")
}

func init() {
	rootCmd.AddCommand(runCmd)

	// Sampling options (apply per-request)
	runCmd.Flags().Float64VarP(&temperature, "temp", "t", 0, "Temperature")
	runCmd.Flags().Float64Var(&topP, "top-p", 0, "Top-p sampling")
	runCmd.Flags().IntVar(&topK, "top-k", 0, "Top-k sampling")
	runCmd.Flags().Float64Var(&minP, "min-p", 0, "Min-p sampling")
	runCmd.Flags().Float64Var(&repeatPenalty, "repeat-penalty", 0, "Repeat penalty")
	runCmd.Flags().Float64Var(&presencePenalty, "presence-penalty", 0, "Presence penalty")
	runCmd.Flags().Float64Var(&frequencyPenalty, "frequency-penalty", 0, "Frequency penalty")
	runCmd.Flags().IntVarP(&tokens, "predict", "n", 0, "Max tokens to generate")
	runCmd.Flags().StringVarP(&systemPrompt, "system", "s", "", "System prompt")

	// Server options (affect model loading)
	runCmd.Flags().IntVar(&ctxSize, "ctx-size", 0, "Context size (0 = model default)")
	runCmd.Flags().IntVar(&gpuLayers, "gpu-layers", 0, "GPU layers to offload (0 = auto)")
	runCmd.Flags().IntVar(&threads, "threads", 0, "CPU threads (0 = auto)")
}
