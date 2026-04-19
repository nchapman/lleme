package proxy

import (
	"bufio"
	"fmt"
	"maps"
	"os"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/llama"
)

// LlamaRuntime implements Runtime for llama.cpp's llama-server.
type LlamaRuntime struct {
	appConfig *config.Config
}

// NewLlamaRuntime constructs a llama-server runtime. appConfig supplies the
// LlamaCpp.Options map that's merged into per-backend options at start time.
func NewLlamaRuntime(appConfig *config.Config) *LlamaRuntime {
	return &LlamaRuntime{appConfig: appConfig}
}

func (r *LlamaRuntime) Kind() BackendKind { return BackendKindLlama }

func (r *LlamaRuntime) BinaryPath() string { return llama.ServerPath() }

func (r *LlamaRuntime) WorkingDir() string { return config.BinPath() }

func (r *LlamaRuntime) HealthURL(host string, port int) string {
	return fmt.Sprintf("http://%s:%d/health", host, port)
}

func (r *LlamaRuntime) BuildArgs(backend *Backend, host string) []string {
	args := []string{
		"--model", backend.ModelPath,
		"--host", host,
		"--port", fmt.Sprintf("%d", backend.Port),
		"--embeddings", // Enable /v1/embeddings endpoint
		"--no-webui",   // Disable built-in web UI (lleme is a proxy)
	}

	// Check for mmproj file (vision model support)
	if mmprojPath := findMMProjForModel(backend.ModelName); mmprojPath != "" {
		args = append(args, "--mmproj", mmprojPath)
	}

	// Apply template patches to work around llama-server issues.
	// See template.go for the patch registry and documentation.
	if templatePath, err := ExtractAndPatchTemplate(backend.ModelPath); err == nil && templatePath != "" {
		args = append(args, "--chat-template-file", templatePath)
	}

	// Merge config options with backend-specific options (backend overrides config)
	mergedOptions := make(map[string]any)
	if r.appConfig != nil {
		maps.Copy(mergedOptions, r.appConfig.LlamaCpp.Options)
	}
	maps.Copy(mergedOptions, backend.Options)

	args = append(args, buildLlamaServerArgs(mergedOptions)...)
	return args
}

func (r *LlamaRuntime) IsStartupError(logPath string) bool {
	return hasStartupError(logPath)
}

// findMMProjForModel parses the model name and checks if an mmproj file exists.
// ModelName format: "user/repo:quant" (e.g., "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M")
func findMMProjForModel(modelName string) string {
	parts := strings.Split(modelName, ":")
	if len(parts) != 2 {
		return ""
	}

	repoRef := parts[0]
	quant := parts[1]

	repoParts := strings.Split(repoRef, "/")
	if len(repoParts) != 2 {
		return ""
	}

	user := repoParts[0]
	repo := repoParts[1]

	return hf.FindMMProjFile(user, repo, quant)
}

// buildLlamaServerArgs converts the llama_server config map to command-line arguments.
func buildLlamaServerArgs(config map[string]any) []string {
	if config == nil {
		return nil
	}

	var args []string
	for key, value := range config {
		flag := "--" + key

		switch v := value.(type) {
		case bool:
			if v {
				args = append(args, flag)
			}
			// false booleans are omitted (use default)
		case int:
			args = append(args, flag, fmt.Sprintf("%d", v))
		case float64:
			// YAML parses numbers as float64, check if it's a whole number
			if v == float64(int(v)) {
				args = append(args, flag, fmt.Sprintf("%d", int(v)))
			} else {
				args = append(args, flag, fmt.Sprintf("%g", v))
			}
		case string:
			if v != "" {
				args = append(args, flag, v)
			}
		}
	}

	return args
}

func hasStartupError(logFile string) bool {
	file, err := os.Open(logFile)
	if err != nil {
		return false
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.ToLower(scanner.Text())
		if strings.Contains(line, "error") && strings.Contains(line, "failed") {
			return true
		}
		if strings.Contains(line, "could not load model") {
			return true
		}
	}
	return false
}
