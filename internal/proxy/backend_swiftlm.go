package proxy

import (
	"bufio"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/swiftlm"
)

// SwiftLMRuntime implements Runtime for SharpAI/SwiftLM — the Apple-Silicon
// MLX server. Its CLI surface is much narrower than llama-server's, so we
// translate a known set of keys and silently drop the rest rather than pass
// everything through like the llama runtime does. Until Commit 7's per-
// backend FilterSamplingKeys lands, this local allow-list is the guard that
// keeps llama-specific keys from reaching SwiftLM as invalid flags.
type SwiftLMRuntime struct {
	appConfig *config.Config
}

func NewSwiftLMRuntime(appConfig *config.Config) *SwiftLMRuntime {
	return &SwiftLMRuntime{appConfig: appConfig}
}

func (r *SwiftLMRuntime) Kind() BackendKind { return BackendKindMLX }

func (r *SwiftLMRuntime) HFAppName() string { return "mlx-lm" }

func (r *SwiftLMRuntime) BinaryPath() string { return swiftlm.ServerPath() }

// WorkingDir points at the versioned SwiftLM directory so the binary finds
// its sibling mlx.metallib at runtime.
func (r *SwiftLMRuntime) WorkingDir() string {
	return filepath.Dir(swiftlm.ServerPath())
}

func (r *SwiftLMRuntime) HealthURL(host string, port int) string {
	return fmt.Sprintf("http://%s:%d/health", host, port)
}

// swiftlmValueFlags maps an option key to the CLI flag SwiftLM expects.
// Keys are the snake_case / kebab-case forms our options pipeline produces
// (CLI flags → config → preset → persona → request body); values are
// SwiftLM's own flag names, confirmed against `SwiftLM --help` on b517.
//
// Anything not on this list is dropped — SwiftLM rejects unknown flags at
// startup with a bare `Error:` line, so silently ignoring llama-only keys
// (mirostat, xtc_*, dry_*, frequency_penalty, grammar, ...) is strictly
// safer than forwarding them.
var swiftlmValueFlags = map[string]string{
	// Core
	"ctx_size":     "--ctx-size",
	"ctx-size":     "--ctx-size",
	"max_tokens":   "--max-tokens",
	"max-tokens":   "--max-tokens",
	"parallel":     "--parallel",
	"gpu_layers":   "--gpu-layers",
	"gpu-layers":   "--gpu-layers",
	"mem_limit":    "--mem-limit",
	"mem-limit":    "--mem-limit",
	"prefill":      "--prefill-size",
	"prefill_size": "--prefill-size",
	"prefill-size": "--prefill-size",
	"api_key":      "--api-key",
	"api-key":      "--api-key",
	"cors":         "--cors",
	// Sampling
	"temp":               "--temp",
	"temperature":        "--temp",
	"top_p":              "--top-p",
	"top-p":              "--top-p",
	"top_k":              "--top-k",
	"top-k":              "--top-k",
	"min_p":              "--min-p",
	"min-p":              "--min-p",
	"repeat_penalty":     "--repeat-penalty",
	"repeat-penalty":     "--repeat-penalty",
	"repetition_penalty": "--repeat-penalty",
	// Speculative decoding
	"draft_model":      "--draft-model",
	"draft-model":      "--draft-model",
	"num_draft_tokens": "--num-draft-tokens",
	"num-draft-tokens": "--num-draft-tokens",
}

// swiftlmBoolFlags are SwiftLM's boolean switches. A truthy value emits the
// flag; a falsy value omits it.
var swiftlmBoolFlags = map[string]string{
	"thinking":       "--thinking",
	"vision":         "--vision",
	"audio":          "--audio",
	"stream_experts": "--stream-experts",
	"stream-experts": "--stream-experts",
	"turbo_kv":       "--turbo-kv",
	"turbo-kv":       "--turbo-kv",
	"ssd_prefetch":   "--ssd-prefetch",
	"ssd-prefetch":   "--ssd-prefetch",
	"calibrate":      "--calibrate",
}

func (r *SwiftLMRuntime) BuildArgs(backend *Backend, host string) []string {
	args := []string{
		"--model", backend.ModelPath,
		"--host", host,
		"--port", fmt.Sprintf("%d", backend.Port),
	}

	// Merge config defaults with per-request options (request wins). Config
	// lives under a SwiftLM-specific key so llama-server options don't bleed
	// in; the key is optional, so missing config is fine.
	merged := make(map[string]any)
	if r.appConfig != nil {
		maps.Copy(merged, r.appConfig.SwiftLM.Options)
	}
	maps.Copy(merged, backend.Options)

	args = append(args, mapSwiftLMOptions(merged)...)
	return args
}

// mapSwiftLMOptions turns an option map into SwiftLM CLI args, dropping any
// key that isn't on the allow-list. Value handling mirrors the llama
// runtime: int/float/string formatted verbatim, bool emits the flag if true.
func mapSwiftLMOptions(opts map[string]any) []string {
	var args []string
	for key, value := range opts {
		if flag, ok := swiftlmBoolFlags[key]; ok {
			if isTruthy(value) {
				args = append(args, flag)
			}
			continue
		}
		flag, ok := swiftlmValueFlags[key]
		if !ok {
			continue // silently drop unsupported keys
		}
		switch v := value.(type) {
		case int:
			args = append(args, flag, fmt.Sprintf("%d", v))
		case float64:
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

// isTruthy accepts the common boolean representations YAML/JSON produce.
// Strict bool-only matching would silently drop `thinking: 1` or
// `vision: "true"` — surprising for users who've copy-pasted from other
// configs or request bodies.
func isTruthy(v any) bool {
	switch val := v.(type) {
	case bool:
		return val
	case int:
		return val != 0
	case float64:
		return val != 0
	case string:
		switch val {
		case "true", "True", "TRUE", "1", "yes", "on":
			return true
		}
	}
	return false
}

// IsStartupError scans the backend log for SwiftLM's error marker. The
// binary prints a plain `Error: ...` line (no prefix) when arg parsing or
// model loading fails, and keeps running otherwise. We only treat an Error
// line as fatal if it appears before any serving-ready marker — SwiftLM
// prints `[SwiftLM]` progress lines during model load that we don't want to
// misread. Today any Error: line is fatal (server exits), so the simple
// scan is correct; revisit if SwiftLM grows non-fatal Error: logs.
func (r *SwiftLMRuntime) IsStartupError(logPath string) bool {
	file, err := os.Open(logPath)
	if err != nil {
		return false
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Default buffer caps at 64 KiB; a long path embedded in an Error: line
	// would otherwise silently truncate and we'd miss a fatal. 256 KiB is
	// well above anything realistic.
	scanner.Buffer(make([]byte, 64*1024), 256*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "Error:") {
			return true
		}
	}
	return false
}
