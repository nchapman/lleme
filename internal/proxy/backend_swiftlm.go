package proxy

import (
	"bufio"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/logs"
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

// SignificantOptions returns SwiftLM flags that force a reload when
// changed. Per-request sampling (temp, top-p, top-k, min-p, repeat-penalty)
// and max_tokens are overridable on each /v1/chat/completions call and
// stay off the list. Keys are kebab-case; optionsChanged normalizes inputs
// so callers can supply either form.
func (r *SwiftLMRuntime) SignificantOptions() []string {
	return []string{
		"ctx-size",
		"parallel",
		"gpu-layers",
		"mem-limit",
		"prefill-size",
		"prefill",
		"thinking",
		"vision",
		"audio",
		"stream-experts",
		"turbo-kv",
		"ssd-prefetch",
		"calibrate",
		"draft-model",
		"num-draft-tokens",
		"api-key",
	}
}

// swiftlmValueFlags maps kebab-case option keys to the CLI flag SwiftLM
// expects. Callers feed mixed casing (snake_case from YAML/JSON, kebab-case
// from CLI flags); mapSwiftLMOptions normalizes `_` → `-` before lookup so
// this table stays a single entry per flag. Confirmed against
// `SwiftLM --help` on b517.
//
// Anything not on this list (or swiftlmBoolFlags) is dropped — SwiftLM
// rejects unknown flags at startup with a bare `Error:` line, so silently
// ignoring llama-only keys (mirostat, xtc-*, dry-*, frequency-penalty,
// grammar, ...) is strictly safer than forwarding them.
var swiftlmValueFlags = map[string]string{
	// Core
	"ctx-size":     "--ctx-size",
	"max-tokens":   "--max-tokens",
	"parallel":     "--parallel",
	"gpu-layers":   "--gpu-layers",
	"mem-limit":    "--mem-limit",
	"prefill":      "--prefill-size",
	"prefill-size": "--prefill-size",
	"api-key":      "--api-key",
	"cors":         "--cors",
	// Sampling
	"temp":               "--temp",
	"temperature":        "--temp",
	"top-p":              "--top-p",
	"top-k":              "--top-k",
	"min-p":              "--min-p",
	"repeat-penalty":     "--repeat-penalty",
	"repetition-penalty": "--repeat-penalty",
	// Speculative decoding
	"draft-model":      "--draft-model",
	"num-draft-tokens": "--num-draft-tokens",
}

// swiftlmBoolFlags are SwiftLM's boolean switches. A truthy value emits the
// flag; a falsy value omits it. Keys are kebab-case (snake_case normalized).
var swiftlmBoolFlags = map[string]string{
	"thinking":       "--thinking",
	"vision":         "--vision",
	"audio":          "--audio",
	"stream-experts": "--stream-experts",
	"turbo-kv":       "--turbo-kv",
	"ssd-prefetch":   "--ssd-prefetch",
	"calibrate":      "--calibrate",
}

// TODO(backend-parity): Template patching is currently llama.cpp-only via
// ExtractAndPatchTemplate (see backend_llama.go). SwiftLM reads
// chat_template.jinja directly from the MLX repo directory with no CLI
// override, so we cannot inject a patched template today. MLX variants of
// models whose Jinja templates trip llama-server's known issues (e.g. the
// empty tools array handling in proxy/template.go) will produce degraded
// tool-call output. Revisit when SwiftLM grows a --chat-template-file flag
// or we ship an inline template-rewriting pass that writes a sibling file
// SwiftLM picks up before loading.
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
// Keys are normalized snake_case → kebab-case before lookup so the allow-list
// only needs one entry per flag.
func mapSwiftLMOptions(opts map[string]any) []string {
	var args []string
	for rawKey, value := range opts {
		key := strings.ReplaceAll(rawKey, "_", "-")
		if flag, ok := swiftlmBoolFlags[key]; ok {
			if isTruthy(value) {
				args = append(args, flag)
			}
			continue
		}
		flag, ok := swiftlmValueFlags[key]
		if !ok {
			// Log at debug so typos in swiftlm.options leave a trail
			// without polluting default output — users legitimately share
			// options between backends via preset/persona, so most drops
			// are intentional.
			logs.Debug("dropping unsupported SwiftLM option", "key", rawKey)
			continue
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

// swiftLMStartupErrorMarkers lists the shapes SwiftLM emits when it fails
// to come up. The binary is a Swift ArgumentParser app, so it prints a
// plain `Error: …` for arg-validation failures. Runtime failures — OOM via
// fatalError, a forced-try on missing weights, a bad model path — surface
// as `Fatal error:` lines (sometimes prefixed with `Swift/<file>:…` stacks
// or `Thread 1:`). Missing metallib or dylib dependencies abort via dyld
// before any model code runs, producing `dyld[...]: ... Library not
// loaded` on stderr.
//
// Ordering is significant only for marker documentation; lineIndicatesStartupError
// returns on first match. `substr` markers use strings.Contains so they catch
// variants like `Thread 1: Fatal error: ...` and `Swift/ErrorType.swift:XX:
// Fatal error: 'try!' expression …`.
type swiftLMMarker struct {
	kind string // "prefix" or "substr"
	text string
}

var swiftLMStartupErrorMarkers = []swiftLMMarker{
	{kind: "prefix", text: "Error:"},
	{kind: "prefix", text: "Fatal error:"},
	{kind: "prefix", text: "fatal error:"},
	{kind: "substr", text: "Fatal error"}, // covers "Swift/X: Fatal error" and "Thread 1: Fatal error"
	{kind: "substr", text: "Library not loaded"},
}

func lineIndicatesStartupError(line string) bool {
	for _, m := range swiftLMStartupErrorMarkers {
		switch m.kind {
		case "prefix":
			if strings.HasPrefix(line, m.text) {
				return true
			}
		case "substr":
			if strings.Contains(line, m.text) {
				return true
			}
		}
	}
	return false
}

// IsStartupError scans the backend log for any shape SwiftLM uses to
// signal a startup failure. SwiftLM prints `[SwiftLM]` progress lines
// during model load which we don't want to misread — every marker above
// is narrow enough that the progress output can't accidentally match.
func (r *SwiftLMRuntime) IsStartupError(logPath string) bool {
	file, err := os.Open(logPath)
	if err != nil {
		return false
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Default buffer caps at 64 KiB; a long path embedded in an error line
	// would otherwise silently truncate and we'd miss a fatal. 256 KiB is
	// well above anything realistic.
	scanner.Buffer(make([]byte, 64*1024), 256*1024)
	for scanner.Scan() {
		if lineIndicatesStartupError(scanner.Text()) {
			return true
		}
	}
	return false
}
