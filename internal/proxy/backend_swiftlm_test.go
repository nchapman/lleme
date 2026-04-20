package proxy

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/nchapman/lleme/internal/config"
)

func TestSwiftLMBuildArgsBaseline(t *testing.T) {
	r := NewSwiftLMRuntime(nil)
	b := &Backend{ModelName: "u/r:q", ModelPath: "/models/mlx/q4", Port: 12345}
	args := r.BuildArgs(b, "127.0.0.1")

	want := []string{"--model", "/models/mlx/q4", "--host", "127.0.0.1", "--port", "12345"}
	if !slices.Equal(args, want) {
		t.Errorf("args = %v, want %v", args, want)
	}
}

func TestSwiftLMBuildArgsMapsKnownOptions(t *testing.T) {
	r := NewSwiftLMRuntime(nil)
	b := &Backend{
		ModelPath: "/m",
		Port:      5000,
		Options: map[string]any{
			"temperature":       0.7,
			"top_p":             0.95,
			"top_k":             40,
			"min_p":             0.05,
			"repeat_penalty":    1.1,
			"max_tokens":        1024,
			"ctx_size":          8192,
			"vision":            true,
			"thinking":          false, // falsy → omitted
			"frequency_penalty": 0.1,   // unknown → dropped
			"mirostat":          2,     // unknown → dropped
		},
	}
	args := r.BuildArgs(b, "127.0.0.1")
	argStr := strings.Join(args, " ")

	for _, flag := range []string{"--temp", "--top-p", "--top-k", "--min-p", "--repeat-penalty", "--max-tokens", "--ctx-size", "--vision"} {
		if !strings.Contains(argStr, flag) {
			t.Errorf("missing expected flag %q in %v", flag, args)
		}
	}
	for _, flag := range []string{"--frequency-penalty", "--mirostat", "--thinking"} {
		if strings.Contains(argStr, flag) {
			t.Errorf("unexpected flag %q in %v", flag, args)
		}
	}
}

func TestSwiftLMBuildArgsRequestOverridesConfig(t *testing.T) {
	cfg := &config.Config{}
	cfg.SwiftLM.Options = map[string]any{"temp": 0.5, "top_k": 20}

	r := NewSwiftLMRuntime(cfg)
	b := &Backend{
		ModelPath: "/m",
		Port:      6000,
		Options:   map[string]any{"temp": 0.9}, // should win
	}
	args := r.BuildArgs(b, "127.0.0.1")

	idx := slices.Index(args, "--temp")
	if idx < 0 || idx+1 >= len(args) || args[idx+1] != "0.9" {
		t.Errorf("--temp = %v, want 0.9 (request should override config): %v", argAt(args, idx+1), args)
	}
	// top_k from config should still appear (not clobbered by the request).
	if !slices.Contains(args, "--top-k") {
		t.Errorf("--top-k from config missing: %v", args)
	}
}

func TestSwiftLMIsTruthy(t *testing.T) {
	truthy := []any{true, 1, int(5), float64(1.0), "true", "True", "1", "yes", "on"}
	falsy := []any{false, 0, float64(0), "", "false", "no", "off", "anything-else", nil}
	for _, v := range truthy {
		if !isTruthy(v) {
			t.Errorf("isTruthy(%v) = false, want true", v)
		}
	}
	for _, v := range falsy {
		if isTruthy(v) {
			t.Errorf("isTruthy(%v) = true, want false", v)
		}
	}
}

func argAt(args []string, i int) string {
	if i < 0 || i >= len(args) {
		return "<oob>"
	}
	return args[i]
}

func TestSwiftLMIsStartupError(t *testing.T) {
	tests := []struct {
		name string
		log  string
		want bool
	}{
		// ArgumentParser validation / explicit error lines.
		{"fatal Error line", "[SwiftLM] Loading model: /x\nError: File not found: main\n", true},
		{"missing arg error", "Error: Missing expected argument '--model <model>'\n", true},

		// Swift runtime aborts (OOM, forced-try, fatalError).
		{"Fatal error prefix", "Fatal error: Unexpectedly found nil\n", true},
		{"forced-try swift stack", "Swift/ErrorType.swift:200: Fatal error: 'try!' expression\n", true},
		{"thread 1 fatal", "Thread 1: Fatal error: index out of range\n", true},

		// dyld load failures (missing metallib / dylib).
		{"dyld library not loaded", "dyld[12345]: Library not loaded: @rpath/libmlx.metallib\n", true},

		// Negatives.
		{"info only", "[SwiftLM] Loading model: /ok\n[SwiftLM] Ready\n", false},
		{"empty", "", false},
		{"lowercase error not matched", "error: something\n", false},
	}
	r := NewSwiftLMRuntime(nil)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "log")
			if err := os.WriteFile(path, []byte(tt.log), 0644); err != nil {
				t.Fatal(err)
			}
			if got := r.IsStartupError(path); got != tt.want {
				t.Errorf("IsStartupError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSwiftLMHealthURL(t *testing.T) {
	r := NewSwiftLMRuntime(nil)
	if got := r.HealthURL("localhost", 7000); got != "http://localhost:7000/health" {
		t.Errorf("HealthURL = %q", got)
	}
}

func TestSwiftLMKindIsMLX(t *testing.T) {
	if k := NewSwiftLMRuntime(nil).Kind(); k != BackendKindMLX {
		t.Errorf("Kind() = %v, want mlx", k)
	}
}
