package proxy

import (
	"reflect"
	"strings"
	"testing"
)

func TestBuildLlamaServerArgs(t *testing.T) {
	tests := []struct {
		name     string
		config   map[string]any
		expected map[string]string // flag -> value ("" for boolean flags)
	}{
		{
			name:     "nil config",
			config:   nil,
			expected: map[string]string{},
		},
		{
			name:     "empty config",
			config:   map[string]any{},
			expected: map[string]string{},
		},
		{
			name: "boolean true",
			config: map[string]any{
				"mlock": true,
			},
			expected: map[string]string{"mlock": ""},
		},
		{
			name: "boolean false omitted",
			config: map[string]any{
				"mlock": false,
			},
			expected: map[string]string{},
		},
		{
			name: "integer value",
			config: map[string]any{
				"parallel": 4,
			},
			expected: map[string]string{"parallel": "4"},
		},
		{
			name: "float64 whole number as int",
			config: map[string]any{
				"threads": float64(8),
			},
			expected: map[string]string{"threads": "8"},
		},
		{
			name: "float64 decimal",
			config: map[string]any{
				"temp": 0.7,
			},
			expected: map[string]string{"temp": "0.7"},
		},
		{
			name: "string value",
			config: map[string]any{
				"flash-attn": "off",
			},
			expected: map[string]string{"flash-attn": "off"},
		},
		{
			name: "empty string omitted",
			config: map[string]any{
				"api-key": "",
			},
			expected: map[string]string{},
		},
		{
			name: "multiple options",
			config: map[string]any{
				"parallel": float64(4),
				"mlock":    true,
				"threads":  float64(8),
			},
			expected: map[string]string{
				"parallel": "4",
				"mlock":    "",
				"threads":  "8",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildLlamaServerArgs(tt.config)
			resultMap := parseArgsToMap(result)

			if !reflect.DeepEqual(resultMap, tt.expected) {
				t.Errorf("buildLlamaServerArgs() = %v (parsed: %v), want %v", result, resultMap, tt.expected)
			}
		})
	}
}

// parseArgsToMap converts ["--flag", "value", "--bool"] to {"flag": "value", "bool": ""}
func parseArgsToMap(args []string) map[string]string {
	m := make(map[string]string)
	for i := 0; i < len(args); i++ {
		if key, ok := strings.CutPrefix(args[i], "--"); ok {
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				m[key] = args[i+1]
				i++
			} else {
				m[key] = "" // boolean flag
			}
		}
	}
	return m
}

func TestSetStateChangeCallback(t *testing.T) {
	cfg := DefaultConfig()
	manager := NewModelManager(cfg, nil)

	callCount := 0
	manager.SetStateChangeCallback(func() {
		callCount++
	})

	// Verify callback is set (we can't trigger it without starting a real backend,
	// but we can verify the mechanism works)
	if manager.onStateChange == nil {
		t.Error("expected onStateChange callback to be set")
	}

	// Call it directly to verify it works
	manager.onStateChange()
	if callCount != 1 {
		t.Errorf("expected callback to be called once, got %d", callCount)
	}
}

func TestOptionsChanged(t *testing.T) {
	tests := []struct {
		name     string
		current  map[string]any
		new      map[string]any
		expected bool
	}{
		{
			name:     "both nil",
			current:  nil,
			new:      nil,
			expected: false,
		},
		{
			name:     "new nil",
			current:  map[string]any{"ctx-size": 4096},
			new:      nil,
			expected: false,
		},
		{
			name:     "new empty",
			current:  map[string]any{"ctx-size": 4096},
			new:      map[string]any{},
			expected: false,
		},
		{
			name:     "current nil new has values",
			current:  nil,
			new:      map[string]any{"ctx-size": 4096},
			expected: true,
		},
		{
			name:     "same values",
			current:  map[string]any{"ctx-size": 4096},
			new:      map[string]any{"ctx-size": 4096},
			expected: false,
		},
		{
			name:     "different values",
			current:  map[string]any{"ctx-size": 4096},
			new:      map[string]any{"ctx-size": 8192},
			expected: true,
		},
		{
			name:     "int vs float64 same value",
			current:  map[string]any{"ctx-size": 4096},
			new:      map[string]any{"ctx-size": float64(4096)},
			expected: false,
		},
		{
			name:     "new key added",
			current:  map[string]any{"ctx-size": 4096},
			new:      map[string]any{"ctx-size": 4096, "gpu-layers": 35},
			expected: true,
		},
		{
			name:     "non-server option ignored",
			current:  map[string]any{"ctx-size": 4096},
			new:      map[string]any{"ctx-size": 4096, "custom-option": "value"},
			expected: false,
		},
	}

	rt := NewLlamaRuntime(nil)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := optionsChanged(rt, tt.current, tt.new)
			if result != tt.expected {
				t.Errorf("optionsChanged(%v, %v) = %v, want %v", tt.current, tt.new, result, tt.expected)
			}
		})
	}
}

// Each runtime has its own reload-worthy key list. A SwiftLM-only flag
// change must trigger a reload when the backend is SwiftLM, and NOT when
// the backend is llama (which ignores it).
func TestOptionsChangedPerRuntime(t *testing.T) {
	cur := map[string]any{"turbo-kv": false}
	next := map[string]any{"turbo-kv": true}

	if !optionsChanged(NewSwiftLMRuntime(nil), cur, next) {
		t.Error("SwiftLM: turbo-kv change should trigger a reload")
	}
	if optionsChanged(NewLlamaRuntime(nil), cur, next) {
		t.Error("llama: turbo-kv is not a llama option; should not trigger a reload")
	}

	// Symmetric check: mirostat matters to llama, not SwiftLM.
	cur = map[string]any{"flash-attn": "auto"}
	next = map[string]any{"flash-attn": "on"}
	if !optionsChanged(NewLlamaRuntime(nil), cur, next) {
		t.Error("llama: flash-attn change should trigger a reload")
	}
	if optionsChanged(NewSwiftLMRuntime(nil), cur, next) {
		t.Error("SwiftLM: flash-attn is not a SwiftLM option; should not trigger a reload")
	}
}

func TestOptionValuesEqual(t *testing.T) {
	tests := []struct {
		name string
		a, b any
		want bool
	}{
		{"int equal", 42, 42, true},
		{"int not equal", 42, 43, false},
		{"int vs float64 equal", 42, float64(42), true},
		{"float64 equal", 3.14, 3.14, true},
		{"float64 not equal", 3.14, 3.15, false},
		{"string equal", "test", "test", true},
		{"string not equal", "test", "other", false},
		{"bool equal", true, true, true},
		{"bool not equal", true, false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := optionValuesEqual(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("optionValuesEqual(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestToFloat64(t *testing.T) {
	tests := []struct {
		name   string
		input  any
		want   float64
		wantOk bool
	}{
		{"int", 42, 42.0, true},
		{"int64", int64(42), 42.0, true},
		{"float64", 3.14, 3.14, true},
		{"float32", float32(3.14), float64(float32(3.14)), true},
		{"string", "42", 0, false},
		{"bool", true, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := toFloat64(tt.input)
			if ok != tt.wantOk {
				t.Errorf("toFloat64(%v) ok = %v, want %v", tt.input, ok, tt.wantOk)
			}
			if ok && got != tt.want {
				t.Errorf("toFloat64(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}
