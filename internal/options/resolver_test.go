package options

import (
	"testing"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/presets"
)

func TestResolveFloat(t *testing.T) {
	tests := []struct {
		name       string
		sessionVal float64
		persona    *config.Persona
		preset     *presets.Preset
		config     *config.Config
		key        string
		want       float64
	}{
		{
			name:       "session value takes priority",
			sessionVal: 0.5,
			persona:    &config.Persona{Options: map[string]any{"temp": 0.7}},
			preset:     &presets.Preset{Options: map[string]any{"temp": 0.8}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.9}}},
			key:        "temp",
			want:       0.5,
		},
		{
			name:       "persona value beats preset",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{"temp": 0.7}},
			preset:     &presets.Preset{Options: map[string]any{"temp": 0.8}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.9}}},
			key:        "temp",
			want:       0.7,
		},
		{
			name:       "preset value beats config",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{}},
			preset:     &presets.Preset{Options: map[string]any{"temp": 0.8}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.9}}},
			key:        "temp",
			want:       0.8,
		},
		{
			name:       "config value when session, persona, and preset are zero",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{}},
			preset:     nil,
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.9}}},
			key:        "temp",
			want:       0.9,
		},
		{
			name:       "nil persona falls back to preset",
			sessionVal: 0,
			persona:    nil,
			preset:     &presets.Preset{Options: map[string]any{"temp": 0.8}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.9}}},
			key:        "temp",
			want:       0.8,
		},
		{
			name:       "nil persona and nil preset falls back to config",
			sessionVal: 0,
			persona:    nil,
			preset:     nil,
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.9}}},
			key:        "temp",
			want:       0.9,
		},
		{
			name:       "returns zero when nothing is set",
			sessionVal: 0,
			persona:    nil,
			preset:     nil,
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{}}},
			key:        "temp",
			want:       0,
		},
		{
			name:       "handles int value in persona options",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{"temp": 1}},
			preset:     nil,
			config:     &config.Config{},
			key:        "temp",
			want:       1.0,
		},
		{
			name:       "explicit 0.0 in preset is respected (not treated as unset)",
			sessionVal: 0,
			persona:    nil,
			preset:     &presets.Preset{Options: map[string]any{"min-p": 0.0}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"min-p": 0.05}}},
			key:        "min-p",
			want:       0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewResolver(tt.persona, tt.config, tt.preset)
			got := r.ResolveFloat(tt.sessionVal, tt.key)
			if got != tt.want {
				t.Errorf("ResolveFloat() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResolveInt(t *testing.T) {
	tests := []struct {
		name       string
		sessionVal int
		persona    *config.Persona
		preset     *presets.Preset
		config     *config.Config
		key        string
		want       int
	}{
		{
			name:       "session value takes priority",
			sessionVal: 100,
			persona:    &config.Persona{Options: map[string]any{"top-k": 50}},
			preset:     &presets.Preset{Options: map[string]any{"top-k": 30}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"top-k": 40}}},
			key:        "top-k",
			want:       100,
		},
		{
			name:       "persona beats preset",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{"top-k": 50}},
			preset:     &presets.Preset{Options: map[string]any{"top-k": 20}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"top-k": 40}}},
			key:        "top-k",
			want:       50,
		},
		{
			name:       "preset beats config",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{}},
			preset:     &presets.Preset{Options: map[string]any{"top-k": 20}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"top-k": 40}}},
			key:        "top-k",
			want:       20,
		},
		{
			name:       "config value when all others zero",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{}},
			preset:     nil,
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"top-k": 40}}},
			key:        "top-k",
			want:       40,
		},
		{
			name:       "nil persona falls back to preset",
			sessionVal: 0,
			persona:    nil,
			preset:     &presets.Preset{Options: map[string]any{"top-k": 20}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"top-k": 40}}},
			key:        "top-k",
			want:       20,
		},
		{
			name:       "handles float64 value in persona options",
			sessionVal: 0,
			persona:    &config.Persona{Options: map[string]any{"top-k": 50.0}},
			preset:     nil,
			config:     &config.Config{},
			key:        "top-k",
			want:       50,
		},
		{
			name:       "explicit 0 in preset is respected (not treated as unset)",
			sessionVal: 0,
			persona:    nil,
			preset:     &presets.Preset{Options: map[string]any{"top-k": 0}},
			config:     &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"top-k": 40}}},
			key:        "top-k",
			want:       0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewResolver(tt.persona, tt.config, tt.preset)
			got := r.ResolveInt(tt.sessionVal, tt.key)
			if got != tt.want {
				t.Errorf("ResolveInt() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetConfigInt(t *testing.T) {
	tests := []struct {
		name    string
		persona *config.Persona
		preset  *presets.Preset
		config  *config.Config
		key     string
		want    int
	}{
		{
			name:    "persona value takes priority",
			persona: &config.Persona{Options: map[string]any{"ctx-size": 4096}},
			preset:  &presets.Preset{Options: map[string]any{"ctx-size": 2000}},
			config:  &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"ctx-size": 2048}}},
			key:     "ctx-size",
			want:    4096,
		},
		{
			name:    "preset beats config",
			persona: &config.Persona{Options: map[string]any{}},
			preset:  &presets.Preset{Options: map[string]any{"ctx-size": 2000}},
			config:  &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"ctx-size": 2048}}},
			key:     "ctx-size",
			want:    2000,
		},
		{
			name:    "config value when persona and preset have no value",
			persona: &config.Persona{Options: map[string]any{}},
			preset:  nil,
			config:  &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"ctx-size": 2048}}},
			key:     "ctx-size",
			want:    2048,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewResolver(tt.persona, tt.config, tt.preset)
			got, _ := r.GetConfigIntWithSource(tt.key)
			if got != tt.want {
				t.Errorf("GetConfigIntWithSource() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetConfigFloat(t *testing.T) {
	tests := []struct {
		name    string
		persona *config.Persona
		preset  *presets.Preset
		config  *config.Config
		key     string
		want    float64
	}{
		{
			name:    "persona value takes priority",
			persona: &config.Persona{Options: map[string]any{"min-p": 0.05}},
			preset:  &presets.Preset{Options: map[string]any{"min-p": 0.03}},
			config:  &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"min-p": 0.1}}},
			key:     "min-p",
			want:    0.05,
		},
		{
			name:    "preset beats config",
			persona: &config.Persona{Options: map[string]any{}},
			preset:  &presets.Preset{Options: map[string]any{"min-p": 0.03}},
			config:  &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"min-p": 0.1}}},
			key:     "min-p",
			want:    0.03,
		},
		{
			name:    "config value when persona and preset have no value",
			persona: &config.Persona{Options: map[string]any{}},
			preset:  nil,
			config:  &config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"min-p": 0.1}}},
			key:     "min-p",
			want:    0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewResolver(tt.persona, tt.config, tt.preset)
			got, _ := r.GetConfigFloatWithSource(tt.key)
			if got != tt.want {
				t.Errorf("GetConfigFloatWithSource() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestWithSourceReportsCorrectLayer verifies that the *WithSource variants
// return the layer the value actually came from, including when the value is
// an explicit zero. This is what the /set command uses to show users where a
// value originated.
func TestWithSourceReportsCorrectLayer(t *testing.T) {
	t.Run("float: persona zero beats preset and config", func(t *testing.T) {
		r := NewResolver(
			&config.Persona{Options: map[string]any{"min-p": 0.0}},
			&config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"min-p": 0.1}}},
			&presets.Preset{Options: map[string]any{"min-p": 0.05}},
		)
		v, source := r.GetConfigFloatWithSource("min-p")
		if v != 0.0 || source != "persona" {
			t.Errorf("got (%v, %q), want (0.0, \"persona\")", v, source)
		}
	})

	t.Run("int: preset zero beats config", func(t *testing.T) {
		r := NewResolver(
			nil,
			&config.Config{LlamaCpp: config.LlamaCpp{Options: map[string]any{"top-k": 40}}},
			&presets.Preset{Options: map[string]any{"top-k": 0}},
		)
		v, source := r.GetConfigIntWithSource("top-k")
		if v != 0 || source != "preset" {
			t.Errorf("got (%v, %q), want (0, \"preset\")", v, source)
		}
	})

	t.Run("missing key returns empty source", func(t *testing.T) {
		r := NewResolver(nil, &config.Config{}, nil)
		v, source := r.GetConfigFloatWithSource("nope")
		if v != 0 || source != "" {
			t.Errorf("got (%v, %q), want (0, \"\")", v, source)
		}
	})
}

// The config layer is backend-scoped: an MLX resolver reads swiftlm.options,
// a gguf resolver reads llamacpp.options, and neither cross-reads. Guards
// against regressions to the pre-fix behavior where llamacpp.options always
// won the config layer for both backends.
func TestResolver_ConfigLayerBackendScoped(t *testing.T) {
	cfg := &config.Config{
		LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.1}},
		SwiftLM:  config.SwiftLM{Options: map[string]any{"temp": 0.9}},
	}

	t.Run("mlx reads swiftlm.options", func(t *testing.T) {
		r := NewResolver(nil, cfg, nil).WithBackendKind(hf.BackendMLX)
		v, source := r.GetConfigFloatWithSource("temp")
		if v != 0.9 || source != "config" {
			t.Errorf("mlx got (%v, %q), want (0.9, \"config\")", v, source)
		}
	})

	t.Run("gguf reads llamacpp.options", func(t *testing.T) {
		r := NewResolver(nil, cfg, nil).WithBackendKind(hf.BackendGGUF)
		v, source := r.GetConfigFloatWithSource("temp")
		if v != 0.1 || source != "config" {
			t.Errorf("gguf got (%v, %q), want (0.1, \"config\")", v, source)
		}
	})

	t.Run("empty kind defaults to llamacpp (legacy callers)", func(t *testing.T) {
		r := NewResolver(nil, cfg, nil)
		v, source := r.GetConfigFloatWithSource("temp")
		if v != 0.1 || source != "config" {
			t.Errorf("legacy got (%v, %q), want (0.1, \"config\")", v, source)
		}
	})

	t.Run("mlx does not cross-read llamacpp when swiftlm missing key", func(t *testing.T) {
		mlxOnly := &config.Config{
			LlamaCpp: config.LlamaCpp{Options: map[string]any{"temp": 0.1}},
			SwiftLM:  config.SwiftLM{Options: map[string]any{}},
		}
		r := NewResolver(nil, mlxOnly, nil).WithBackendKind(hf.BackendMLX)
		v, source := r.GetConfigFloatWithSource("temp")
		if v != 0 || source != "" {
			t.Errorf("mlx must not cross-read: got (%v, %q), want (0, \"\")", v, source)
		}
	})
}
