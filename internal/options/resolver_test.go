package options

import (
	"testing"

	"github.com/nchapman/lleme/internal/config"
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
			got := r.GetConfigInt(tt.key)
			if got != tt.want {
				t.Errorf("GetConfigInt() = %v, want %v", got, tt.want)
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
			got := r.GetConfigFloat(tt.key)
			if got != tt.want {
				t.Errorf("GetConfigFloat() = %v, want %v", got, tt.want)
			}
		})
	}
}
