package options

import (
	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/logs"
	"github.com/nchapman/lleme/internal/presets"
)

// Resolver resolves option values with priority: session > persona > preset > config.
//
// BackendKind selects which config-layer block to read at the bottom of the
// precedence chain. Empty or BackendGGUF reads from Config.LlamaCpp.Options;
// BackendMLX reads from Config.SwiftLM.Options. Unknown values fall back to
// LlamaCpp for safety — legacy callers that construct a Resolver without a
// kind continue to see the pre-SwiftLM behavior.
type Resolver struct {
	Persona     *config.Persona
	Preset      *presets.Preset
	Config      *config.Config
	BackendKind string
}

// NewResolver creates a new option resolver.
func NewResolver(persona *config.Persona, cfg *config.Config, preset *presets.Preset) *Resolver {
	return &Resolver{
		Persona: persona,
		Preset:  preset,
		Config:  cfg,
	}
}

// WithBackendKind sets the backend kind that drives which config block the
// resolver reads at the config layer. Returns the receiver for chaining so
// call sites stay compact. Pass hf.BackendMLX / hf.BackendGGUF; empty string
// behaves as BackendGGUF.
func (r *Resolver) WithBackendKind(kind string) *Resolver {
	r.BackendKind = kind
	return r
}

// ResolveFloat returns the first set value from: sessionVal, persona, preset, config.
// Zero is treated as "not set" for the session layer (CLI flag default).
func (r *Resolver) ResolveFloat(sessionVal float64, key string) float64 {
	if sessionVal != 0 {
		logs.Debug("Resolved option", "key", key, "value", sessionVal, "source", "session")
		return sessionVal
	}
	v, _ := r.GetConfigFloatWithSource(key)
	return v
}

// ResolveInt returns the first set value from: sessionVal, persona, preset, config.
// Zero is treated as "not set" for the session layer (CLI flag default).
func (r *Resolver) ResolveInt(sessionVal int, key string) int {
	if sessionVal != 0 {
		logs.Debug("Resolved option", "key", key, "value", sessionVal, "source", "session")
		return sessionVal
	}
	v, _ := r.GetConfigIntWithSource(key)
	return v
}

// lookupRaw returns the first value found for key across persona, preset, and config,
// along with its source label. Uses key-existence semantics so any value (including 0) is valid.
func (r *Resolver) lookupRaw(key string) (any, string) {
	if r.Persona != nil {
		if val, ok := r.Persona.Options[key]; ok {
			return val, "persona"
		}
	}
	if r.Preset != nil {
		if val, ok := r.Preset.Options[key]; ok {
			return val, "preset"
		}
	}
	if val, ok := r.configLayerOption(key); ok {
		return val, "config"
	}
	return nil, ""
}

// configLayerOption reads the config-layer option for the resolver's
// configured backend. Non-MLX kinds (empty, "gguf", anything unknown) read
// from LlamaCpp so existing call sites behave identically until they opt in.
func (r *Resolver) configLayerOption(key string) (any, bool) {
	if r.BackendKind == hf.BackendMLX {
		return r.Config.SwiftLM.GetOption(key)
	}
	return r.Config.LlamaCpp.GetOption(key)
}

// GetConfigFloatWithSource returns the first set float value from persona, preset, or config,
// along with the source label. Explicit 0.0 is valid and distinct from "not set".
func (r *Resolver) GetConfigFloatWithSource(key string) (float64, string) {
	val, source := r.lookupRaw(key)
	if val == nil {
		return 0, ""
	}
	if v, ok := asFloat64(val); ok {
		logs.Debug("Resolved option", "key", key, "value", v, "source", source)
		return v, source
	}
	return 0, ""
}

// GetConfigIntWithSource returns the first set int value from persona, preset, or config,
// along with the source label. Explicit 0 is valid and distinct from "not set".
func (r *Resolver) GetConfigIntWithSource(key string) (int, string) {
	val, source := r.lookupRaw(key)
	if val == nil {
		return 0, ""
	}
	if v, ok := asInt(val); ok {
		logs.Debug("Resolved option", "key", key, "value", v, "source", source)
		return v, source
	}
	return 0, ""
}

func asFloat64(val any) (float64, bool) {
	switch v := val.(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	}
	return 0, false
}

func asInt(val any) (int, bool) {
	switch v := val.(type) {
	case int:
		return v, true
	case int64:
		return int(v), true
	case float64:
		return int(v), true
	}
	return 0, false
}
