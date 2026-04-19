package options

import (
	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/logs"
	"github.com/nchapman/lleme/internal/presets"
)

// Resolver resolves option values with priority: session > persona > preset > config.
type Resolver struct {
	Persona *config.Persona
	Preset  *presets.Preset
	Config  *config.Config
}

// NewResolver creates a new option resolver.
func NewResolver(persona *config.Persona, cfg *config.Config, preset *presets.Preset) *Resolver {
	return &Resolver{
		Persona: persona,
		Preset:  preset,
		Config:  cfg,
	}
}

// ResolveFloat returns the first set value from: sessionVal, persona, preset, config.
// Zero is treated as "not set" for the session layer (CLI flag default).
func (r *Resolver) ResolveFloat(sessionVal float64, key string) float64 {
	if sessionVal != 0 {
		logs.Debug("Resolved option", "key", key, "value", sessionVal, "source", "session")
		return sessionVal
	}
	return r.GetConfigFloat(key)
}

// ResolveInt returns the first set value from: sessionVal, persona, preset, config.
// Zero is treated as "not set" for the session layer (CLI flag default).
func (r *Resolver) ResolveInt(sessionVal int, key string) int {
	if sessionVal != 0 {
		logs.Debug("Resolved option", "key", key, "value", sessionVal, "source", "session")
		return sessionVal
	}
	return r.GetConfigInt(key)
}

// GetConfigInt returns the first set value from: persona, preset, config.
func (r *Resolver) GetConfigInt(key string) int {
	v, _ := r.GetConfigIntWithSource(key)
	return v
}

// GetConfigFloat returns the first set value from: persona, preset, config.
func (r *Resolver) GetConfigFloat(key string) float64 {
	v, _ := r.GetConfigFloatWithSource(key)
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
	if val, ok := r.Config.LlamaCpp.GetOption(key); ok {
		return val, "config"
	}
	return nil, ""
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
