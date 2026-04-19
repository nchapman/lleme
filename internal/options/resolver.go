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

// ResolveFloat returns the first non-zero value from: sessionVal, persona, preset, config.
// Zero is treated as "not set".
func (r *Resolver) ResolveFloat(sessionVal float64, key string) float64 {
	if sessionVal != 0 {
		logs.Debug("Resolved option", "key", key, "value", sessionVal, "source", "session")
		return sessionVal
	}
	return r.GetConfigFloat(key)
}

// ResolveInt returns the first non-zero value from: sessionVal, persona, preset, config.
func (r *Resolver) ResolveInt(sessionVal int, key string) int {
	if sessionVal != 0 {
		logs.Debug("Resolved option", "key", key, "value", sessionVal, "source", "session")
		return sessionVal
	}
	return r.GetConfigInt(key)
}

// GetConfigInt returns the first non-zero value from: persona, preset, config.
func (r *Resolver) GetConfigInt(key string) int {
	v, _ := r.GetConfigIntWithSource(key)
	return v
}

// GetConfigFloat returns the first non-zero value from: persona, preset, config.
func (r *Resolver) GetConfigFloat(key string) float64 {
	v, _ := r.GetConfigFloatWithSource(key)
	return v
}

// GetConfigFloatWithSource returns the first non-zero value from persona, preset, or config,
// along with the source label for display (e.g. "persona", "preset", "config").
func (r *Resolver) GetConfigFloatWithSource(key string) (float64, string) {
	if r.Persona != nil {
		if v := r.Persona.GetFloatOption(key, 0); v != 0 {
			logs.Debug("Resolved option", "key", key, "value", v, "source", "persona")
			return v, "persona"
		}
	}
	if r.Preset != nil {
		if v := r.Preset.GetFloatOption(key, 0); v != 0 {
			logs.Debug("Resolved option", "key", key, "value", v, "source", "preset")
			return v, "preset"
		}
	}
	v := r.Config.LlamaCpp.GetFloatOption(key, 0)
	if v != 0 {
		logs.Debug("Resolved option", "key", key, "value", v, "source", "config")
		return v, "config"
	}
	return 0, ""
}

// GetConfigIntWithSource returns the first non-zero value from persona, preset, or config,
// along with the source label for display (e.g. "persona", "preset", "config").
func (r *Resolver) GetConfigIntWithSource(key string) (int, string) {
	if r.Persona != nil {
		if val, ok := r.Persona.Options[key]; ok {
			switch v := val.(type) {
			case int:
				logs.Debug("Resolved option", "key", key, "value", v, "source", "persona")
				return v, "persona"
			case float64:
				logs.Debug("Resolved option", "key", key, "value", int(v), "source", "persona")
				return int(v), "persona"
			}
		}
	}
	if r.Preset != nil {
		if v := r.Preset.GetIntOption(key, 0); v != 0 {
			logs.Debug("Resolved option", "key", key, "value", v, "source", "preset")
			return v, "preset"
		}
	}
	v := r.Config.LlamaCpp.GetIntOption(key, 0)
	if v != 0 {
		logs.Debug("Resolved option", "key", key, "value", v, "source", "config")
		return v, "config"
	}
	return 0, ""
}
