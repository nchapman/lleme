package presets

import (
	"io/fs"
	"path"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/logs"
)

// PresetBackendSection holds runtime-specific option overrides. When set,
// these layer on top of Preset.Options for the matching backend only.
type PresetBackendSection struct {
	Options map[string]any `yaml:"options,omitempty"`
}

// Preset holds curated inference settings for a model family.
//
// Options layering (per backend) is:
//
//	shared Options → backend-specific Options (later wins)
//
// so the common Qwen3 sampling defaults can live under the top-level
// `options:` and backend-specific knobs (mirostat for llama.cpp, turbo-kv
// for SwiftLM) sit under their respective sub-blocks without affecting the
// other runtime.
type Preset struct {
	Name     string               `yaml:"name"`
	Source   string               `yaml:"source"`
	Match    []string             `yaml:"match"`
	Options  map[string]any       `yaml:"options"`
	LlamaCpp PresetBackendSection `yaml:"llamacpp,omitempty"`
	SwiftLM  PresetBackendSection `yaml:"swiftlm,omitempty"`

	filename string // set by Load, used for deterministic sort
}

// GetOptions returns a shallow copy of the preset's options, layered for
// the given backend kind (hf.BackendGGUF / hf.BackendMLX). An empty kind
// applies only the shared Options map.
func (p *Preset) GetOptions(kind string) map[string]any {
	if p == nil {
		return nil
	}
	specific := p.backendOptions(kind)
	if len(p.Options) == 0 && len(specific) == 0 {
		return nil
	}
	out := make(map[string]any, len(p.Options)+len(specific))
	for k, v := range p.Options {
		out[k] = v
	}
	for k, v := range specific {
		out[k] = v
	}
	return out
}

func (p *Preset) backendOptions(kind string) map[string]any {
	switch kind {
	case hf.BackendGGUF:
		return p.LlamaCpp.Options
	case hf.BackendMLX:
		return p.SwiftLM.Options
	}
	return nil
}

// Load parses all embedded *.yaml preset files, sorted alphabetically by filename.
func Load() ([]*Preset, error) {
	fsys, err := PresetsFS()
	if err != nil {
		return nil, err
	}

	entries, err := fs.ReadDir(fsys, ".")
	if err != nil {
		return nil, err
	}

	// Collect filenames and sort for deterministic match order.
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".yaml") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)

	var presets []*Preset
	for _, name := range names {
		data, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, err
		}
		var p Preset
		if err := yaml.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		p.filename = name
		presets = append(presets, &p)
	}
	return presets, nil
}

// MergeServerOptions builds a server options map for the given backend kind.
// It layers preset options (shared then backend-specific, via GetOptions)
// under the caller-supplied personaOpts, which should already be resolved
// for the same kind via Persona.GetServerOptions. Persona keys win on
// conflict. Returns nil if both inputs are empty. kind should be
// hf.BackendGGUF or hf.BackendMLX; any other value drops the preset's
// backend-specific layer and uses its shared options only.
func MergeServerOptions(preset *Preset, personaOpts map[string]any, kind string) map[string]any {
	presetOpts := preset.GetOptions(kind)
	if presetOpts == nil && personaOpts == nil {
		return nil
	}
	merged := make(map[string]any, len(presetOpts)+len(personaOpts))
	for k, v := range presetOpts {
		merged[k] = v
	}
	for k, v := range personaOpts {
		merged[k] = v
	}
	return merged
}

// repoName strips the :quant suffix from a full HF model name, returning user/repo.
func repoName(modelName string) string {
	if i := strings.LastIndex(modelName, ":"); i != -1 {
		return modelName[:i]
	}
	return modelName
}

// Find returns the first preset (alphabetical by filename) whose match patterns
// match modelName (user/repo:quant). Returns (preset, matchedPattern) or (nil, "").
// Match is performed against the user/repo portion (quant suffix stripped) and
// is case-insensitive — both the repo name and patterns are lowercased.
func Find(modelName string) (*Preset, string) {
	presets, err := Load()
	if err != nil {
		logs.Warn("Failed to load presets", "error", err)
		return nil, ""
	}
	repo := strings.ToLower(repoName(modelName))
	for _, p := range presets {
		for _, pattern := range p.Match {
			if ok, _ := path.Match(strings.ToLower(pattern), repo); ok {
				return p, pattern
			}
		}
	}
	return nil, ""
}
