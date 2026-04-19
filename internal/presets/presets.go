package presets

import (
	"io/fs"
	"path"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/nchapman/lleme/internal/logs"
)

// Preset holds curated inference settings for a model family.
type Preset struct {
	Name    string         `yaml:"name"`
	Source  string         `yaml:"source"`
	Match   []string       `yaml:"match"`
	Options map[string]any `yaml:"options"`

	filename string // set by Load, used for deterministic sort
}

// GetOptions returns a shallow copy of the preset's options map.
func (p *Preset) GetOptions() map[string]any {
	if p == nil || len(p.Options) == 0 {
		return nil
	}
	out := make(map[string]any, len(p.Options))
	for k, v := range p.Options {
		out[k] = v
	}
	return out
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

// MergeServerOptions builds a server options map merging preset and persona,
// with persona taking priority. Returns nil if both have no options.
func MergeServerOptions(preset *Preset, personaOpts map[string]any) map[string]any {
	presetOpts := preset.GetOptions()
	if presetOpts == nil && personaOpts == nil {
		return nil
	}
	merged := make(map[string]any)
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
