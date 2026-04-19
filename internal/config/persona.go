package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Persona represents a saved model configuration with optional system prompt and options.
type Persona struct {
	Model   string         `yaml:"model,omitempty"`
	System  string         `yaml:"system,omitempty"`
	Options map[string]any `yaml:"options,omitempty"`
}

// GetServerOptions returns all persona options to pass to the model loading API.
// Returns a shallow copy to prevent callers from mutating persona state.
func (p *Persona) GetServerOptions() map[string]any {
	if p == nil || len(p.Options) == 0 {
		return nil
	}
	out := make(map[string]any, len(p.Options))
	for k, v := range p.Options {
		out[k] = v
	}
	return out
}

const personasDir = "personas"

// ValidatePersonaName checks if a persona name is valid for use as a filename.
func ValidatePersonaName(name string) error {
	if name == "" {
		return fmt.Errorf("persona name cannot be empty")
	}
	if strings.ContainsAny(name, `/\:*?"<>|`) {
		return fmt.Errorf("persona name contains invalid characters")
	}
	if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "-") {
		return fmt.Errorf("persona name cannot start with '.' or '-'")
	}
	if strings.Contains(name, "..") {
		return fmt.Errorf("persona name cannot contain '..'")
	}
	return nil
}

// PersonasPath returns the path to the personas directory.
func PersonasPath() string {
	return filepath.Join(BaseDir(), personasDir)
}

// PersonaPath returns the path to a specific persona file.
func PersonaPath(name string) string {
	return filepath.Join(PersonasPath(), name+".yaml")
}

// LoadPersona loads a persona by name.
func LoadPersona(name string) (*Persona, error) {
	path := PersonaPath(name)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("persona '%s' not found", name)
		}
		return nil, fmt.Errorf("failed to read persona: %w", err)
	}

	var persona Persona
	if err := yaml.Unmarshal(data, &persona); err != nil {
		return nil, fmt.Errorf("failed to parse persona: %w", err)
	}

	return &persona, nil
}

// SavePersonaTemplate saves a persona with helpful comments.
func SavePersonaTemplate(name string, persona *Persona) error {
	if err := os.MkdirAll(PersonasPath(), 0755); err != nil {
		return fmt.Errorf("failed to create personas directory: %w", err)
	}

	var b strings.Builder
	b.WriteString("# Persona: " + name + "\n")
	b.WriteString("#\n")
	b.WriteString("# Run with: lleme run " + name + "\n\n")

	writePersonaModelSection(&b, persona)
	writePersonaSystemSection(&b, persona)
	if err := writePersonaOptionsSection(&b, persona); err != nil {
		return err
	}

	path := PersonaPath(name)
	if err := os.WriteFile(path, []byte(b.String()), 0644); err != nil {
		return fmt.Errorf("failed to write persona: %w", err)
	}

	return nil
}

func writePersonaModelSection(b *strings.Builder, persona *Persona) {
	if persona.Model != "" {
		b.WriteString("model: " + persona.Model + "\n\n")
		return
	}
	b.WriteString("# Base model (required, or specify at runtime)\n")
	b.WriteString("# model: bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M\n\n")
}

func writePersonaSystemSection(b *strings.Builder, persona *Persona) {
	if persona.System != "" {
		b.WriteString("system: |\n")
		for line := range strings.SplitSeq(persona.System, "\n") {
			b.WriteString("  " + line + "\n")
		}
		b.WriteString("\n")
		return
	}
	b.WriteString("# System prompt\n")
	b.WriteString("# system: |\n")
	b.WriteString("#   You are a helpful assistant.\n\n")
}

func writePersonaOptionsSection(b *strings.Builder, persona *Persona) error {
	b.WriteString("# llama.cpp options (same as config llamacpp.options)\n")
	b.WriteString("# options:\n")
	b.WriteString("#   temp: 0.8\n")
	b.WriteString("#   top-p: 0.9\n")
	b.WriteString("#   top-k: 40\n")
	b.WriteString("#   repeat-penalty: 1.0\n")

	if len(persona.Options) == 0 {
		return nil
	}

	b.WriteString("\noptions:\n")
	optData, err := yaml.Marshal(persona.Options)
	if err != nil {
		return fmt.Errorf("failed to marshal options: %w", err)
	}
	for line := range strings.SplitSeq(string(optData), "\n") {
		if line != "" {
			b.WriteString("  " + line + "\n")
		}
	}
	return nil
}

// DeletePersona removes a persona by name.
func DeletePersona(name string) error {
	path := PersonaPath(name)
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("persona '%s' not found", name)
		}
		return fmt.Errorf("failed to delete persona: %w", err)
	}
	return nil
}

// PersonaInfo holds metadata about a persona.
type PersonaInfo struct {
	Name     string
	Path     string
	HasModel bool
	Model    string
}

// ListPersonas returns all available personas.
func ListPersonas() ([]PersonaInfo, error) {
	dir := PersonasPath()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return []PersonaInfo{}, nil
		}
		return nil, fmt.Errorf("failed to read personas directory: %w", err)
	}

	var personas []PersonaInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}

		name := strings.TrimSuffix(entry.Name(), ".yaml")
		info := PersonaInfo{
			Name: name,
			Path: filepath.Join(dir, entry.Name()),
		}

		// Try to load to get model info
		if p, err := LoadPersona(name); err == nil {
			info.HasModel = p.Model != ""
			info.Model = p.Model
		}

		personas = append(personas, info)
	}

	return personas, nil
}

// PersonaExists checks if a persona with the given name exists.
func PersonaExists(name string) bool {
	_, err := os.Stat(PersonaPath(name))
	return err == nil
}
