package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidatePersonaName(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"simple", "coder", false},
		{"with dash", "code-reviewer", false},
		{"with space", "my persona", false},
		{"empty", "", true},
		{"path separator", "a/b", true},
		{"windows separators", `a\b`, true},
		{"dot prefix", ".hidden", true},
		{"dash prefix", "-flag", true},
		{"parent traversal", "..", true},
		{"nested traversal", "a/../b", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidatePersonaName(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidatePersonaName(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
		})
	}
}

func TestPersonaSaveLoadRoundTrip(t *testing.T) {
	t.Setenv("LLEME_HOME", t.TempDir())

	persona := &Persona{
		Model:  "testuser/model-repo:Q4_K_M",
		System: "You are a concise assistant.\nAnswer in one line.",
		Options: map[string]any{
			"temp":  0.7,
			"top-k": 40,
			"min-p": 0.05,
		},
	}
	if err := SavePersonaTemplate("helper", persona); err != nil {
		t.Fatalf("SavePersonaTemplate: %v", err)
	}

	if !PersonaExists("helper") {
		t.Fatal("PersonaExists = false after save")
	}

	loaded, err := LoadPersona("helper")
	if err != nil {
		t.Fatalf("LoadPersona: %v", err)
	}
	if loaded.Model != persona.Model {
		t.Errorf("model = %q, want %q", loaded.Model, persona.Model)
	}
	// The template serializes the system prompt as a YAML literal block,
	// which preserves exactly one trailing newline.
	wantSystem := persona.System + "\n"
	if loaded.System != wantSystem {
		t.Errorf("system = %q, want %q", loaded.System, wantSystem)
	}
	for key, want := range persona.Options {
		got, ok := loaded.Options[key]
		if !ok {
			t.Fatalf("option %q missing after round trip: %+v", key, loaded.Options)
		}
		// YAML unmarshals numbers as float64 or int depending on shape.
		switch w := want.(type) {
		case float64:
			if g, ok := got.(float64); !ok || g != w {
				t.Errorf("option %q = %v (%T), want %v", key, got, got, want)
			}
		case int:
			if g, ok := got.(int); !ok || g != w {
				t.Errorf("option %q = %v (%T), want %v", key, got, got, want)
			}
		}
	}
}

func TestPersonaTemplateHasComments(t *testing.T) {
	t.Setenv("LLEME_HOME", t.TempDir())

	if err := SavePersonaTemplate("empty", &Persona{}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(PersonaPath("empty"))
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	for _, want := range []string{"# Persona: empty", "# model:", "# system:", "# options:"} {
		if !strings.Contains(content, want) {
			t.Errorf("template missing %q:\n%s", want, content)
		}
	}
	// The commented template must still parse as an empty persona.
	loaded, err := LoadPersona("empty")
	if err != nil {
		t.Fatalf("LoadPersona on template: %v", err)
	}
	if loaded.Model != "" || loaded.System != "" || len(loaded.Options) != 0 {
		t.Errorf("template comments leaked into persona: %+v", loaded)
	}
}

func TestListAndDeletePersonas(t *testing.T) {
	t.Setenv("LLEME_HOME", t.TempDir())

	if err := SavePersonaTemplate("one", &Persona{Model: "u/a:Q4"}); err != nil {
		t.Fatal(err)
	}
	if err := SavePersonaTemplate("two", &Persona{}); err != nil {
		t.Fatal(err)
	}
	// Non-YAML file must be ignored by the listing.
	if err := os.MkdirAll(PersonasPath(), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(PersonasPath(), "notes.txt"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}

	personas, err := ListPersonas()
	if err != nil {
		t.Fatalf("ListPersonas: %v", err)
	}
	if len(personas) != 2 {
		t.Fatalf("got %d personas, want 2: %+v", len(personas), personas)
	}
	byName := map[string]PersonaInfo{}
	for _, p := range personas {
		byName[p.Name] = p
	}
	if !byName["one"].HasModel || byName["one"].Model != "u/a:Q4" {
		t.Errorf("persona one model info wrong: %+v", byName["one"])
	}
	if byName["two"].HasModel {
		t.Errorf("persona two HasModel = true, want false")
	}

	if err := DeletePersona("one"); err != nil {
		t.Fatalf("DeletePersona: %v", err)
	}
	if PersonaExists("one") {
		t.Error("persona still exists after delete")
	}
	if err := DeletePersona("one"); err == nil {
		t.Error("deleting a missing persona must error")
	}
}

func TestLoadPersonaNotFound(t *testing.T) {
	t.Setenv("LLEME_HOME", t.TempDir())
	if _, err := LoadPersona("ghost"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("err = %v, want not-found error", err)
	}
}

func TestGetServerOptionsNilSafe(t *testing.T) {
	var p *Persona
	if opts := p.GetServerOptions(); opts != nil {
		t.Errorf("nil persona options = %v, want nil", opts)
	}
}
