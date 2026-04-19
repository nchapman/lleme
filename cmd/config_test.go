package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nchapman/lleme/internal/config"
	"gopkg.in/yaml.v3"
)

func TestGetEditor(t *testing.T) {
	// Save original env vars
	origVisual := os.Getenv("VISUAL")
	origEditor := os.Getenv("EDITOR")
	defer func() {
		os.Setenv("VISUAL", origVisual)
		os.Setenv("EDITOR", origEditor)
	}()

	t.Run("prefers VISUAL over EDITOR", func(t *testing.T) {
		os.Setenv("VISUAL", "code")
		os.Setenv("EDITOR", "vim")

		editor := getEditor()
		if editor != "code" {
			t.Errorf("Expected 'code', got '%s'", editor)
		}
	})

	t.Run("uses EDITOR when VISUAL not set", func(t *testing.T) {
		os.Setenv("VISUAL", "")
		os.Setenv("EDITOR", "nano")

		editor := getEditor()
		if editor != "nano" {
			t.Errorf("Expected 'nano', got '%s'", editor)
		}
	})

	t.Run("falls back to common editors when env vars not set", func(t *testing.T) {
		os.Setenv("VISUAL", "")
		os.Setenv("EDITOR", "")

		editor := getEditor()
		// Should find at least one of nano, vim, vi on most systems
		if editor == "" {
			t.Skip("No fallback editor found on this system")
		}
		// Verify it found one of the expected fallbacks
		validEditors := []string{"nano", "vim", "vi"}
		found := false
		for _, valid := range validEditors {
			if strings.HasSuffix(editor, valid) || editor == valid {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected one of %v, got '%s'", validEditors, editor)
		}
	})

	t.Run("returns empty when no editor available", func(t *testing.T) {
		os.Setenv("VISUAL", "")
		os.Setenv("EDITOR", "")
		// Save PATH and set to empty to simulate no editors
		origPath := os.Getenv("PATH")
		os.Setenv("PATH", "")
		defer os.Setenv("PATH", origPath)

		editor := getEditor()
		if editor != "" {
			t.Errorf("Expected empty string when no editors available, got '%s'", editor)
		}
	})
}

func TestPrintConfigOutput(t *testing.T) {
	// Test that config can be loaded and marshaled to YAML
	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	data, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("Failed to marshal config: %v", err)
	}

	// Verify it's valid YAML by unmarshaling back
	var parsed config.Config
	if err := yaml.Unmarshal(data, &parsed); err != nil {
		t.Errorf("Config output is not valid YAML: %v", err)
	}

	// Verify key fields are present in output
	output := string(data)
	expectedFields := []string{
		"huggingface:",
		"llamacpp:",
		"server:",
	}
	for _, field := range expectedFields {
		if !strings.Contains(output, field) {
			t.Errorf("Expected config output to contain '%s'", field)
		}
	}
}

func TestConfigPath(t *testing.T) {
	path := config.ConfigPath()

	if path == "" {
		t.Error("Expected non-empty config path")
	}

	if !strings.HasSuffix(path, "config.yaml") {
		t.Errorf("Expected path to end with 'config.yaml', got '%s'", path)
	}

	if !strings.Contains(path, ".lleme") {
		t.Errorf("Expected path to contain '.lleme', got '%s'", path)
	}
}

func TestOpenInEditorCreatesDefaultConfig(t *testing.T) {
	// Create temp directory
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	configPath := filepath.Join(tmpDir, ".lleme", "config.yaml")

	// Verify config doesn't exist
	if _, err := os.Stat(configPath); !os.IsNotExist(err) {
		t.Fatal("Config file should not exist initially")
	}

	// We can't actually test openInEditor without opening an editor,
	// but we can test the config creation logic separately
	cfg := config.DefaultConfig()
	if err := config.Save(cfg); err != nil {
		t.Fatalf("Failed to save default config: %v", err)
	}

	// Verify config was created
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Error("Config file should have been created")
	}

	// Verify it contains valid config
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read config: %v", err)
	}

	var loaded config.Config
	if err := yaml.Unmarshal(data, &loaded); err != nil {
		t.Errorf("Created config is not valid YAML: %v", err)
	}

	// Verify default values
	if loaded.HuggingFace.DefaultQuant != "Q4_K_M" {
		t.Errorf("Expected default_quant Q4_K_M, got %s", loaded.HuggingFace.DefaultQuant)
	}
}

func TestConfigSubcommands(t *testing.T) {
	// Test that subcommands are properly registered
	cmd := configCmd

	subCmds := cmd.Commands()
	expectedCmds := []string{"edit", "show", "path", "reset"}

	for _, expected := range expectedCmds {
		found := false
		for _, sub := range subCmds {
			if sub.Name() == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected subcommand '%s' to be registered", expected)
		}
	}
}

func TestConfigYAMLRoundTrip(t *testing.T) {
	// Test that config survives a round-trip through YAML
	original := config.DefaultConfig()
	original.LlamaCpp.Options = map[string]any{
		"temp":       0.5,
		"ctx-size":   8192,
		"gpu-layers": 32,
	}

	// Marshal to YAML
	data, err := yaml.Marshal(original)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	// Unmarshal back
	var restored config.Config
	if err := yaml.Unmarshal(data, &restored); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	// Verify values preserved
	if v, ok := restored.LlamaCpp.GetOption("temp"); !ok || v != 0.5 {
		t.Errorf("temp: expected 0.5, got %v (ok=%v)", v, ok)
	}
	if v, ok := restored.LlamaCpp.GetOption("ctx-size"); !ok || v != 8192 {
		t.Errorf("ctx-size: expected 8192, got %v (ok=%v)", v, ok)
	}
	if v, ok := restored.LlamaCpp.GetOption("gpu-layers"); !ok || v != 32 {
		t.Errorf("gpu-layers: expected 32, got %v (ok=%v)", v, ok)
	}
}

func TestGetEditorWithPath(t *testing.T) {
	// Test that EDITOR with full path works
	origEditor := os.Getenv("EDITOR")
	origVisual := os.Getenv("VISUAL")
	defer func() {
		os.Setenv("EDITOR", origEditor)
		os.Setenv("VISUAL", origVisual)
	}()

	os.Setenv("VISUAL", "")
	os.Setenv("EDITOR", "/usr/bin/vim")

	editor := getEditor()
	if editor != "/usr/bin/vim" {
		t.Errorf("Expected '/usr/bin/vim', got '%s'", editor)
	}
}

func TestResetToDefaults(t *testing.T) {
	// Create temp directory
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	configPath := filepath.Join(tmpDir, ".lleme", "config.yaml")

	// Create a modified config
	cfg := config.DefaultConfig()
	cfg.LlamaCpp.Options = map[string]any{
		"temp":       0.9,
		"ctx-size":   16384,
		"gpu-layers": 99,
	}
	cfg.HuggingFace.DefaultQuant = "Q6_K"
	if err := config.Save(cfg); err != nil {
		t.Fatalf("Failed to save modified config: %v", err)
	}

	// Verify modified config was saved
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read config: %v", err)
	}
	var modified config.Config
	if err := yaml.Unmarshal(data, &modified); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}
	if modified.HuggingFace.DefaultQuant != "Q6_K" {
		t.Fatalf("Expected modified default_quant Q6_K, got %s", modified.HuggingFace.DefaultQuant)
	}

	// Reset to defaults
	resetToDefaults(configPath)

	// Verify config was reset by reading the file content
	// Note: SaveDefault writes a template with comments, so we check the file contains defaults
	data, err = os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read reset config: %v", err)
	}

	content := string(data)
	// Check for default values in template
	if !strings.Contains(content, "default_quant: Q4_K_M") {
		t.Error("Expected reset config to contain default_quant: Q4_K_M")
	}
	if !strings.Contains(content, "# threads:") {
		t.Error("Expected reset config to contain commented options")
	}
}

func TestParseValue(t *testing.T) {
	tests := []struct {
		input    string
		expected any
	}{
		// Booleans
		{"true", true},
		{"false", false},
		// Integers
		{"0", 0},
		{"42", 42},
		{"-7", -7},
		{"8192", 8192},
		// Floats (only with decimal point)
		{"0.5", 0.5},
		{"3.14", 3.14},
		{"-0.001", -0.001},
		// Strings (including things that look like numbers but aren't valid ints/floats)
		{"hello", "hello"},
		{"127.0.0.1", "127.0.0.1"}, // IP address stays string
		{"", ""},
		{"True", "True"},   // not "true"
		{"FALSE", "FALSE"}, // not "false"
		{"3.14.159", "3.14.159"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := parseValue(tt.input)
			if got != tt.expected {
				t.Errorf("parseValue(%q) = %v (%T), want %v (%T)",
					tt.input, got, got, tt.expected, tt.expected)
			}
		})
	}
}

func TestGetValueByPath(t *testing.T) {
	m := map[string]any{
		"server": map[string]any{
			"port": 11313,
			"host": "127.0.0.1",
		},
		"llamacpp": map[string]any{
			"options": map[string]any{
				"ctx-size": 8192,
				"temp":     0.8,
			},
		},
		"simple": "value",
	}

	tests := []struct {
		path     string
		expected any
		wantErr  bool
	}{
		{"server.port", 11313, false},
		{"server.host", "127.0.0.1", false},
		{"llamacpp.options.ctx-size", 8192, false},
		{"llamacpp.options.temp", 0.8, false},
		{"simple", "value", false},
		{"nonexistent", nil, true},
		{"server.nonexistent", nil, true},
		{"simple.nested", nil, true}, // simple is not a map
		{"", nil, true},              // empty path
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got, err := getValueByPath(m, tt.path)
			if tt.wantErr {
				if err == nil {
					t.Errorf("getValueByPath(%q) expected error, got nil", tt.path)
				}
				return
			}
			if err != nil {
				t.Errorf("getValueByPath(%q) unexpected error: %v", tt.path, err)
				return
			}
			if got != tt.expected {
				t.Errorf("getValueByPath(%q) = %v, want %v", tt.path, got, tt.expected)
			}
		})
	}

	// Test getting a map (for llamacpp.options)
	t.Run("get map value", func(t *testing.T) {
		got, err := getValueByPath(m, "llamacpp.options")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		gotMap, ok := got.(map[string]any)
		if !ok {
			t.Fatalf("expected map, got %T", got)
		}
		if gotMap["ctx-size"] != 8192 {
			t.Errorf("expected ctx-size 8192, got %v", gotMap["ctx-size"])
		}
	})
}

func TestSetValueByPath(t *testing.T) {
	t.Run("set existing value", func(t *testing.T) {
		m := map[string]any{
			"server": map[string]any{
				"port": 11313,
			},
		}
		err := setValueByPath(m, "server.port", 8080)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got, _ := getValueByPath(m, "server.port")
		if got != 8080 {
			t.Errorf("expected 8080, got %v", got)
		}
	})

	t.Run("create intermediate maps", func(t *testing.T) {
		m := map[string]any{
			"llamacpp": map[string]any{},
		}
		err := setValueByPath(m, "llamacpp.options.ctx-size", 8192)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got, _ := getValueByPath(m, "llamacpp.options.ctx-size")
		if got != 8192 {
			t.Errorf("expected 8192, got %v", got)
		}
	})

	t.Run("create all intermediate maps", func(t *testing.T) {
		m := map[string]any{}
		err := setValueByPath(m, "a.b.c", "value")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got, _ := getValueByPath(m, "a.b.c")
		if got != "value" {
			t.Errorf("expected 'value', got %v", got)
		}
	})

	t.Run("error when path through non-map", func(t *testing.T) {
		m := map[string]any{
			"simple": "value",
		}
		err := setValueByPath(m, "simple.nested", "value")
		if err == nil {
			t.Error("expected error when setting path through non-map")
		}
	})

	t.Run("set single-segment path", func(t *testing.T) {
		m := map[string]any{}
		err := setValueByPath(m, "key", "value")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if m["key"] != "value" {
			t.Errorf("expected 'value', got %v", m["key"])
		}
	})

	t.Run("empty path returns error", func(t *testing.T) {
		m := map[string]any{}
		err := setValueByPath(m, "", "value")
		if err == nil {
			t.Error("expected error for empty path")
		}
	})
}

func TestConfigSetGetIntegration(t *testing.T) {
	// Create temp directory for isolated config
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	// Start with default config
	cfg := config.DefaultConfig()
	if err := config.Save(cfg); err != nil {
		t.Fatalf("Failed to save initial config: %v", err)
	}

	// Test set and get for server.port
	t.Run("set and get server.port", func(t *testing.T) {
		cfg, _ := config.Load()
		m, _ := configToMap(cfg)
		setValueByPath(m, "server.port", 9999)
		newCfg, _ := mapToConfig(m)
		config.Save(newCfg)

		// Reload and verify
		loaded, _ := config.Load()
		if loaded.Server.Port != 9999 {
			t.Errorf("expected port 9999, got %d", loaded.Server.Port)
		}
	})

	// Test creating options map
	t.Run("set llamacpp.options.ctx-size", func(t *testing.T) {
		cfg, _ := config.Load()
		m, _ := configToMap(cfg)
		setValueByPath(m, "llamacpp.options.ctx-size", 8192)
		newCfg, _ := mapToConfig(m)
		config.Save(newCfg)

		// Reload and verify
		loaded, _ := config.Load()
		if v, ok := loaded.LlamaCpp.GetOption("ctx-size"); !ok || v != 8192 {
			t.Errorf("expected ctx-size 8192, got %v (ok=%v)", v, ok)
		}
	})

	// Test boolean option
	t.Run("set llamacpp.options.flash-attn", func(t *testing.T) {
		cfg, _ := config.Load()
		m, _ := configToMap(cfg)
		setValueByPath(m, "llamacpp.options.flash-attn", true)
		newCfg, _ := mapToConfig(m)
		config.Save(newCfg)

		// Reload and verify
		loaded, _ := config.Load()
		val, ok := loaded.LlamaCpp.GetOption("flash-attn")
		if !ok {
			t.Fatal("expected flash-attn to be set")
		}
		if val != true {
			t.Errorf("expected flash-attn true, got %v", val)
		}
	})
}

func TestFormatValue(t *testing.T) {
	tests := []struct {
		name     string
		input    any
		expected string
	}{
		{"int", 42, "42"},
		{"string", "hello", "hello"},
		{"bool", true, "true"},
		{"float", 3.14, "3.14"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatValue(tt.input)
			if got != tt.expected {
				t.Errorf("formatValue(%v) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}

	// Test map formatting (YAML output)
	t.Run("map", func(t *testing.T) {
		m := map[string]any{
			"key": "value",
		}
		got := formatValue(m)
		if !strings.Contains(got, "key: value") {
			t.Errorf("formatValue(map) = %q, expected to contain 'key: value'", got)
		}
	})
}

func TestConfigGetSetSubcommands(t *testing.T) {
	// Test that subcommands are properly registered
	cmd := configCmd
	subCmds := cmd.Commands()

	expectedCmds := []string{"get", "set"}
	for _, expected := range expectedCmds {
		found := false
		for _, sub := range subCmds {
			if sub.Name() == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected subcommand '%s' to be registered", expected)
		}
	}
}

func TestConfigSetSilentFailures(t *testing.T) {
	// Document current behavior: some invalid sets succeed but don't persist.
	// TODO: Consider adding validation to detect these cases.

	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	cfg := config.DefaultConfig()
	config.Save(cfg)

	t.Run("unknown field in struct is silently dropped", func(t *testing.T) {
		cfg, _ := config.Load()
		m, _ := configToMap(cfg)
		// This sets a field that doesn't exist in the Server struct
		setValueByPath(m, "server.bogus_field", 123)
		newCfg, err := mapToConfig(m)
		if err != nil {
			t.Fatalf("mapToConfig failed: %v", err)
		}
		config.Save(newCfg)

		// Verify the field was silently dropped
		loaded, _ := config.Load()
		verifyMap, _ := configToMap(loaded)
		_, err = getValueByPath(verifyMap, "server.bogus_field")
		if err == nil {
			t.Error("expected bogus_field to be dropped, but it persisted")
		}
	})

	t.Run("unknown top-level key is silently dropped", func(t *testing.T) {
		cfg, _ := config.Load()
		m, _ := configToMap(cfg)
		setValueByPath(m, "totally_fake.nested", "value")
		newCfg, _ := mapToConfig(m)
		config.Save(newCfg)

		// Verify the key was silently dropped
		loaded, _ := config.Load()
		verifyMap, _ := configToMap(loaded)
		_, err := getValueByPath(verifyMap, "totally_fake.nested")
		if err == nil {
			t.Error("expected totally_fake to be dropped, but it persisted")
		}
	})

	t.Run("type mismatch on struct field fails", func(t *testing.T) {
		cfg, _ := config.Load()
		m, _ := configToMap(cfg)
		// Try to set server (a struct) to a string
		m["server"] = "bingo-bango"
		_, err := mapToConfig(m)
		if err == nil {
			t.Error("expected type mismatch to fail, but it succeeded")
		}
	})

	t.Run("llamacpp.options allows arbitrary keys", func(t *testing.T) {
		cfg, _ := config.Load()
		m, _ := configToMap(cfg)
		setValueByPath(m, "llamacpp.options.custom-option", "custom-value")
		newCfg, _ := mapToConfig(m)
		config.Save(newCfg)

		// Verify the custom option persisted (options is map[string]any)
		loaded, _ := config.Load()
		val, ok := loaded.LlamaCpp.GetOption("custom-option")
		if !ok {
			t.Error("expected custom-option to persist in llamacpp.options")
		}
		if val != "custom-value" {
			t.Errorf("expected 'custom-value', got %v", val)
		}
	})
}
