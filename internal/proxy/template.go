package proxy

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
)

// TemplatePatch defines a single, focused fix for a chat template issue.
// Each patch should address exactly one problem and be as minimal as possible.
type TemplatePatch struct {
	// ID is a short identifier for the patch (e.g., "empty-tools-array")
	ID string

	// Description explains what problem this patch fixes
	Description string

	// Reference links to relevant issues or documentation (optional)
	Reference string

	// Apply transforms the template. Returns the modified template.
	// Should be idempotent - applying twice should have the same effect as once.
	Apply func(template string) string
}

// templatePatches is the registry of all patches to apply to chat templates.
// Patches are applied in order. Each patch should be:
//   - Focused: Fix exactly one problem
//   - Minimal: Change only what's necessary
//   - Idempotent: Safe to apply multiple times
//   - Documented: Clear description of what and why
var templatePatches = []TemplatePatch{
	patchEmptyToolsArray,
}

// patchEmptyToolsArray fixes llama-server passing tools=[] instead of tools=none.
//
// Problem:
//
//	llama-server passes an empty array (tools=[]) rather than none/null when
//	no tools are specified in the API request. Jinja templates that check
//	"if tools is not none" evaluate to true for an empty array, causing
//	tool-calling instructions to appear in the prompt even when tools aren't used.
//
// Fix:
//
//	Add a length check so the condition is false for empty arrays:
//	  "tools is not none" -> "(tools is not none and tools | length > 0)"
//
// Affected templates:
//
//	Most tool-capable model templates (Llama 3.x, Qwen, Mistral, etc.)
var patchEmptyToolsArray = TemplatePatch{
	ID:          "empty-tools-array",
	Description: "Fix llama-server passing tools=[] instead of tools=none",
	Reference:   "https://github.com/ggerganov/llama.cpp/issues/9909",
	Apply: func(template string) string {
		// Each replacement targets a specific Jinja syntax variant.
		// We only replace exact matches to avoid breaking other logic.
		replacements := []struct {
			pattern     string // Exact string to find
			replacement string // What to replace it with
		}{
			// "tools is not none" - most common form (Llama 3.x, Qwen)
			{
				pattern:     "tools is not none",
				replacement: "(tools is not none and tools | length > 0)",
			},
			// "tools != none" - alternative syntax
			{
				pattern:     "tools != none",
				replacement: "(tools != none and tools | length > 0)",
			},
			// "not tools is none" - inverted form
			{
				pattern:     "not tools is none",
				replacement: "(not tools is none and tools | length > 0)",
			},
		}

		result := template
		for _, r := range replacements {
			// Skip if already patched (idempotency check)
			if strings.Contains(result, r.replacement) {
				continue
			}
			result = strings.ReplaceAll(result, r.pattern, r.replacement)
		}
		return result
	},
}

// ExtractAndPatchTemplate extracts the chat template from a GGUF file and
// applies all registered patches. Returns the path to the patched template
// file, or empty string if no patches were needed.
func ExtractAndPatchTemplate(modelPath string) (string, error) {
	template, err := extractChatTemplate(modelPath)
	if err != nil {
		return "", err
	}

	if template == "" {
		return "", nil // No template in model, let llama-server use defaults
	}

	// Apply all patches
	patched := applyPatches(template)

	// If no changes were made, no need for a custom template file
	if patched == template {
		return "", nil
	}

	// Write to cache
	return writeTemplateCache(modelPath, patched)
}

// applyPatches applies all registered patches to a template.
func applyPatches(template string) string {
	result := template
	for _, patch := range templatePatches {
		result = patch.Apply(result)
	}
	return result
}

// extractChatTemplate reads the chat_template from a GGUF file's metadata.
func extractChatTemplate(modelPath string) (string, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return "", fmt.Errorf("failed to open model: %w", err)
	}
	defer f.Close()

	kvCount, err := readGGUFTemplateHeader(f)
	if err != nil {
		return "", err
	}

	for i := uint64(0); i < kvCount; i++ {
		key, vtype, err := readGGUFKVKey(f)
		if err != nil {
			return "", err
		}
		if key == "tokenizer.chat_template" && vtype == 8 { // 8 = string
			return readGGUFTemplateString(f)
		}
		if err := hf.SkipGGUFValue(f, vtype); err != nil {
			return "", fmt.Errorf("failed to skip value: %w", err)
		}
	}

	return "", nil
}

// readGGUFTemplateHeader reads and validates the GGUF preamble (magic, version,
// tensor count, KV count) for supported v2/v3 files. Returns the KV count.
func readGGUFTemplateHeader(f *os.File) (kvCount uint64, err error) {
	magic := make([]byte, 4)
	if _, err = f.Read(magic); err != nil {
		return 0, fmt.Errorf("failed to read magic: %w", err)
	}
	if string(magic) != "GGUF" {
		return 0, fmt.Errorf("not a GGUF file")
	}
	var version uint32
	if err = binary.Read(f, binary.LittleEndian, &version); err != nil {
		return 0, fmt.Errorf("failed to read version: %w", err)
	}
	if version < 2 || version > 3 {
		return 0, fmt.Errorf("unsupported GGUF version %d (expected 2 or 3)", version)
	}
	var tensorCount uint64
	if err = binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return 0, fmt.Errorf("failed to read tensor count: %w", err)
	}
	if err = binary.Read(f, binary.LittleEndian, &kvCount); err != nil {
		return 0, fmt.Errorf("failed to read kv count: %w", err)
	}
	return kvCount, nil
}

// readGGUFKVKey reads the key string and value type of a GGUF KV pair.
func readGGUFKVKey(f *os.File) (key string, vtype uint32, err error) {
	key, err = readGGUFTemplateString(f)
	if err != nil {
		return "", 0, fmt.Errorf("failed to read key: %w", err)
	}
	if err = binary.Read(f, binary.LittleEndian, &vtype); err != nil {
		return "", 0, fmt.Errorf("failed to read value type: %w", err)
	}
	return key, vtype, nil
}

// readGGUFTemplateString reads a length-prefixed string from a GGUF file.
func readGGUFTemplateString(f *os.File) (string, error) {
	var length uint64
	if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1024*1024 {
		return "", fmt.Errorf("string too long: %d", length)
	}
	data := make([]byte, length)
	if _, err := f.Read(data); err != nil {
		return "", err
	}
	return string(data), nil
}

// writeTemplateCache writes a patched template to a cache file and returns its path.
func writeTemplateCache(modelPath, template string) (string, error) {
	cacheDir := filepath.Join(config.CachePath(), "templates")
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create template cache dir: %w", err)
	}

	// Hash model path + mtime so cache invalidates when model is updated
	info, err := os.Stat(modelPath)
	if err != nil {
		return "", fmt.Errorf("failed to stat model: %w", err)
	}
	hashInput := fmt.Sprintf("%s:%d", modelPath, info.ModTime().UnixNano())
	hash := sha256.Sum256([]byte(hashInput))
	filename := fmt.Sprintf("%x.jinja", hash[:8])
	cachePath := filepath.Join(cacheDir, filename)

	if err := os.WriteFile(cachePath, []byte(template), 0644); err != nil {
		return "", fmt.Errorf("failed to write template cache: %w", err)
	}

	return cachePath, nil
}
