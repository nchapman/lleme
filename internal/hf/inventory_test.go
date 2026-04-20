package hf

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	data := make([]byte, size)
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func TestListLocalModelsMixedBackends(t *testing.T) {
	modelsDir := t.TempDir()

	// GGUF single file with metadata.
	gguf := filepath.Join(modelsDir, "bartowski", "gguf-model")
	writeFile(t, filepath.Join(gguf, "Q4_K_M.gguf"), 100)
	writeFile(t, filepath.Join(gguf, "metadata.yaml"),
		0)
	_ = os.WriteFile(filepath.Join(gguf, "metadata.yaml"),
		[]byte("quants:\n  Q4_K_M:\n    backend: gguf\n"), 0644)

	// MLX directory with metadata.
	mlx := filepath.Join(modelsDir, "mlx-community", "qwen-4bit")
	writeFile(t, filepath.Join(mlx, "4bit", "model.safetensors"), 200)
	writeFile(t, filepath.Join(mlx, "4bit", "config.json"), 50)
	_ = os.WriteFile(filepath.Join(mlx, "metadata.yaml"),
		[]byte("quants:\n  4bit:\n    backend: mlx\n"), 0644)

	// Legacy GGUF with no metadata at all — must still be listed.
	legacy := filepath.Join(modelsDir, "legacy-user", "legacy-repo")
	writeFile(t, filepath.Join(legacy, "Q8_0.gguf"), 75)

	got, err := ListLocalModelsInDir(modelsDir)
	if err != nil {
		t.Fatalf("ListLocalModelsInDir: %v", err)
	}

	if len(got) != 3 {
		t.Fatalf("got %d models, want 3: %+v", len(got), got)
	}

	byKey := func(m LocalModel) string { return m.User + "/" + m.Repo + ":" + m.Quant }
	seen := map[string]LocalModel{}
	for _, m := range got {
		seen[byKey(m)] = m
	}

	if m, ok := seen["bartowski/gguf-model:Q4_K_M"]; !ok {
		t.Error("missing GGUF entry")
	} else if m.Backend != BackendGGUF {
		t.Errorf("gguf backend = %q", m.Backend)
	} else if m.Size != 100 {
		t.Errorf("gguf size = %d, want 100", m.Size)
	}

	if m, ok := seen["mlx-community/qwen-4bit:4bit"]; !ok {
		t.Error("missing MLX entry")
	} else if m.Backend != BackendMLX {
		t.Errorf("mlx backend = %q", m.Backend)
	} else if m.Size != 250 {
		t.Errorf("mlx size = %d, want 250 (sum of dir contents)", m.Size)
	}

	if m, ok := seen["legacy-user/legacy-repo:Q8_0"]; !ok {
		t.Error("missing legacy entry (no metadata.yaml fallback broken)")
	} else if m.Backend != BackendGGUF {
		t.Errorf("legacy backend = %q, want gguf fallback", m.Backend)
	}
}

// Regression: MLX repos preserve the HuggingFace tree layout, so files
// can sit under nested paths (e.g. snapshots/<rev>/model.safetensors).
// dirSize must walk recursively or the reported model size undercounts.
func TestListLocalModelsMLXNestedSizeRecursive(t *testing.T) {
	modelsDir := t.TempDir()

	repoDir := filepath.Join(modelsDir, "mlx-community", "nested-repo")
	quantDir := filepath.Join(repoDir, "4bit")
	// Flat files (tokenizer/config) at the quant root.
	writeFile(t, filepath.Join(quantDir, "config.json"), 100)
	writeFile(t, filepath.Join(quantDir, "tokenizer.json"), 200)
	// Nested weight shard under a snapshots/<rev>/ tree, like HF serves.
	writeFile(t, filepath.Join(quantDir, "snapshots", "abc123", "model.safetensors"), 5000)
	writeFile(t, filepath.Join(quantDir, "snapshots", "abc123", "model.safetensors.index.json"), 300)

	_ = os.WriteFile(filepath.Join(repoDir, "metadata.yaml"),
		[]byte("quants:\n  4bit:\n    backend: mlx\n"), 0644)

	got, err := ListLocalModelsInDir(modelsDir)
	if err != nil {
		t.Fatalf("ListLocalModelsInDir: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1 model, got %d: %+v", len(got), got)
	}
	const wantSize = int64(100 + 200 + 5000 + 300)
	if got[0].Size != wantSize {
		t.Errorf("MLX nested size = %d, want %d (flat+nested files)", got[0].Size, wantSize)
	}
}

func TestListLocalModelsEmpty(t *testing.T) {
	dir := t.TempDir()
	got, err := ListLocalModelsInDir(dir)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected empty slice, got %d", len(got))
	}

	// Missing dir is also fine (fresh install).
	got, err = ListLocalModelsInDir(filepath.Join(dir, "does-not-exist"))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected empty for missing dir, got %d", len(got))
	}
}
