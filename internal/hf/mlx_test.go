package hf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectFormat(t *testing.T) {
	tests := []struct {
		name  string
		files []FileTree
		want  string
	}{
		{
			name: "gguf preferred over mlx when both present",
			files: []FileTree{
				{Path: "config.json", Type: "file"},
				{Path: "model.safetensors", Type: "file"},
				{Path: "model.Q4_K_M.gguf", Type: "file"},
			},
			want: BackendGGUF,
		},
		{
			name: "mlx detected from config and safetensors",
			files: []FileTree{
				{Path: "config.json", Type: "file"},
				{Path: "tokenizer.json", Type: "file"},
				{Path: "model.safetensors", Type: "file"},
			},
			want: BackendMLX,
		},
		{
			name: "gguf only",
			files: []FileTree{
				{Path: "llama.Q8_0.gguf", Type: "file"},
			},
			want: BackendGGUF,
		},
		{
			name: "config without safetensors is unknown",
			files: []FileTree{
				{Path: "config.json", Type: "file"},
				{Path: "pytorch_model.bin", Type: "file"},
			},
			want: "",
		},
		{
			name:  "empty repo is unknown",
			files: nil,
			want:  "",
		},
		{
			name: "config.json in subdir still counts",
			files: []FileTree{
				{Path: "snapshots/abc/config.json", Type: "file"},
				{Path: "snapshots/abc/model.safetensors", Type: "file"},
			},
			want: BackendMLX,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectFormat(tt.files); got != tt.want {
				t.Errorf("DetectFormat() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMLXQuantFromRepo(t *testing.T) {
	tests := []struct {
		repo string
		want string
	}{
		{"Qwen3-4B-Instruct-4bit", "4bit"},
		{"Llama-3-8B-8bit", "8bit"},
		{"Mistral-7B-bf16", "bf16"},
		{"Phi-3-mini-fp16", "fp16"},
		{"Gemma-2-9B-it", "default"},
		{"Some-Model-4bit-eval", "default"}, // not anchored at end
		{"Qwen3-4B-Instruct-4BIT", "4bit"},  // case-insensitive
	}

	for _, tt := range tests {
		t.Run(tt.repo, func(t *testing.T) {
			if got := MLXQuantFromRepo(tt.repo); got != tt.want {
				t.Errorf("MLXQuantFromRepo(%q) = %q, want %q", tt.repo, got, tt.want)
			}
		})
	}
}

func TestListMLXFiles(t *testing.T) {
	files := []FileTree{
		{Path: "README.md", Type: "file"},
		{Path: "config.json", Type: "file"},
		{Path: "tokenizer.json", Type: "file"},
		{Path: "model-00001-of-00002.safetensors", Type: "file"},
		{Path: "model-00002-of-00002.safetensors", Type: "file"},
		{Path: "model.safetensors.index.json", Type: "file"},
		{Path: "pytorch_model.bin", Type: "file"},
		{Path: "assets", Type: "directory"},
		{Path: "special_tokens_map.json", Type: "file"},
	}

	got := ListMLXFiles(files)
	wantPaths := map[string]bool{
		"config.json":                      true,
		"tokenizer.json":                   true,
		"model-00001-of-00002.safetensors": true,
		"model-00002-of-00002.safetensors": true,
		"model.safetensors.index.json":     true,
		"special_tokens_map.json":          true,
	}
	if len(got) != len(wantPaths) {
		t.Fatalf("got %d files, want %d: %+v", len(got), len(wantPaths), got)
	}
	for _, f := range got {
		if !wantPaths[f.Path] {
			t.Errorf("unexpected file in output: %s", f.Path)
		}
	}
}

// Path-traversal guard is enforced in PullMLXModel via binaryrelease.SafeJoin.
// The behavior is covered by binaryrelease's own TestExtractTarGzRejects*
// suite; here we just pin that the guard exists at the MLX call site so a
// refactor can't accidentally regress it to a bare filepath.Join.
func TestPullMLXModelUsesSafeJoin(t *testing.T) {
	data, err := os.ReadFile("mlx.go")
	if err != nil {
		t.Fatalf("read mlx.go: %v", err)
	}
	if !strings.Contains(string(data), "binaryrelease.SafeJoin") {
		t.Error("PullMLXModel must route HF-sourced paths through binaryrelease.SafeJoin")
	}
}

// MLX manifest round-trip: write then read back matches the input.
func TestMLXManifestRoundTrip(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	remote := []FileTree{
		{Path: "config.json", Size: 100},
		{Path: "model.safetensors", Size: 9001},
	}
	if err := SaveMLXManifest("u", "r", "4bit", remote); err != nil {
		t.Fatalf("SaveMLXManifest: %v", err)
	}
	got, err := LoadMLXManifest("u", "r", "4bit")
	if err != nil {
		t.Fatalf("LoadMLXManifest: %v", err)
	}
	if got == nil || got.Version != mlxManifestVersion || len(got.Files) != 2 {
		t.Fatalf("unexpected manifest: %+v", got)
	}
	if got.Files[1].Size != 9001 {
		t.Errorf("size not round-tripped: %+v", got.Files[1])
	}
}

// manifestMatches hits when sizes match, misses on drift, and misses when a
// local file was deleted between pulls.
func TestMLXManifestMatches(t *testing.T) {
	dir := t.TempDir()
	writeFile := func(rel string, size int64) {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, size), 0644); err != nil {
			t.Fatal(err)
		}
	}
	writeFile("config.json", 100)
	writeFile("model.safetensors", 9001)

	remote := []FileTree{
		{Path: "config.json", Size: 100},
		{Path: "model.safetensors", Size: 9001},
	}
	manifest := &MLXManifest{
		Version: mlxManifestVersion,
		Files: []MLXManifestFile{
			{Path: "config.json", Size: 100},
			{Path: "model.safetensors", Size: 9001},
		},
	}

	if !mlxManifestMatches(dir, remote, manifest) {
		t.Error("matching remote + local + manifest should be up-to-date")
	}

	// Remote drift: one file resized upstream.
	driftedRemote := []FileTree{
		{Path: "config.json", Size: 100},
		{Path: "model.safetensors", Size: 9999},
	}
	if mlxManifestMatches(dir, driftedRemote, manifest) {
		t.Error("size drift in remote should not be up-to-date")
	}

	// Local drift: user deleted a file.
	if err := os.Remove(filepath.Join(dir, "config.json")); err != nil {
		t.Fatal(err)
	}
	if mlxManifestMatches(dir, remote, manifest) {
		t.Error("missing local file should not be up-to-date")
	}
}

// The fallback path (no manifest) matches when every remote file is present
// locally with the expected size, and misses otherwise.
func TestMLXFilesMatchDiskFallback(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), make([]byte, 50), 0644); err != nil {
		t.Fatal(err)
	}
	remote := []FileTree{{Path: "config.json", Size: 50}}
	if !mlxFilesMatchDisk(dir, remote) {
		t.Error("exact-size match should return true")
	}
	remote[0].Size = 60
	if mlxFilesMatchDisk(dir, remote) {
		t.Error("size mismatch should return false")
	}
}

func TestListMLXFilesPreservesNestedPaths(t *testing.T) {
	files := []FileTree{
		{Path: "snapshots/abc/config.json", Type: "file"},
		{Path: "snapshots/abc/model.safetensors", Type: "file"},
		{Path: "snapshots/abc/README.md", Type: "file"},
	}
	got := ListMLXFiles(files)
	if len(got) != 2 {
		t.Fatalf("got %d files, want 2: %+v", len(got), got)
	}
	for _, f := range got {
		if !strings.HasPrefix(f.Path, "snapshots/abc/") {
			t.Errorf("path not preserved: %s", f.Path)
		}
	}
}
