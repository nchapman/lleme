package hf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNewDownloaderWithProgress(t *testing.T) {
	client := &Client{}
	progressCalled := false

	downloader := NewDownloaderWithProgress(client, func(downloaded, total int64, speed float64, eta time.Duration) {
		progressCalled = true
	})

	if downloader == nil {
		t.Fatal("NewDownloaderWithProgress() returned nil")
	}

	if downloader.client != client {
		t.Error("downloader.client is not the provided client")
	}

	if downloader.progress == nil {
		t.Error("downloader.progress callback is nil")
	}

	downloader.progress(100, 1000, 1000000, 10*time.Second)

	if !progressCalled {
		t.Error("progress callback was not called")
	}
}

func TestCalculateProgress(t *testing.T) {
	now := time.Now()
	downloader := &Downloader{
		lastUpdate: now.Add(-2 * time.Second),
		lastBytes:  0,
	}

	progress := downloader.calculateProgress(2048, 4096)

	if progress.Downloaded != 2048 {
		t.Errorf("progress.Downloaded = %v, want 2048", progress.Downloaded)
	}

	if progress.Total != 4096 {
		t.Errorf("progress.Total = %v, want 4096", progress.Total)
	}

	if progress.Speed < 900 || progress.Speed > 1100 {
		t.Errorf("progress.Speed = %v, want approximately 1024", progress.Speed)
	}
}

func TestGetModelPath(t *testing.T) {
	user := "testuser"
	repo := "testrepo"

	path := GetModelPath(user, repo)

	if path == "" {
		t.Error("GetModelPath() returned empty string")
	}

	if filepath.Base(path) != repo {
		t.Errorf("GetModelPath() basename = %v, want %v", filepath.Base(path), repo)
	}
}

func TestGetModelFilePath(t *testing.T) {
	user := "testuser"
	repo := "testrepo"
	quant := "Q4_K_M"

	path := GetModelFilePath(user, repo, quant)

	if path == "" {
		t.Error("GetModelFilePath() returned empty string")
	}

	if !filepath.IsAbs(path) {
		t.Error("GetModelFilePath() should return absolute path")
	}

	if filepath.Base(path) != quant+".gguf" {
		t.Errorf("GetModelFilePath() filename = %v, want %v", filepath.Base(path), quant+".gguf")
	}
}

func TestCalculateSHA256WithProgress(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	testContent := "Hello, World!"

	if err := os.WriteFile(testFile, []byte(testContent), 0644); err != nil {
		t.Fatal(err)
	}

	hash, err := CalculateSHA256WithProgress(testFile, nil)
	if err != nil {
		t.Fatalf("CalculateSHA256WithProgress() error = %v", err)
	}

	expectedHash := "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
	if hash != expectedHash {
		t.Errorf("CalculateSHA256WithProgress() = %v, want %v", hash, expectedHash)
	}
}

func TestCleanupPartialFiles(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)

	os.Setenv("HOME", tmpDir)

	binDir := filepath.Join(tmpDir, ".lleme", "bin")
	modelsDir := filepath.Join(tmpDir, ".lleme", "models", "user", "repo")

	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatalf("Failed to create bin dir: %v", err)
	}
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}

	// Create some partial files
	partialFiles := []string{
		filepath.Join(binDir, "llama-cli.partial"),
		filepath.Join(binDir, "llama-server.partial"),
		filepath.Join(modelsDir, "model.gguf.partial"),
	}

	// Create some regular files that should NOT be deleted
	regularFiles := []string{
		filepath.Join(binDir, "llama-cli"),
		filepath.Join(modelsDir, "model.gguf"),
	}

	for _, f := range partialFiles {
		if err := os.WriteFile(f, []byte("partial"), 0644); err != nil {
			t.Fatalf("Failed to create partial file %s: %v", f, err)
		}
	}

	for _, f := range regularFiles {
		if err := os.WriteFile(f, []byte("complete"), 0644); err != nil {
			t.Fatalf("Failed to create regular file %s: %v", f, err)
		}
	}

	// Run cleanup
	count, err := CleanupPartialFiles()
	if err != nil {
		t.Fatalf("CleanupPartialFiles() error = %v", err)
	}
	if count != 3 {
		t.Errorf("CleanupPartialFiles() count = %d, want 3", count)
	}

	// Check partial files were deleted
	for _, f := range partialFiles {
		if _, err := os.Stat(f); !os.IsNotExist(err) {
			t.Errorf("Partial file %s should have been deleted", f)
		}
	}

	// Check regular files still exist
	for _, f := range regularFiles {
		if _, err := os.Stat(f); err != nil {
			t.Errorf("Regular file %s should still exist: %v", f, err)
		}
	}
}

func TestGetSplitModelDir(t *testing.T) {
	user := "testuser"
	repo := "testrepo"
	quant := "Q4_K_M"

	path := GetSplitModelDir(user, repo, quant)

	if path == "" {
		t.Error("GetSplitModelDir() returned empty string")
	}

	if !filepath.IsAbs(path) {
		t.Error("GetSplitModelDir() should return absolute path")
	}

	if filepath.Base(path) != quant {
		t.Errorf("GetSplitModelDir() basename = %v, want %v", filepath.Base(path), quant)
	}
}

func TestFindFirstSplitFile(t *testing.T) {
	tmpDir := t.TempDir()

	// Create split files
	splitFiles := []string{
		"model-00001-of-00003.gguf",
		"model-00002-of-00003.gguf",
		"model-00003-of-00003.gguf",
	}

	for _, f := range splitFiles {
		if err := os.WriteFile(filepath.Join(tmpDir, f), []byte("test"), 0644); err != nil {
			t.Fatalf("Failed to create file %s: %v", f, err)
		}
	}

	got := FindFirstSplitFile(tmpDir)
	want := filepath.Join(tmpDir, "model-00001-of-00003.gguf")

	if got != want {
		t.Errorf("FindFirstSplitFile() = %v, want %v", got, want)
	}
}

func TestFindFirstSplitFileNotFound(t *testing.T) {
	tmpDir := t.TempDir()

	// Create non-split file
	if err := os.WriteFile(filepath.Join(tmpDir, "model.gguf"), []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}

	got := FindFirstSplitFile(tmpDir)
	if got != "" {
		t.Errorf("FindFirstSplitFile() = %v, want empty string", got)
	}
}

func TestFindFirstSplitFileEmptyDir(t *testing.T) {
	tmpDir := t.TempDir()

	got := FindFirstSplitFile(tmpDir)
	if got != "" {
		t.Errorf("FindFirstSplitFile() = %v, want empty string", got)
	}
}

func TestBackendKindRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	user, repo, quant := "u", "r", "Q4_K_M"
	if err := os.MkdirAll(GetModelPath(user, repo), 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	// Unknown model → "gguf" as the legacy default.
	got, err := GetBackendKind(user, repo, quant)
	if err != nil {
		t.Fatalf("GetBackendKind: %v", err)
	}
	if got != BackendGGUF {
		t.Errorf("default kind = %q, want %q", got, BackendGGUF)
	}

	if err := SetBackendKind(user, repo, quant, BackendMLX); err != nil {
		t.Fatalf("SetBackendKind: %v", err)
	}
	got, err = GetBackendKind(user, repo, quant)
	if err != nil {
		t.Fatalf("GetBackendKind: %v", err)
	}
	if got != BackendMLX {
		t.Errorf("after set, kind = %q, want %q", got, BackendMLX)
	}

	// DownloadedAt auto-filled on first write.
	meta, err := LoadMetadata(user, repo)
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if meta.Quants[quant].DownloadedAt.IsZero() {
		t.Error("DownloadedAt was not set")
	}

	// Overwriting kind should keep DownloadedAt.
	first := meta.Quants[quant].DownloadedAt
	if err := SetBackendKind(user, repo, quant, BackendGGUF); err != nil {
		t.Fatalf("SetBackendKind: %v", err)
	}
	meta, _ = LoadMetadata(user, repo)
	if !meta.Quants[quant].DownloadedAt.Equal(first) {
		t.Error("DownloadedAt should not change on rewrite")
	}

	// Empty Backend in metadata still reports gguf.
	q := meta.Quants[quant]
	q.Backend = ""
	meta.Quants[quant] = q
	if err := SaveMetadata(user, repo, meta); err != nil {
		t.Fatalf("SaveMetadata: %v", err)
	}
	got, err = GetBackendKind(user, repo, quant)
	if err != nil {
		t.Fatalf("GetBackendKind: %v", err)
	}
	if got != BackendGGUF {
		t.Errorf("empty field kind = %q, want %q", got, BackendGGUF)
	}
}

func TestGetBackendKindRejectsUnknown(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	user, repo, quant := "u", "r", "q"
	if err := os.MkdirAll(GetModelPath(user, repo), 0755); err != nil {
		t.Fatal(err)
	}
	// Manually plant a corrupt / tampered metadata.yaml with an unknown
	// backend kind. Reading it must error rather than silently dispatch.
	yaml := "quants:\n  q:\n    backend: evil\n"
	if err := os.WriteFile(GetMetadataPath(user, repo), []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := GetBackendKind(user, repo, quant)
	if err == nil || !strings.Contains(err.Error(), "unrecognized backend kind") {
		t.Errorf("err = %v, want unrecognized backend kind", err)
	}
}

func TestFindModelFileSingleFile(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	user := "testuser"
	repo := "testrepo"
	quant := "Q4_K_M"

	// Create model directory and file
	modelDir := filepath.Join(tmpDir, ".lleme", "models", user, repo)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("Failed to create model dir: %v", err)
	}

	modelFile := filepath.Join(modelDir, quant+".gguf")
	if err := os.WriteFile(modelFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create model file: %v", err)
	}

	got := FindModelFile(user, repo, quant)
	if got != modelFile {
		t.Errorf("FindModelFile() = %v, want %v", got, modelFile)
	}
}

func TestFindModelFileSplitFiles(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	user := "testuser"
	repo := "testrepo"
	quant := "Q4_K_M"

	// Create split directory and files
	splitDir := filepath.Join(tmpDir, ".lleme", "models", user, repo, quant)
	if err := os.MkdirAll(splitDir, 0755); err != nil {
		t.Fatalf("Failed to create split dir: %v", err)
	}

	splitFiles := []string{
		"model-00001-of-00002.gguf",
		"model-00002-of-00002.gguf",
	}

	for _, f := range splitFiles {
		if err := os.WriteFile(filepath.Join(splitDir, f), []byte("test"), 0644); err != nil {
			t.Fatalf("Failed to create split file %s: %v", f, err)
		}
	}

	got := FindModelFile(user, repo, quant)
	want := filepath.Join(splitDir, "model-00001-of-00002.gguf")

	if got != want {
		t.Errorf("FindModelFile() = %v, want %v", got, want)
	}
}

func TestFindModelFileNotFound(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	// Create models directory but no model files
	modelsDir := filepath.Join(tmpDir, ".lleme", "models")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}

	got := FindModelFile("testuser", "testrepo", "Q4_K_M")
	if got != "" {
		t.Errorf("FindModelFile() = %v, want empty string", got)
	}
}

func TestCleanupPartialFilesEmptyDirs(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)

	os.Setenv("HOME", tmpDir)

	// Create the directories but don't put any partial files in them
	binDir := filepath.Join(tmpDir, ".lleme", "bin")
	modelsDir := filepath.Join(tmpDir, ".lleme", "models")

	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatalf("Failed to create bin dir: %v", err)
	}
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}

	count, err := CleanupPartialFiles()
	if err != nil {
		t.Errorf("CleanupPartialFiles() error = %v", err)
	}
	if count != 0 {
		t.Errorf("CleanupPartialFiles() count = %d, want 0", count)
	}
}

func TestIsMMProjFileName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"Q4_K_M-mmproj.gguf", true},
		{"UD-Q4_K_XL-mmproj.gguf", true},
		{"Q4_K_M-MMPROJ.gguf", true}, // case-insensitive
		{"Q4_K_M.gguf", false},
		{"UD-Q4_K_XL.gguf", false},
		{"mmproj-BF16.gguf", false}, // prefix form, not written by lleme
		{"mmproj.gguf", false},
		{"Q4_K_M-mmproj.bin", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsMMProjFileName(tt.name); got != tt.want {
				t.Errorf("IsMMProjFileName(%q) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
