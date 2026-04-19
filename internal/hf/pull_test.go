package hf

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGetManifestInfo(t *testing.T) {
	info := &ManifestInfo{
		GGUFSize:   4000000000,
		MMProjSize: 100000000,
		TotalSize:  4100000000,
		IsVision:   true,
	}

	if info.GGUFSize != 4000000000 {
		t.Errorf("GGUFSize = %d, want 4000000000", info.GGUFSize)
	}
	if info.MMProjSize != 100000000 {
		t.Errorf("MMProjSize = %d, want 100000000", info.MMProjSize)
	}
	if info.TotalSize != 4100000000 {
		t.Errorf("TotalSize = %d, want 4100000000", info.TotalSize)
	}
	if !info.IsVision {
		t.Error("IsVision should be true")
	}
}

func TestPullOptions(t *testing.T) {
	manifest := &Manifest{
		GGUFFile: &ManifestFile{
			RFilename: "model.gguf",
			Size:      1000,
		},
	}
	manifestJSON := []byte(`{"ggufFile":{"rfilename":"model.gguf","size":1000}}`)

	opts := &PullOptions{
		Manifest:     manifest,
		ManifestJSON: manifestJSON,
	}

	if opts.Manifest != manifest {
		t.Error("Manifest not stored correctly")
	}
	if string(opts.ManifestJSON) != string(manifestJSON) {
		t.Error("ManifestJSON not stored correctly")
	}
}

func TestPullResult(t *testing.T) {
	result := &PullResult{
		ModelPath:  "/path/to/model.gguf",
		MMProjPath: "/path/to/mmproj.gguf",
		IsVision:   true,
		TotalSize:  5000000000,
		GGUFSize:   4500000000,
		MMProjSize: 500000000,
	}

	if result.ModelPath != "/path/to/model.gguf" {
		t.Errorf("ModelPath = %s, want /path/to/model.gguf", result.ModelPath)
	}
	if result.MMProjPath != "/path/to/mmproj.gguf" {
		t.Errorf("MMProjPath = %s, want /path/to/mmproj.gguf", result.MMProjPath)
	}
	if !result.IsVision {
		t.Error("IsVision should be true")
	}
}

func TestPullProgress(t *testing.T) {
	progress := PullProgress{
		Phase:   "download",
		Current: 500,
		Total:   1000,
	}

	if progress.Phase != "download" {
		t.Errorf("Phase = %s, want download", progress.Phase)
	}
	if progress.Current != 500 {
		t.Errorf("Current = %d, want 500", progress.Current)
	}
	if progress.Total != 1000 {
		t.Errorf("Total = %d, want 1000", progress.Total)
	}
}

func TestHashesMatch(t *testing.T) {
	tests := []struct {
		name   string
		local  *ManifestFile
		remote *ManifestFile
		want   bool
	}{
		{
			name:   "both nil",
			local:  nil,
			remote: nil,
			want:   true,
		},
		{
			name:   "local nil",
			local:  nil,
			remote: &ManifestFile{Size: 100},
			want:   false,
		},
		{
			name:   "remote nil",
			local:  &ManifestFile{Size: 100},
			remote: nil,
			want:   false,
		},
		{
			name:   "matching hashes",
			local:  &ManifestFile{LFS: &ManifestLFS{SHA256: "abc123"}},
			remote: &ManifestFile{LFS: &ManifestLFS{SHA256: "abc123"}},
			want:   true,
		},
		{
			name:   "different hashes",
			local:  &ManifestFile{LFS: &ManifestLFS{SHA256: "abc123"}},
			remote: &ManifestFile{LFS: &ManifestLFS{SHA256: "def456"}},
			want:   false,
		},
		{
			name:   "no LFS - matching sizes",
			local:  &ManifestFile{Size: 1000},
			remote: &ManifestFile{Size: 1000},
			want:   true,
		},
		{
			name:   "no LFS - different sizes",
			local:  &ManifestFile{Size: 1000},
			remote: &ManifestFile{Size: 2000},
			want:   false,
		},
		{
			name:   "local no LFS, remote has LFS - match by size",
			local:  &ManifestFile{Size: 1000},
			remote: &ManifestFile{Size: 1000, LFS: &ManifestLFS{SHA256: "abc123"}},
			want:   true,
		},
		{
			name:   "remote no LFS, local has LFS - match by size",
			local:  &ManifestFile{Size: 1000, LFS: &ManifestLFS{SHA256: "abc123"}},
			remote: &ManifestFile{Size: 1000},
			want:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hashesMatch(tt.local, tt.remote)
			if got != tt.want {
				t.Errorf("hashesMatch() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseManifest(t *testing.T) {
	jsonData := []byte(`{
		"ggufFile": {
			"rfilename": "model-Q4_K_M.gguf",
			"size": 4000000000,
			"lfs": {
				"sha256": "abc123",
				"size": 4000000000
			}
		},
		"mmprojFile": {
			"rfilename": "mmproj.gguf",
			"size": 100000000
		}
	}`)

	var manifest Manifest
	err := json.Unmarshal(jsonData, &manifest)
	if err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if manifest.GGUFFile == nil {
		t.Fatal("GGUFFile should not be nil")
	}
	if manifest.GGUFFile.RFilename != "model-Q4_K_M.gguf" {
		t.Errorf("GGUFFile.RFilename = %s, want model-Q4_K_M.gguf", manifest.GGUFFile.RFilename)
	}
	if manifest.GGUFFile.Size != 4000000000 {
		t.Errorf("GGUFFile.Size = %d, want 4000000000", manifest.GGUFFile.Size)
	}
	if manifest.GGUFFile.LFS == nil {
		t.Fatal("GGUFFile.LFS should not be nil")
	}
	if manifest.GGUFFile.LFS.SHA256 != "abc123" {
		t.Errorf("GGUFFile.LFS.SHA256 = %s, want abc123", manifest.GGUFFile.LFS.SHA256)
	}
	if manifest.MMProjFile == nil {
		t.Fatal("MMProjFile should not be nil")
	}
	if manifest.MMProjFile.RFilename != "mmproj.gguf" {
		t.Errorf("MMProjFile.RFilename = %s, want mmproj.gguf", manifest.MMProjFile.RFilename)
	}
}

func TestParseManifestInvalid(t *testing.T) {
	jsonData := []byte(`invalid json`)

	var manifest Manifest
	err := json.Unmarshal(jsonData, &manifest)
	if err == nil {
		t.Error("json.Unmarshal() should return error for invalid JSON")
	}
}

func TestCalculateResultSizes(t *testing.T) {
	tests := []struct {
		name       string
		manifest   *Manifest
		splitInfo  *SplitInfo
		wantGGUF   int64
		wantTotal  int64
		wantVision bool
	}{
		{
			name: "single file",
			manifest: &Manifest{
				GGUFFile: &ManifestFile{Size: 4000000000},
			},
			splitInfo:  nil,
			wantGGUF:   4000000000,
			wantTotal:  4000000000,
			wantVision: false,
		},
		{
			name: "single file with mmproj",
			manifest: &Manifest{
				GGUFFile:   &ManifestFile{Size: 4000000000},
				MMProjFile: &ManifestFile{Size: 500000000},
			},
			splitInfo:  nil,
			wantGGUF:   4000000000,
			wantTotal:  4500000000,
			wantVision: true,
		},
		{
			name: "split files",
			manifest: &Manifest{
				GGUFFile: &ManifestFile{Size: 2000000000},
				SplitFiles: []*ManifestFile{
					{Size: 2000000000},
					{Size: 2000000000},
				},
			},
			splitInfo:  &SplitInfo{SplitCount: 3},
			wantGGUF:   6000000000,
			wantTotal:  6000000000,
			wantVision: false,
		},
		{
			name: "split files with mmproj",
			manifest: &Manifest{
				GGUFFile: &ManifestFile{Size: 2000000000},
				SplitFiles: []*ManifestFile{
					{Size: 2000000000},
				},
				MMProjFile: &ManifestFile{Size: 500000000},
			},
			splitInfo:  &SplitInfo{SplitCount: 2},
			wantGGUF:   4000000000,
			wantTotal:  4500000000,
			wantVision: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateResultSizes(tt.manifest, tt.splitInfo)
			if result.GGUFSize != tt.wantGGUF {
				t.Errorf("GGUFSize = %d, want %d", result.GGUFSize, tt.wantGGUF)
			}
			if result.TotalSize != tt.wantTotal {
				t.Errorf("TotalSize = %d, want %d", result.TotalSize, tt.wantTotal)
			}
			if result.IsVision != tt.wantVision {
				t.Errorf("IsVision = %v, want %v", result.IsVision, tt.wantVision)
			}
		})
	}
}

func TestVerifyFile(t *testing.T) {
	// Create temp file with known content
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.bin")
	content := []byte("test content for hashing")
	if err := os.WriteFile(testFile, content, 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// Calculate expected hash
	h := sha256.Sum256(content)
	expectedHash := hex.EncodeToString(h[:])

	tests := []struct {
		name    string
		hash    string
		wantErr bool
	}{
		{
			name:    "matching hash",
			hash:    expectedHash,
			wantErr: false,
		},
		{
			name:    "wrong hash",
			hash:    "0000000000000000000000000000000000000000000000000000000000000000",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var progressCalled bool
			err := verifyFile(testFile, tt.hash, func(current, total int64) {
				progressCalled = true
			})
			if (err != nil) != tt.wantErr {
				t.Errorf("verifyFile() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && !progressCalled {
				t.Error("progress callback should be called on success")
			}
		})
	}
}

func TestVerifyFileCaseInsensitive(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.bin")
	content := []byte("test content")
	os.WriteFile(testFile, content, 0644)

	h := sha256.Sum256(content)
	lowerHash := hex.EncodeToString(h[:])
	upperHash := hex.EncodeToString(h[:])
	for i := range upperHash {
		if upperHash[i] >= 'a' && upperHash[i] <= 'f' {
			upperHash = upperHash[:i] + string(upperHash[i]-32) + upperHash[i+1:]
		}
	}

	// Both should work
	if err := verifyFile(testFile, lowerHash, nil); err != nil {
		t.Errorf("lowercase hash should match: %v", err)
	}
	if err := verifyFile(testFile, upperHash, nil); err != nil {
		t.Errorf("uppercase hash should match: %v", err)
	}
}

func TestVerifyFileNotFound(t *testing.T) {
	err := verifyFile("/nonexistent/file.bin", "abc123", nil)
	if err == nil {
		t.Error("verifyFile() should return error for nonexistent file")
	}
}

func TestCleanupFilesSingleFiles(t *testing.T) {
	tmpDir := t.TempDir()
	file1 := filepath.Join(tmpDir, "file1.gguf")
	file2 := filepath.Join(tmpDir, "file2.gguf")

	os.WriteFile(file1, []byte("test"), 0644)
	os.WriteFile(file2, []byte("test"), 0644)

	files := []fileDownload{
		{destPath: file1},
		{destPath: file2},
	}

	cleanupFiles(files, nil, "user", "repo", Quantization{Name: "Q4_K_M"})

	if _, err := os.Stat(file1); !os.IsNotExist(err) {
		t.Error("file1 should be deleted")
	}
	if _, err := os.Stat(file2); !os.IsNotExist(err) {
		t.Error("file2 should be deleted")
	}
}

func TestVerifyAllFilesSuccess(t *testing.T) {
	tmpDir := t.TempDir()

	content := []byte("test content")
	h := sha256.Sum256(content)
	hash := hex.EncodeToString(h[:])

	testFile := filepath.Join(tmpDir, "model.gguf")
	os.WriteFile(testFile, content, 0644)

	files := []fileDownload{
		{
			file:     &ManifestFile{RFilename: "model.gguf", Size: int64(len(content)), LFS: &ManifestLFS{SHA256: hash}},
			destPath: testFile,
		},
	}

	err := verifyAllFiles(files, int64(len(content)), nil)
	if err != nil {
		t.Fatalf("verifyAllFiles() error = %v", err)
	}
}

func TestVerifyAllFilesHashMismatch(t *testing.T) {
	tmpDir := t.TempDir()

	badFile := filepath.Join(tmpDir, "bad.gguf")
	os.WriteFile(badFile, []byte("bad content"), 0644)

	files := []fileDownload{
		{
			file:     &ManifestFile{RFilename: "bad.gguf", Size: 11, LFS: &ManifestLFS{SHA256: "wrong_hash"}},
			destPath: badFile,
		},
	}

	err := verifyAllFiles(files, 11, nil)
	if err == nil {
		t.Error("verifyAllFiles() should fail for wrong hash")
	}

	// File should be deleted
	if _, err := os.Stat(badFile); !os.IsNotExist(err) {
		t.Error("bad file should be deleted after verification failure")
	}
}

func TestVerifyAllFilesSkipsWithoutHash(t *testing.T) {
	tmpDir := t.TempDir()

	noHashFile := filepath.Join(tmpDir, "nohash.gguf")
	os.WriteFile(noHashFile, []byte("content"), 0644)

	files := []fileDownload{
		{
			file:     &ManifestFile{RFilename: "nohash.gguf", Size: 7},
			destPath: noHashFile,
		},
	}

	err := verifyAllFiles(files, 7, nil)
	if err != nil {
		t.Fatalf("verifyAllFiles() should not fail for files without hash: %v", err)
	}
}

func TestGetOrFetchManifestUsesProvided(t *testing.T) {
	manifest := &Manifest{
		GGUFFile: &ManifestFile{RFilename: "model.gguf"},
	}
	manifestJSON := []byte(`{"test": true}`)

	opts := &PullOptions{
		Manifest:     manifest,
		ManifestJSON: manifestJSON,
	}

	got, gotJSON, err := getOrFetchManifest(nil, "user", "repo", Quantization{}, opts)
	if err != nil {
		t.Fatalf("getOrFetchManifest() error = %v", err)
	}
	if got != manifest {
		t.Error("should return provided manifest")
	}
	if string(gotJSON) != string(manifestJSON) {
		t.Error("should return provided manifestJSON")
	}
}

func TestGetOrFetchManifestRequiresClient(t *testing.T) {
	// Without opts.Manifest, function needs a valid client
	_, _, err := getOrFetchManifest(nil, "user", "repo", Quantization{}, nil)
	if err == nil {
		t.Error("should error without client when no opts.Manifest provided")
	}
}

func TestBuildFileListSingleFile(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	manifest := &Manifest{
		GGUFFile: &ManifestFile{RFilename: "model-Q4_K_M.gguf", Size: 1000},
	}
	result := &PullResult{}
	quant := Quantization{Name: "Q4_K_M"}

	files, err := buildFileList("user", "repo", quant, manifest, nil, result)
	if err != nil {
		t.Fatalf("buildFileList() error = %v", err)
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file, got %d", len(files))
	}
	if result.ModelPath == "" {
		t.Error("ModelPath should be set")
	}
}

func TestBuildFileListSplitFiles(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	manifest := &Manifest{
		GGUFFile: &ManifestFile{RFilename: "model-00001-of-00002.gguf", Size: 500},
		SplitFiles: []*ManifestFile{
			{RFilename: "model-00002-of-00002.gguf", Size: 500},
		},
	}
	result := &PullResult{}
	quant := Quantization{Name: "Q4_K_M"}
	splitInfo := &SplitInfo{SplitCount: 2, Prefix: "model"}

	files, err := buildFileList("user", "repo", quant, manifest, splitInfo, result)
	if err != nil {
		t.Fatalf("buildFileList() error = %v", err)
	}
	if len(files) != 2 {
		t.Errorf("expected 2 files, got %d", len(files))
	}
	if result.ModelPath == "" {
		t.Error("ModelPath should be set")
	}
}

func TestBuildFileListVision(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	manifest := &Manifest{
		GGUFFile:   &ManifestFile{RFilename: "model.gguf", Size: 1000},
		MMProjFile: &ManifestFile{RFilename: "mmproj.gguf", Size: 100},
	}
	result := &PullResult{IsVision: true}
	quant := Quantization{Name: "Q4_K_M"}

	files, err := buildFileList("user", "repo", quant, manifest, nil, result)
	if err != nil {
		t.Fatalf("buildFileList() error = %v", err)
	}
	if len(files) != 2 {
		t.Errorf("expected 2 files, got %d", len(files))
	}
	if result.MMProjPath == "" {
		t.Error("MMProjPath should be set for vision model")
	}
}

func TestSaveManifestSingleFile(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	manifest := &Manifest{
		GGUFFile: &ManifestFile{RFilename: "model.gguf", Size: 1000},
	}
	manifestJSON := []byte(`{"gguf_file":{"rfilename":"model.gguf","size":1000}}`)

	// Create the model directory
	modelDir := GetModelPath("user", "repo")
	os.MkdirAll(modelDir, 0755)

	err := saveManifest("user", "repo", "Q4_K_M", manifest, manifestJSON)
	if err != nil {
		t.Fatalf("saveManifest() error = %v", err)
	}

	// Verify file was written
	manifestPath := GetManifestFilePath("user", "repo", "Q4_K_M")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("failed to read manifest: %v", err)
	}
	if string(data) != string(manifestJSON) {
		t.Error("manifest content should match for single file")
	}
}

func TestSaveManifestSplitFiles(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	manifest := &Manifest{
		GGUFFile: &ManifestFile{RFilename: "model-00001-of-00002.gguf", Size: 500},
		SplitFiles: []*ManifestFile{
			{RFilename: "model-00002-of-00002.gguf", Size: 500},
		},
	}
	manifestJSON := []byte(`original`)

	// Create the model directory
	modelDir := GetModelPath("user", "repo")
	os.MkdirAll(modelDir, 0755)

	err := saveManifest("user", "repo", "Q4_K_M", manifest, manifestJSON)
	if err != nil {
		t.Fatalf("saveManifest() error = %v", err)
	}

	// Verify file was written with marshaled manifest (not original JSON)
	manifestPath := GetManifestFilePath("user", "repo", "Q4_K_M")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("failed to read manifest: %v", err)
	}
	// Split files should re-marshal, not use original JSON
	if string(data) == "original" {
		t.Error("split manifest should be re-marshaled, not use original JSON")
	}
}

func TestCleanupFilesSplit(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	// Create split directory with files
	splitDir := GetSplitModelDir("user", "repo", "Q4_K_M")
	os.MkdirAll(splitDir, 0755)
	os.WriteFile(filepath.Join(splitDir, "model-00001-of-00002.gguf"), []byte("test"), 0644)
	os.WriteFile(filepath.Join(splitDir, "model-00002-of-00002.gguf"), []byte("test"), 0644)

	splitInfo := &SplitInfo{SplitCount: 2}
	cleanupFiles(nil, splitInfo, "user", "repo", Quantization{Name: "Q4_K_M"})

	// Verify directory was removed
	if _, err := os.Stat(splitDir); !os.IsNotExist(err) {
		t.Error("split directory should be deleted")
	}
}
