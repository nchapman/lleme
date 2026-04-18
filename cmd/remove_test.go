package cmd

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"
)

func TestParseDuration(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    time.Duration
		wantErr bool
	}{
		// Valid durations
		{"1 hour", "1h", 1 * time.Hour, false},
		{"24 hours", "24h", 24 * time.Hour, false},
		{"1 day", "1d", 24 * time.Hour, false},
		{"7 days", "7d", 7 * 24 * time.Hour, false},
		{"30 days", "30d", 30 * 24 * time.Hour, false},
		{"1 week", "1w", 7 * 24 * time.Hour, false},
		{"4 weeks", "4w", 28 * 24 * time.Hour, false},
		{"uppercase", "7D", 7 * 24 * time.Hour, false},
		{"with spaces", " 7d ", 7 * 24 * time.Hour, false},

		// Invalid durations
		{"empty", "", 0, true},
		{"just number", "7", 0, true},
		{"just unit", "d", 0, true},
		{"invalid unit", "7m", 0, true},
		{"invalid unit 2", "7x", 0, true},
		{"negative", "-7d", 0, true},
		{"float", "7.5d", 0, true},
		{"no number", "days", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseDuration(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseDuration(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("parseDuration(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseSize(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    int64
		wantErr bool
	}{
		// Valid sizes
		{"bytes", "100B", 100, false},
		{"kilobytes", "1KB", 1024, false},
		{"kilobytes short", "1K", 1024, false},
		{"megabytes", "1MB", 1024 * 1024, false},
		{"megabytes short", "1M", 1024 * 1024, false},
		{"gigabytes", "1GB", 1024 * 1024 * 1024, false},
		{"gigabytes short", "1G", 1024 * 1024 * 1024, false},
		{"terabytes", "1TB", 1024 * 1024 * 1024 * 1024, false},
		{"terabytes short", "1T", 1024 * 1024 * 1024 * 1024, false},
		{"10 gigabytes", "10GB", 10 * 1024 * 1024 * 1024, false},
		{"500 megabytes", "500MB", 500 * 1024 * 1024, false},
		{"lowercase", "10gb", 10 * 1024 * 1024 * 1024, false},
		{"with spaces", " 10GB ", 10 * 1024 * 1024 * 1024, false},
		{"float", "1.5GB", int64(1.5 * 1024 * 1024 * 1024), false},
		{"float megabytes", "2.5MB", int64(2.5 * 1024 * 1024), false},

		// Invalid sizes
		{"empty", "", 0, true},
		{"just number", "100", 0, true},
		{"just unit", "GB", 0, true},
		{"invalid unit", "100XB", 0, true},
		{"negative", "-10GB", 0, true},
		{"no number", "gigabytes", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseSize(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseSize(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("parseSize(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestGlobToRegex(t *testing.T) {
	tests := []struct {
		name    string
		glob    string
		input   string
		matches bool
	}{
		// Wildcard patterns
		{"match all", "*", "anything", true},
		{"match all with slash", "*", "user/repo", true},
		{"prefix wildcard", "user/*", "user/repo", true},
		{"prefix wildcard no match", "user/*", "other/repo", false},
		{"suffix wildcard", "*/repo", "user/repo", true},
		{"suffix wildcard no match", "*/repo", "user/other", false},
		{"middle wildcard", "user/*/quant", "user/repo/quant", true},
		{"double wildcard", "*/*", "user/repo", true},

		// Question mark
		{"single char", "use?/repo", "user/repo", true},
		{"single char no match", "use?/repo", "users/repo", false},

		// Exact match
		{"exact match", "user/repo", "user/repo", true},
		{"exact no match", "user/repo", "user/other", false},

		// Special characters (regression test for dot escaping)
		{"dots in name", "user/GLM-4.7-Flash", "user/GLM-4.7-Flash", true},
		{"dots escaped properly", "user/GLM-4.7-Flash", "user/GLM-4x7-Flash", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			regex := globToRegex(tt.glob)
			matched, err := matchesPattern("^"+regex+"$", tt.input)
			if err != nil {
				t.Fatalf("regex compile error: %v", err)
			}
			if matched != tt.matches {
				t.Errorf("globToRegex(%q) matching %q = %v, want %v (regex: %s)", tt.glob, tt.input, matched, tt.matches, regex)
			}
		})
	}
}

func matchesPattern(pattern, input string) (bool, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return false, err
	}
	return re.MatchString(input), nil
}

func TestFindModels(t *testing.T) {
	// Create a temporary models directory
	tmpDir := t.TempDir()

	// Create test model structure
	testModels := []struct {
		user  string
		repo  string
		quant string
		size  int64
		age   time.Duration
	}{
		{"userA", "model1", "Q4_K_M", 1024 * 1024 * 100, 2 * 24 * time.Hour},        // 100MB, 2 days old
		{"userA", "model1", "Q8_0", 1024 * 1024 * 200, 5 * 24 * time.Hour},          // 200MB, 5 days old
		{"userA", "model2", "Q4_K_M", 1024 * 1024 * 1024 * 5, 1 * 24 * time.Hour},   // 5GB, 1 day old
		{"userB", "model1", "Q4_K_M", 1024 * 1024 * 1024 * 10, 10 * 24 * time.Hour}, // 10GB, 10 days old
		{"userB", "model2", "Q4_K_M", 1024 * 1024 * 50, 30 * 24 * time.Hour},        // 50MB, 30 days old
	}

	for _, m := range testModels {
		modelDir := filepath.Join(tmpDir, m.user, m.repo)
		if err := os.MkdirAll(modelDir, 0755); err != nil {
			t.Fatalf("Failed to create model dir: %v", err)
		}

		modelPath := filepath.Join(modelDir, m.quant+".gguf")
		if err := createTestFile(modelPath, m.size); err != nil {
			t.Fatalf("Failed to create model file: %v", err)
		}

		// Set modification time for age-based filtering
		modTime := time.Now().Add(-m.age)
		if err := os.Chtimes(modelPath, modTime, modTime); err != nil {
			t.Fatalf("Failed to set mod time: %v", err)
		}
	}

	tests := []struct {
		name       string
		pattern    string
		olderThan  time.Duration
		largerThan int64
		wantCount  int
		wantModels []string // user/repo:quant
	}{
		{
			name:      "match all",
			pattern:   "*",
			wantCount: 5,
		},
		{
			name:       "match user",
			pattern:    "userA/*",
			wantCount:  3,
			wantModels: []string{"userA/model1:Q4_K_M", "userA/model1:Q8_0", "userA/model2:Q4_K_M"},
		},
		{
			name:       "match specific repo",
			pattern:    "userA/model1",
			wantCount:  2,
			wantModels: []string{"userA/model1:Q4_K_M", "userA/model1:Q8_0"},
		},
		{
			name:       "match specific quant",
			pattern:    "userA/model1:Q4_K_M",
			wantCount:  1,
			wantModels: []string{"userA/model1:Q4_K_M"},
		},
		{
			name:      "older than 7 days",
			pattern:   "*",
			olderThan: 7 * 24 * time.Hour,
			wantCount: 2, // userB/model1 (10d) and userB/model2 (30d)
		},
		{
			name:       "larger than 1GB",
			pattern:    "*",
			largerThan: 1024 * 1024 * 1024,
			wantCount:  2, // userA/model2 (5GB) and userB/model1 (10GB)
		},
		{
			name:       "combined filters",
			pattern:    "*",
			olderThan:  3 * 24 * time.Hour,
			largerThan: 1024 * 1024 * 1024,
			wantCount:  1, // only userB/model1 (10GB, 10 days old)
		},
		{
			name:      "no matches",
			pattern:   "nonexistent/*",
			wantCount: 0,
		},
		{
			name:      "pattern with filter",
			pattern:   "userA/*",
			olderThan: 3 * 24 * time.Hour,
			wantCount: 1, // only userA/model1:Q8_0 (5 days old)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			models, err := findModelsInDir(tmpDir, tt.pattern, tt.olderThan, tt.largerThan)
			if err != nil {
				t.Fatalf("findModels() error = %v", err)
			}

			if len(models) != tt.wantCount {
				names := make([]string, len(models))
				for i, m := range models {
					names[i] = m.User + "/" + m.Repo + ":" + m.Quant
				}
				t.Errorf("findModels() returned %d models %v, want %d", len(models), names, tt.wantCount)
			}

			if tt.wantModels != nil {
				got := make(map[string]bool)
				for _, m := range models {
					got[m.User+"/"+m.Repo+":"+m.Quant] = true
				}
				for _, want := range tt.wantModels {
					if !got[want] {
						t.Errorf("findModels() missing expected model %s", want)
					}
				}
			}
		})
	}
}

func TestFindModelsSplitFiles(t *testing.T) {
	// Create a temporary models directory
	tmpDir := t.TempDir()

	// Create a split model (user/repo/quant/model-NNNNN-of-MMMMM.gguf)
	splitDir := filepath.Join(tmpDir, "userA", "bigmodel", "Q4_K_M")
	if err := os.MkdirAll(splitDir, 0755); err != nil {
		t.Fatalf("Failed to create split dir: %v", err)
	}

	// Create split files with different sizes
	splitFiles := []struct {
		name string
		size int64
	}{
		{"model-00001-of-00003.gguf", 1024 * 1024 * 100}, // 100MB
		{"model-00002-of-00003.gguf", 1024 * 1024 * 100}, // 100MB
		{"model-00003-of-00003.gguf", 1024 * 1024 * 50},  // 50MB (last split smaller)
	}

	for _, f := range splitFiles {
		path := filepath.Join(splitDir, f.name)
		if err := createTestFile(path, f.size); err != nil {
			t.Fatalf("Failed to create split file: %v", err)
		}
	}

	// Create a regular single-file model for comparison
	singleDir := filepath.Join(tmpDir, "userB", "smallmodel")
	if err := os.MkdirAll(singleDir, 0755); err != nil {
		t.Fatalf("Failed to create single model dir: %v", err)
	}
	singlePath := filepath.Join(singleDir, "Q4_K_M.gguf")
	if err := createTestFile(singlePath, 1024*1024*50); err != nil { // 50MB
		t.Fatalf("Failed to create single model file: %v", err)
	}

	// Test: find all models
	models, err := findModelsInDir(tmpDir, "*", 0, 0)
	if err != nil {
		t.Fatalf("findModelsInDir() error = %v", err)
	}

	if len(models) != 2 {
		names := make([]string, len(models))
		for i, m := range models {
			names[i] = m.User + "/" + m.Repo + ":" + m.Quant
		}
		t.Fatalf("findModelsInDir() returned %d models %v, want 2", len(models), names)
	}

	// Check that split model is found with correct total size
	var splitModel *ModelInfo
	var singleModel *ModelInfo
	for i := range models {
		if models[i].User == "userA" && models[i].Repo == "bigmodel" {
			splitModel = &models[i]
		}
		if models[i].User == "userB" && models[i].Repo == "smallmodel" {
			singleModel = &models[i]
		}
	}

	if splitModel == nil {
		t.Fatal("Split model not found")
		return
	}
	if splitModel.Quant != "Q4_K_M" {
		t.Errorf("Split model quant = %q, want Q4_K_M", splitModel.Quant)
	}
	// Total size should be 100+100+50 = 250MB
	expectedSize := int64(1024 * 1024 * 250)
	if splitModel.Size != expectedSize {
		t.Errorf("Split model size = %d, want %d", splitModel.Size, expectedSize)
	}

	if singleModel == nil {
		t.Fatal("Single model not found")
		return
	}
	if singleModel.Size != 1024*1024*50 {
		t.Errorf("Single model size = %d, want %d", singleModel.Size, 1024*1024*50)
	}

	// Test: filter by size (larger than 200MB should only match split model)
	models, err = findModelsInDir(tmpDir, "*", 0, 1024*1024*200)
	if err != nil {
		t.Fatalf("findModelsInDir() error = %v", err)
	}
	if len(models) != 1 {
		t.Errorf("findModelsInDir() with size filter returned %d models, want 1", len(models))
	}
	if len(models) == 1 && models[0].User != "userA" {
		t.Errorf("findModelsInDir() with size filter returned wrong model: %s/%s", models[0].User, models[0].Repo)
	}
}

func TestCleanEmptyDir(t *testing.T) {
	t.Run("removes empty directory", func(t *testing.T) {
		tmpDir := t.TempDir()
		emptyDir := filepath.Join(tmpDir, "empty")
		if err := os.Mkdir(emptyDir, 0755); err != nil {
			t.Fatalf("Failed to create dir: %v", err)
		}

		cleanEmptyDir(emptyDir)

		if _, err := os.Stat(emptyDir); !os.IsNotExist(err) {
			t.Error("cleanEmptyDir() did not remove empty directory")
		}
	})

	t.Run("keeps non-empty directory", func(t *testing.T) {
		tmpDir := t.TempDir()
		nonEmptyDir := filepath.Join(tmpDir, "nonempty")
		if err := os.Mkdir(nonEmptyDir, 0755); err != nil {
			t.Fatalf("Failed to create dir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(nonEmptyDir, "file.txt"), []byte("data"), 0644); err != nil {
			t.Fatalf("Failed to create file: %v", err)
		}

		cleanEmptyDir(nonEmptyDir)

		if _, err := os.Stat(nonEmptyDir); os.IsNotExist(err) {
			t.Error("cleanEmptyDir() removed non-empty directory")
		}
	})

	t.Run("handles non-existent directory", func(t *testing.T) {
		// Should not panic
		cleanEmptyDir("/nonexistent/path/that/does/not/exist")
	})
}

// createTestFile creates a sparse file of the given size
func createTestFile(path string, size int64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if size > 0 {
		if err := f.Truncate(size); err != nil {
			return err
		}
	}
	return nil
}
