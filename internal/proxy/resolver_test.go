package proxy

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/nchapman/lleme/internal/hf"
)

func TestLevenshtein(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"a", "", 1},
		{"", "a", 1},
		{"abc", "abc", 0},
		{"abc", "abd", 1},
		{"abc", "adc", 1},
		{"abc", "abcd", 1},
		{"kitten", "sitting", 3},
		{"llama", "lama", 1},
	}

	for _, tt := range tests {
		t.Run(tt.a+"_"+tt.b, func(t *testing.T) {
			got := levenshtein(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("levenshtein(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestPickBestQuant(t *testing.T) {
	models := []DownloadedModel{
		{Quant: "Q8_0"},
		{Quant: "Q4_K_M"},
		{Quant: "Q3_K_S"},
	}

	best := pickBestQuant(models)
	if best == nil {
		t.Fatal("pickBestQuant returned nil")
	}
	if best.Quant != "Q4_K_M" {
		t.Errorf("pickBestQuant() = %s, want Q4_K_M", best.Quant)
	}
}

func TestAllSameRepo(t *testing.T) {
	tests := []struct {
		name   string
		models []DownloadedModel
		want   bool
	}{
		{
			name:   "empty",
			models: []DownloadedModel{},
			want:   true,
		},
		{
			name: "single",
			models: []DownloadedModel{
				{User: "user", Repo: "repo"},
			},
			want: true,
		},
		{
			name: "same repo",
			models: []DownloadedModel{
				{User: "user", Repo: "repo", Quant: "Q4_K_M"},
				{User: "user", Repo: "repo", Quant: "Q8_0"},
			},
			want: true,
		},
		{
			name: "different repos",
			models: []DownloadedModel{
				{User: "user1", Repo: "repo1"},
				{User: "user2", Repo: "repo2"},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := allSameRepo(tt.models)
			if got != tt.want {
				t.Errorf("allSameRepo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetQuantPriority(t *testing.T) {
	// Q4_K_M should have lower (better) priority than Q8_0
	q4 := hf.GetQuantPriority("Q4_K_M")
	q8 := hf.GetQuantPriority("Q8_0")

	if q4 >= q8 {
		t.Errorf("Q4_K_M priority (%d) should be lower than Q8_0 (%d)", q4, q8)
	}

	// Unknown quants should have high priority
	unknown := hf.GetQuantPriority("UNKNOWN")
	if unknown <= q8 {
		t.Errorf("Unknown quant priority (%d) should be higher than Q8_0 (%d)", unknown, q8)
	}
}

func TestModelResolverWithTempDir(t *testing.T) {
	// Create a temporary directory structure
	tmpDir := t.TempDir()

	// Create test model structure
	modelDir := filepath.Join(tmpDir, "bartowski", "Llama-3.2-3B-Instruct-GGUF")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Create a fake gguf file
	ggufPath := filepath.Join(modelDir, "Q4_K_M.gguf")
	if err := os.WriteFile(ggufPath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	// Create resolver with custom path
	resolver := &ModelResolver{modelsPath: tmpDir}

	// Test listing models
	models, err := resolver.ListDownloadedModels()
	if err != nil {
		t.Fatalf("ListDownloadedModels() error = %v", err)
	}

	if len(models) != 1 {
		t.Fatalf("ListDownloadedModels() returned %d models, want 1", len(models))
	}

	model := models[0]
	if model.User != "bartowski" {
		t.Errorf("User = %s, want bartowski", model.User)
	}
	if model.Repo != "Llama-3.2-3B-Instruct-GGUF" {
		t.Errorf("Repo = %s, want Llama-3.2-3B-Instruct-GGUF", model.Repo)
	}
	if model.Quant != "Q4_K_M" {
		t.Errorf("Quant = %s, want Q4_K_M", model.Quant)
	}
}

func TestListDownloadedModelsSkipsMMProj(t *testing.T) {
	tmpDir := t.TempDir()

	modelDir := filepath.Join(tmpDir, "unsloth", "Qwen3.5-9B-GGUF")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "UD-Q4_K_XL.gguf"), []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "UD-Q4_K_XL-mmproj.gguf"), []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	resolver := &ModelResolver{modelsPath: tmpDir}
	models, err := resolver.ListDownloadedModels()
	if err != nil {
		t.Fatalf("ListDownloadedModels() error = %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("ListDownloadedModels() returned %d models, want 1 (mmproj should be skipped)", len(models))
	}
	if models[0].Quant != "UD-Q4_K_XL" {
		t.Errorf("Quant = %s, want UD-Q4_K_XL", models[0].Quant)
	}
}

// setupTestModels creates a test directory with multiple models for resolve testing
func setupTestModels(t *testing.T) *ModelResolver {
	tmpDir := t.TempDir()

	// Create multiple models to test various matching scenarios
	models := []struct {
		user, repo, quant string
	}{
		{"bartowski", "Llama-3.2-3B-Instruct-GGUF", "Q4_K_M"},
		{"bartowski", "Llama-3.2-3B-Instruct-GGUF", "Q8_0"},
		{"bartowski", "Mistral-7B-Instruct-v0.3-GGUF", "Q4_K_M"},
		{"mistralai", "Mistral-7B-Instruct-GGUF", "Q4_K_M"},
		{"microsoft", "phi-2-gguf", "Q4_0"},
	}

	for _, m := range models {
		dir := filepath.Join(tmpDir, m.user, m.repo)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, m.quant+".gguf")
		if err := os.WriteFile(path, []byte("fake"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	return &ModelResolver{modelsPath: tmpDir}
}

func TestResolve(t *testing.T) {
	resolver := setupTestModels(t)

	tests := []struct {
		name          string
		query         string
		wantMatch     bool   // Should find a unique match
		wantAmbiguous bool   // Should be ambiguous (multiple different repos)
		wantModel     string // Expected model name (if unique match)
		wantCount     int    // Expected number of matches/suggestions
	}{
		// Exact matches
		{
			name:      "exact full name",
			query:     "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
			wantMatch: true,
			wantModel: "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
		},
		{
			name:      "exact full name case insensitive",
			query:     "bartowski/llama-3.2-3b-instruct-gguf:q4_k_m",
			wantMatch: true,
			wantModel: "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
		},

		// User/repo without quant - picks best quant
		{
			name:      "user/repo picks best quant",
			query:     "bartowski/Llama-3.2-3B-Instruct-GGUF",
			wantMatch: true,
			wantModel: "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", // Q4_K_M preferred over Q8_0
		},

		// Repo name only - unique
		{
			name:      "unique repo name",
			query:     "phi-2-gguf",
			wantMatch: true,
			wantModel: "microsoft/phi-2-gguf:Q4_0",
		},
		{
			name:      "unique repo name case insensitive",
			query:     "PHI-2-GGUF",
			wantMatch: true,
			wantModel: "microsoft/phi-2-gguf:Q4_0",
		},

		// Repo name - ambiguous (different users have similar repos)
		{
			name:          "ambiguous repo name",
			query:         "Mistral",
			wantAmbiguous: true,
			wantCount:     2, // bartowski and mistralai both have Mistral
		},

		// Contains match - unique
		{
			name:      "contains match unique",
			query:     "phi",
			wantMatch: true,
			wantModel: "microsoft/phi-2-gguf:Q4_0",
		},

		// Contains match - same repo different quants
		{
			name:      "contains match same repo",
			query:     "llama-3.2-3b",
			wantMatch: true,
			wantModel: "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", // picks best quant
		},

		// No match - should give suggestions
		{
			name:      "typo gives suggestions",
			query:     "lama", // typo for llama
			wantMatch: false,
			wantCount: 0, // No contains match, might have suggestions
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := resolver.Resolve(tt.query)
			if err != nil {
				t.Fatalf("Resolve() error = %v", err)
			}

			if tt.wantMatch {
				if result.Model == nil {
					t.Errorf("Expected match for %q but got none. Matches: %d, Suggestions: %d",
						tt.query, len(result.Matches), len(result.Suggestions))
					return
				}
				if result.Model.FullName != tt.wantModel {
					t.Errorf("Resolve(%q) = %s, want %s", tt.query, result.Model.FullName, tt.wantModel)
				}
			} else if tt.wantAmbiguous {
				if result.Model != nil {
					t.Errorf("Expected ambiguous for %q but got match: %s", tt.query, result.Model.FullName)
				}
				if len(result.Matches) < 2 {
					t.Errorf("Expected multiple matches for %q but got %d", tt.query, len(result.Matches))
				}
				if tt.wantCount > 0 && len(result.Matches) != tt.wantCount {
					t.Errorf("Expected %d matches for %q but got %d", tt.wantCount, tt.query, len(result.Matches))
				}
			}
		})
	}
}
