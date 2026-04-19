package hf

import (
	"testing"
)

func TestParseQuantization(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		want     string
	}{
		{"Q4_K_M", "model-Q4_K_M.gguf", "Q4_K_M"},
		{"Q5_K_S", "model-Q5_K_S.gguf", "Q5_K_S"},
		{"Q6_K", "model-Q6_K.gguf", "Q6_K"},
		{"Q8_0", "model-Q8_0.gguf", "Q8_0"},
		{"Q4_0", "model-Q4_0.gguf", "Q4_0"},
		{"FP16", "model-FP16.gguf", "FP16"},
		{"FP32", "model-FP32.gguf", "FP32"},
		{"F16", "model-F16.gguf", "F16"},
		{"F32", "model-F32.gguf", "F32"},
		{"BF16", "model-BF16.gguf", "BF16"},
		{"I8", "model-I8.gguf", "I8"},
		{"I4", "model-I4.gguf", "I4"},
		{"TQ1_0 ternary", "model-TQ1_0.gguf", "TQ1_0"},
		{"TQ2_0 ternary", "model-TQ2_0.gguf", "TQ2_0"},
		{"IQ4_XS imatrix", "model-IQ4_XS.gguf", "IQ4_XS"},
		{"IQ2_XXS imatrix", "model-IQ2_XXS.gguf", "IQ2_XXS"},
		{"UD-Q4_K_XL prefix", "model-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"},
		{"UD-IQ1_M prefix", "model-UD-IQ1_M.gguf", "UD-IQ1_M"},
		{"underscore separator", "model_Q4_K_M.gguf", "Q4_K_M"},
		{"dot separator", "model.Q4_K_M.gguf", "Q4_K_M"},
		{"no quantization", "model.gguf", ""},
		{"not a gguf file", "model.bin", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseQuantization(tt.filename)
			if got != tt.want {
				t.Errorf("ParseQuantization(%q) = %q, want %q", tt.filename, got, tt.want)
			}
		})
	}
}

func TestExtractQuantizations(t *testing.T) {
	files := []FileTree{
		{Path: "model-Q4_K_M.gguf", Size: 4000000000},
		{Path: "model-Q5_K_S.gguf", Size: 5000000000},
		{Path: "model-Q6_K.gguf", Size: 6000000000},
		{Path: "README.md", Size: 1024},
		{Path: "config.json", Size: 2048},
		{Path: "tokenizer.json", Size: 512},
	}

	quants := ExtractQuantizations(files)

	if len(quants) != 3 {
		t.Errorf("ExtractQuantizations() got %d quants, want 3", len(quants))
	}

	wantNames := []string{"Q4_K_M", "Q5_K_S", "Q6_K"}
	for i, want := range wantNames {
		if i >= len(quants) {
			break
		}
		if quants[i].Name != want {
			t.Errorf("ExtractQuantizations()[%d].Name = %v, want %v", i, quants[i].Name, want)
		}
	}
}

func TestExtractQuantizationsEmpty(t *testing.T) {
	files := []FileTree{
		{Path: "README.md", Size: 1024},
		{Path: "config.json", Size: 2048},
	}

	quants := ExtractQuantizations(files)

	if len(quants) != 0 {
		t.Errorf("ExtractQuantizations() got %d quants, want 0", len(quants))
	}
}

func TestExtractQuantizationsNoSuffix(t *testing.T) {
	files := []FileTree{
		{Path: "gemma-7b.gguf", Size: 34200000000},
		{Path: "README.md", Size: 1024},
		{Path: "config.json", Size: 2048},
	}

	quants := ExtractQuantizations(files)

	if len(quants) != 1 {
		t.Errorf("ExtractQuantizations() got %d quants, want 1", len(quants))
	}

	if quants[0].Name != "default" {
		t.Errorf("ExtractQuantizations()[0].Name = %v, want 'default'", quants[0].Name)
	}

	if quants[0].Tag != "latest" {
		t.Errorf("ExtractQuantizations()[0].Tag = %v, want 'latest'", quants[0].Tag)
	}

	if quants[0].File != "gemma-7b.gguf" {
		t.Errorf("ExtractQuantizations()[0].File = %v, want 'gemma-7b.gguf'", quants[0].File)
	}
}

func TestExtractQuantizationsMixed(t *testing.T) {
	files := []FileTree{
		{Path: "model.gguf", Size: 10000000000},
		{Path: "model-Q4_K_M.gguf", Size: 4000000000},
		{Path: "README.md", Size: 1024},
	}

	quants := ExtractQuantizations(files)

	if len(quants) != 2 {
		t.Errorf("ExtractQuantizations() got %d quants, want 2", len(quants))
	}

	// Check that both default and Q4_K_M are found
	names := make(map[string]bool)
	for _, q := range quants {
		names[q.Name] = true
	}

	if !names["default"] {
		t.Error("ExtractQuantizations() missing 'default' quant")
	}
	if !names["Q4_K_M"] {
		t.Error("ExtractQuantizations() missing 'Q4_K_M' quant")
	}
}

func TestExtractQuantizationsF16Tag(t *testing.T) {
	// Test that F16 files preserve the original name
	files := []FileTree{
		{Path: "model-F16.gguf", Size: 65000000000},
		{Path: "README.md", Size: 1024},
	}

	quants := ExtractQuantizations(files)

	if len(quants) != 1 {
		t.Fatalf("ExtractQuantizations() got %d quants, want 1", len(quants))
	}

	// Name and Tag should both be F16 (no normalization)
	if quants[0].Name != "F16" {
		t.Errorf("ExtractQuantizations()[0].Name = %v, want 'F16'", quants[0].Name)
	}
	if quants[0].Tag != "F16" {
		t.Errorf("ExtractQuantizations()[0].Tag = %v, want 'F16'", quants[0].Tag)
	}
}

func TestExtractQuantizationsFromDirectories(t *testing.T) {
	// Test extraction from directories (like unsloth/gpt-oss-120b-GGUF)
	files := []FileTree{
		{Path: "Q4_K_M", Type: "directory"},
		{Path: "Q5_K_S", Type: "directory"},
		{Path: "Q8_0", Type: "directory"},
		{Path: "gpt-oss-120b-F16.gguf", Size: 65000000000},
		{Path: "README.md", Size: 1024},
		{Path: "config.json", Size: 2048},
	}

	quants := ExtractQuantizations(files)

	if len(quants) != 4 {
		t.Fatalf("ExtractQuantizations() got %d quants, want 4", len(quants))
	}

	// Check that all expected quants are found
	names := make(map[string]string) // name -> tag
	for _, q := range quants {
		names[q.Name] = q.Tag
	}

	// Directory-based quants
	if tag, ok := names["Q4_K_M"]; !ok {
		t.Error("ExtractQuantizations() missing 'Q4_K_M' quant")
	} else if tag != "Q4_K_M" {
		t.Errorf("Q4_K_M tag = %q, want 'Q4_K_M'", tag)
	}

	if _, ok := names["Q5_K_S"]; !ok {
		t.Error("ExtractQuantizations() missing 'Q5_K_S' quant")
	}

	if _, ok := names["Q8_0"]; !ok {
		t.Error("ExtractQuantizations() missing 'Q8_0' quant")
	}

	// File-based quant (F16 stays as F16, no normalization)
	if tag, ok := names["F16"]; !ok {
		t.Error("ExtractQuantizations() missing 'F16' quant")
	} else if tag != "F16" {
		t.Errorf("F16 tag = %q, want 'F16'", tag)
	}
}

func TestExtractQuantizationsDirectoryCasing(t *testing.T) {
	// Test that directory names are uppercased for display but Tag preserves original for API
	files := []FileTree{
		{Path: "UD-Q4_K_XL", Type: "directory"},
		{Path: "q4_k_m", Type: "directory"}, // lowercase - should be uppercased
		{Path: "BF16", Type: "directory"},
		{Path: "ud-iq1_m", Type: "directory"}, // lowercase UD variant
	}

	quants := ExtractQuantizations(files)

	if len(quants) != 4 {
		t.Fatalf("ExtractQuantizations() got %d quants, want 4", len(quants))
	}

	// Build maps for checking
	names := make(map[string]string) // name -> tag
	for _, q := range quants {
		names[q.Name] = q.Tag
	}

	// UD-Q4_K_XL: Name uppercased, Tag preserves original
	if tag, ok := names["UD-Q4_K_XL"]; !ok {
		t.Error("missing 'UD-Q4_K_XL' quant")
	} else if tag != "UD-Q4_K_XL" {
		t.Errorf("UD-Q4_K_XL tag = %q, want 'UD-Q4_K_XL'", tag)
	}

	// q4_k_m: Name should be uppercased to Q4_K_M, Tag preserves lowercase
	if tag, ok := names["Q4_K_M"]; !ok {
		t.Error("missing 'Q4_K_M' quant (from lowercase 'q4_k_m' directory)")
	} else if tag != "q4_k_m" {
		t.Errorf("Q4_K_M tag = %q, want 'q4_k_m' (original case)", tag)
	}

	// BF16: already uppercase
	if tag, ok := names["BF16"]; !ok {
		t.Error("missing 'BF16' quant")
	} else if tag != "BF16" {
		t.Errorf("BF16 tag = %q, want 'BF16'", tag)
	}

	// ud-iq1_m: Name uppercased to UD-IQ1_M, Tag preserves lowercase
	if tag, ok := names["UD-IQ1_M"]; !ok {
		t.Error("missing 'UD-IQ1_M' quant (from lowercase 'ud-iq1_m' directory)")
	} else if tag != "ud-iq1_m" {
		t.Errorf("UD-IQ1_M tag = %q, want 'ud-iq1_m' (original case)", tag)
	}
}

func TestGetBestQuantization(t *testing.T) {
	tests := []struct {
		name   string
		quants []Quantization
		want   string
	}{
		{
			name: "Q4_K_M preferred",
			quants: []Quantization{
				{Name: "Q5_K_S"},
				{Name: "Q4_K_M"},
				{Name: "Q6_K"},
			},
			want: "Q4_K_M",
		},
		{
			name: "Q5_K_M preferred",
			quants: []Quantization{
				{Name: "Q6_K"},
				{Name: "Q5_K_M"},
				{Name: "Q8_0"},
			},
			want: "Q5_K_M",
		},
		{
			name: "no preferred quant",
			quants: []Quantization{
				{Name: "I8"},
				{Name: "I4"},
			},
			want: "I8",
		},
		{
			name:   "empty",
			quants: []Quantization{},
			want:   "",
		},
		{
			name: "default only",
			quants: []Quantization{
				{Name: "default"},
			},
			want: "default",
		},
		{
			name: "default with preferred quants",
			quants: []Quantization{
				{Name: "default"},
				{Name: "Q4_K_M"},
			},
			want: "Q4_K_M",
		},
		{
			name: "UD variant preferred over standard",
			quants: []Quantization{
				{Name: "Q4_K_M"},
				{Name: "UD-Q4_K_XL"},
				{Name: "Q5_K_M"},
			},
			want: "UD-Q4_K_XL",
		},
		{
			name: "UD-Q5_K_XL preferred over Q4_K_S",
			quants: []Quantization{
				{Name: "Q4_K_S"},
				{Name: "UD-Q5_K_XL"},
				{Name: "Q6_K"},
			},
			want: "UD-Q5_K_XL",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetBestQuantization(tt.quants)
			if got != tt.want {
				t.Errorf("GetBestQuantization() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSortQuantizations(t *testing.T) {
	quants := []Quantization{
		{Name: "Q8_0", Size: 8000000000},
		{Name: "Q4_K_M", Size: 4000000000},
		{Name: "BF16", Size: 16000000000},
		{Name: "Q5_K_S", Size: 5000000000},
	}

	sorted := SortQuantizations(quants)

	// Alphabetical order
	wantOrder := []string{"BF16", "Q4_K_M", "Q5_K_S", "Q8_0"}
	for i, want := range wantOrder {
		if sorted[i].Name != want {
			t.Errorf("SortQuantizations()[%d].Name = %v, want %v", i, sorted[i].Name, want)
		}
	}
}

func TestFindQuantization(t *testing.T) {
	quants := []Quantization{
		{Name: "Q4_K_M", File: "model.Q4_K_M.gguf", Size: 4000000000},
		{Name: "Q5_K_S", File: "model.Q5_K_S.gguf", Size: 5000000000},
		{Name: "Q6_K", File: "model.Q6_K.gguf", Size: 6000000000},
	}

	tests := []struct {
		name    string
		quant   string
		want    Quantization
		wantErr bool
	}{
		{
			name:  "found",
			quant: "Q4_K_M",
			want:  Quantization{Name: "Q4_K_M", File: "model.Q4_K_M.gguf", Size: 4000000000},
		},
		{
			name:  "case insensitive",
			quant: "q4_k_m",
			want:  Quantization{Name: "Q4_K_M", File: "model.Q4_K_M.gguf", Size: 4000000000},
		},
		{
			name:    "not found",
			quant:   "Q9_0",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, found := FindQuantization(quants, tt.quant)
			if found != !tt.wantErr {
				t.Errorf("FindQuantization() found = %v, want %v", found, !tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("FindQuantization() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFindQuantizationCaseInsensitive(t *testing.T) {
	// Test case-insensitive lookup - users may type lowercase
	// Name and Tag are the same (no normalization)
	quants := []Quantization{
		{Name: "F16", Tag: "F16", File: "model-F16.gguf", Size: 65000000000},
		{Name: "UD-Q4_K_XL", Tag: "UD-Q4_K_XL", File: "model-UD-Q4_K_XL.gguf", Size: 16000000000},
	}

	// Should find F16 by exact name
	got, found := FindQuantization(quants, "F16")
	if !found {
		t.Fatal("FindQuantization() should find F16")
	}
	if got.Name != "F16" {
		t.Errorf("FindQuantization().Name = %v, want F16", got.Name)
	}

	// Should find by lowercase (case insensitive)
	got, found = FindQuantization(quants, "f16")
	if !found {
		t.Fatal("FindQuantization() should find F16 by lowercase 'f16'")
	}
	if got.Name != "F16" {
		t.Errorf("FindQuantization().Name = %v, want F16", got.Name)
	}

	// Should find UD variant with hyphen
	got, found = FindQuantization(quants, "UD-Q4_K_XL")
	if !found {
		t.Fatal("FindQuantization() should find UD-Q4_K_XL")
	}
	if got.Name != "UD-Q4_K_XL" {
		t.Errorf("FindQuantization().Name = %v, want UD-Q4_K_XL", got.Name)
	}

	// Should find UD variant case insensitive
	got, found = FindQuantization(quants, "ud-q4_k_xl")
	if !found {
		t.Fatal("FindQuantization() should find UD-Q4_K_XL by lowercase")
	}
	if got.Name != "UD-Q4_K_XL" {
		t.Errorf("FindQuantization().Name = %v, want UD-Q4_K_XL", got.Name)
	}
}

func TestGetQuantPriority(t *testing.T) {
	tests := []struct {
		name  string
		quant string
		want  int
	}{
		{"Q4_K_M uppercase", "Q4_K_M", 1},
		{"Q4_K_M lowercase", "q4_k_m", 1},
		{"Q4_K_M mixed case", "Q4_k_M", 1},
		{"UD-Q4_K_XL uppercase", "UD-Q4_K_XL", 0},
		{"UD-Q4_K_XL lowercase", "ud-q4_k_xl", 0},
		{"Q8_0 uppercase", "Q8_0", 9},
		{"Q8_0 lowercase", "q8_0", 9},
		{"unknown quant", "UNKNOWN", 1000},
		{"empty string", "", 1000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetQuantPriority(tt.quant)
			if got != tt.want {
				t.Errorf("GetQuantPriority(%q) = %d, want %d", tt.quant, got, tt.want)
			}
		})
	}
}

func TestGetQuantPriorityCaseConsistency(t *testing.T) {
	// Verify that all variations of the same quant return the same priority
	variants := []struct {
		quants []string
	}{
		{[]string{"Q4_K_M", "q4_k_m", "Q4_K_m", "q4_k_M"}},
		{[]string{"UD-Q4_K_XL", "ud-q4_k_xl", "Ud-Q4_k_xl"}},
		{[]string{"FP16", "fp16", "Fp16"}},
		{[]string{"BF16", "bf16", "Bf16"}},
		{[]string{"IQ4_XS", "iq4_xs", "Iq4_Xs"}},
		{[]string{"TQ1_0", "tq1_0", "Tq1_0"}},
	}

	for _, tt := range variants {
		t.Run(tt.quants[0], func(t *testing.T) {
			expected := GetQuantPriority(tt.quants[0])
			for _, q := range tt.quants[1:] {
				got := GetQuantPriority(q)
				if got != expected {
					t.Errorf("GetQuantPriority(%q) = %d, want %d (same as %q)", q, got, expected, tt.quants[0])
				}
			}
		})
	}
}
