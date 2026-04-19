package hf

import (
	"testing"
)

func TestSplitFilePattern(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		want     bool
	}{
		{
			name:     "split file 1 of 2",
			filename: "model-00001-of-00002.gguf",
			want:     true,
		},
		{
			name:     "split file 2 of 2",
			filename: "model-00002-of-00002.gguf",
			want:     true,
		},
		{
			name:     "split file 1 of 10",
			filename: "model-00001-of-00010.gguf",
			want:     true,
		},
		{
			name:     "split file 10 of 10",
			filename: "model-00010-of-00010.gguf",
			want:     true,
		},
		{
			name:     "complex name",
			filename: "gpt-oss-120b-Q4_K_M-00001-of-00003.gguf",
			want:     true,
		},
		{
			name:     "not a split file",
			filename: "model.gguf",
			want:     false,
		},
		{
			name:     "partial match - wrong format",
			filename: "model-001-of-002.gguf",
			want:     false,
		},
		{
			name:     "partial match - missing gguf",
			filename: "model-00001-of-00002",
			want:     false,
		},
		{
			name:     "regular quantized file",
			filename: "model-Q4_K_M.gguf",
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SplitFilePattern.MatchString(tt.filename)
			if got != tt.want {
				t.Errorf("SplitFilePattern.MatchString(%q) = %v, want %v", tt.filename, got, tt.want)
			}
		})
	}
}

func TestParseSplitFilename(t *testing.T) {
	tests := []struct {
		name       string
		filename   string
		wantNil    bool
		wantPrefix string
		wantNo     int
		wantCount  int
	}{
		{
			name:       "simple split file",
			filename:   "model-00001-of-00003.gguf",
			wantPrefix: "model",
			wantNo:     0,
			wantCount:  3,
		},
		{
			name:       "split file with path",
			filename:   "Q4_K_M/gpt-120b-Q4_K_M-00002-of-00005.gguf",
			wantPrefix: "Q4_K_M/gpt-120b-Q4_K_M",
			wantNo:     1,
			wantCount:  5,
		},
		{
			name:     "not a split file",
			filename: "model.gguf",
			wantNil:  true,
		},
		{
			name:     "wrong format",
			filename: "model-001-of-002.gguf",
			wantNil:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseSplitFilename(tt.filename)
			if tt.wantNil {
				if got != nil {
					t.Errorf("ParseSplitFilename(%q) = %+v, want nil", tt.filename, got)
				}
				return
			}
			if got == nil {
				t.Fatalf("ParseSplitFilename(%q) = nil, want non-nil", tt.filename)
			}
			if got.Prefix != tt.wantPrefix {
				t.Errorf("Prefix = %q, want %q", got.Prefix, tt.wantPrefix)
			}
			if got.SplitNo != tt.wantNo {
				t.Errorf("SplitNo = %d, want %d", got.SplitNo, tt.wantNo)
			}
			if got.SplitCount != tt.wantCount {
				t.Errorf("SplitCount = %d, want %d", got.SplitCount, tt.wantCount)
			}
		})
	}
}

func TestSplitPath(t *testing.T) {
	tests := []struct {
		prefix     string
		splitNo    int
		splitCount int
		want       string
	}{
		{
			prefix:     "model",
			splitNo:    0,
			splitCount: 2,
			want:       "model-00001-of-00002.gguf",
		},
		{
			prefix:     "model",
			splitNo:    1,
			splitCount: 2,
			want:       "model-00002-of-00002.gguf",
		},
		{
			prefix:     "Q4_K_M/gpt-120b-Q4_K_M",
			splitNo:    0,
			splitCount: 3,
			want:       "Q4_K_M/gpt-120b-Q4_K_M-00001-of-00003.gguf",
		},
		{
			prefix:     "Q4_K_M/gpt-120b-Q4_K_M",
			splitNo:    2,
			splitCount: 3,
			want:       "Q4_K_M/gpt-120b-Q4_K_M-00003-of-00003.gguf",
		},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := SplitPath(tt.prefix, tt.splitNo, tt.splitCount)
			if got != tt.want {
				t.Errorf("SplitPath() = %q, want %q", got, tt.want)
			}
		})
	}
}
