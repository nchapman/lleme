package proxy

import (
	"os"
	"strings"
	"testing"

	"github.com/nchapman/lleme/internal/hf"
)

func TestSelectRuntime(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	user, repo := "u", "r"
	if err := os.MkdirAll(hf.GetModelPath(user, repo), 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	m := NewModelManager(DefaultConfig(), nil)

	tests := []struct {
		name     string
		quant    string
		kind     string
		wantErr  string
		wantKind BackendKind
	}{
		{"legacy empty defaults to gguf", "q_legacy", "", "", BackendKindLlama},
		{"gguf returns llama runtime", "q_gguf", hf.BackendGGUF, "", BackendKindLlama},
		{"mlx returns swiftlm runtime", "q_mlx", hf.BackendMLX, "", BackendKindMLX},
		{"unknown kind errors", "q_bogus", "bogus", "unrecognized backend kind", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			quant := tt.quant
			if tt.kind != "" {
				if err := hf.SetBackendKind(user, repo, quant, tt.kind); err != nil {
					t.Fatalf("SetBackendKind: %v", err)
				}
			}

			rt, err := m.selectRuntime(user, repo, quant)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("selectRuntime: %v", err)
				}
				if rt.Kind() != tt.wantKind {
					t.Errorf("kind = %v, want %v", rt.Kind(), tt.wantKind)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("err = %v, want containing %q", err, tt.wantErr)
			}
			if rt != nil {
				t.Error("expected nil runtime on error")
			}
		})
	}
}
