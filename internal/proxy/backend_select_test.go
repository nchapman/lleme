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
		name    string
		quant   string
		kind    string
		wantErr string
	}{
		{"legacy empty defaults to gguf", "q_legacy", "", ""},
		{"gguf returns llama runtime", "q_gguf", hf.BackendGGUF, ""},
		{"mlx not yet implemented", "q_mlx", hf.BackendMLX, "MLX backend not yet implemented"},
		{"unknown kind errors", "q_bogus", "bogus", "unknown backend kind"},
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
				if rt.Kind() != BackendKindLlama {
					t.Errorf("kind = %v, want llama", rt.Kind())
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
