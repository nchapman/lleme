package cmd

import (
	"testing"

	"github.com/nchapman/lleme/internal/hf"
)

func TestDecideBackendsFromInventory(t *testing.T) {
	tests := []struct {
		name   string
		models []hf.LocalModel
		want   neededBackends
	}{
		{
			name:   "empty inventory defaults to llama baseline",
			models: nil,
			want:   neededBackends{Llama: true},
		},
		{
			name:   "gguf only installs llama",
			models: []hf.LocalModel{{Backend: hf.BackendGGUF}},
			want:   neededBackends{Llama: true},
		},
		{
			name:   "mlx only installs SwiftLM",
			models: []hf.LocalModel{{Backend: hf.BackendMLX}},
			want:   neededBackends{SwiftLM: true},
		},
		{
			name:   "mixed inventory installs both",
			models: []hf.LocalModel{{Backend: hf.BackendGGUF}, {Backend: hf.BackendMLX}},
			want:   neededBackends{Llama: true, SwiftLM: true},
		},
		{
			name:   "unknown backend falls back to llama",
			models: []hf.LocalModel{{Backend: "unknown"}},
			want:   neededBackends{Llama: true},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := decideBackendsFromInventory(tt.models)
			if got != tt.want {
				t.Errorf("decideBackendsFromInventory = %+v, want %+v", got, tt.want)
			}
		})
	}
}
