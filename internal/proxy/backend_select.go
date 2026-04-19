package proxy

import (
	"fmt"

	"github.com/nchapman/lleme/internal/hf"
)

// selectRuntime resolves a Runtime for the given model based on the backend
// kind recorded in metadata.yaml at pull time. Legacy metadata without a
// backend field is treated as "gguf" so existing installs keep working.
//
// Returns an error when the recorded kind has no implementation yet (e.g.
// MLX before the SwiftLM runtime lands) so callers surface a clear message
// instead of silently launching the wrong backend.
func (m *ModelManager) selectRuntime(user, repo, quant string) (Runtime, error) {
	kind, err := hf.GetBackendKind(user, repo, quant)
	if err != nil {
		return nil, fmt.Errorf("reading backend metadata for %s/%s:%s: %w", user, repo, quant, err)
	}
	switch kind {
	case hf.BackendGGUF:
		return NewLlamaRuntime(m.appConfig), nil
	case hf.BackendMLX:
		return nil, fmt.Errorf("MLX backend not yet implemented (model %s/%s:%s)", user, repo, quant)
	default:
		return nil, fmt.Errorf("unknown backend kind %q for %s/%s:%s", kind, user, repo, quant)
	}
}
