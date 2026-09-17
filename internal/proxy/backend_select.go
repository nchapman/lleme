package proxy

import (
	"fmt"

	"github.com/nchapman/lleme/internal/hf"
)

// selectRuntime resolves a Runtime for the given model based on the backend
// kind recorded in metadata.yaml at pull time. Legacy metadata without a
// backend field is treated as "gguf" so existing installs keep working.
// Unknown kinds surface as errors from hf.GetBackendKind so a corrupt or
// tampered metadata file can't silently dispatch to the wrong runtime.
func (m *ModelManager) selectRuntime(user, repo, quant string) (Runtime, error) {
	kind, err := hf.GetBackendKind(user, repo, quant)
	if err != nil {
		return nil, fmt.Errorf("reading backend metadata for %s/%s:%s: %w", user, repo, quant, err)
	}
	switch kind {
	case hf.BackendGGUF:
		return NewLlamaRuntime(m.appConfig), nil
	case hf.BackendMLX:
		// Metadata recorded by lleme versions that shipped the SwiftLM/MLX
		// backend. The runtime is gone; tell the user how to move forward
		// instead of failing with a generic unknown-kind error.
		return nil, fmt.Errorf("MLX models are no longer supported (the SwiftLM backend was removed); pull the GGUF build of this model instead")
	default:
		// Unreachable today: GetBackendKind already rejected anything that
		// isn't BackendGGUF / BackendMLX. Kept as a belt-and-suspenders
		// default so adding a new kind to hf without wiring it here still
		// fails closed.
		return nil, fmt.Errorf("no runtime registered for backend kind %q", kind)
	}
}
