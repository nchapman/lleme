package cmd

import (
	"github.com/nchapman/lleme/internal/llama"
)

// ensureLlamaBackend installs llama.cpp when it isn't present yet. lleme is
// llama.cpp-only, so this is the single bootstrap path for every command
// that needs to serve or pull GGUF models.
func ensureLlamaBackend() error {
	if llama.IsInstalled() {
		return nil
	}
	return ensureLlamaInstalled()
}
