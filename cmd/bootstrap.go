package cmd

import (
	"fmt"

	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/swiftlm"
	"github.com/nchapman/lleme/internal/ui"
)

// neededBackends describes which backend binaries must be present for the
// caller to operate. Emptiness means "nothing to install for this invocation"
// — callers should avoid dragging in a baseline llama download just to host
// an MLX-only workload.
type neededBackends struct {
	Llama   bool
	SwiftLM bool
}

// backendsForLocalModels inspects pulled models and reports which backend
// binaries are required to serve them. A nil/empty inventory returns
// {Llama: true} so a fresh install still gets the baseline runtime that
// handles the common case (GGUF models pulled shortly after).
//
// Errors from the inventory read are treated as best-effort: we fall back
// to {Llama: true, SwiftLM: IsSupported()} to avoid turning a transient
// filesystem hiccup into a startup failure.
func backendsForLocalModels() neededBackends {
	models, err := hf.ListLocalModels()
	if err != nil {
		return neededBackends{Llama: true, SwiftLM: swiftlm.IsSupported()}
	}
	return decideBackendsFromInventory(models)
}

// decideBackendsFromInventory is the pure-data core of backendsForLocalModels,
// split out for testing. Nothing here touches the filesystem or probes
// platform support — callers layer both in.
func decideBackendsFromInventory(models []hf.LocalModel) neededBackends {
	if len(models) == 0 {
		return neededBackends{Llama: true}
	}

	var need neededBackends
	for _, m := range models {
		switch m.Backend {
		case hf.BackendMLX:
			need.SwiftLM = true
		case hf.BackendGGUF:
			need.Llama = true
		default:
			need.Llama = true
		}
	}
	return need
}

// ensureBackends installs the backends described by `need`, skipping any
// that are already installed or unsupported on the current platform. An
// MLX model pulled on a non-Apple-Silicon host produces a warning (the
// model is not servable) but does not fail — GGUF models on the same host
// remain usable.
func ensureBackends(need neededBackends) error {
	if need.Llama && !llama.IsInstalled() {
		if err := ensureLlamaInstalled(); err != nil {
			return err
		}
	}
	if need.SwiftLM {
		if !swiftlm.IsSupported() {
			fmt.Println(ui.Warning("Skipping SwiftLM install: MLX requires macOS on Apple Silicon. Any pulled MLX models will not be servable here."))
			return nil
		}
		if !swiftlm.IsInstalled() {
			if err := ensureSwiftLMInstalled(); err != nil {
				return err
			}
		}
	}
	return nil
}

// ensureSwiftLMInstalled mirrors ensureLlamaInstalled for SwiftLM. Callers
// must check swiftlm.IsSupported() first — this helper does not gate
// silently because a caller that asked for SwiftLM on Linux should hear
// about it.
func ensureSwiftLMInstalled() error {
	fmt.Println("Installing SwiftLM...")
	fmt.Println()
	if _, err := swiftlm.InstallLatest(func(msg string) { fmt.Println(msg) }); err != nil {
		return fmt.Errorf("failed to install SwiftLM: %w", err)
	}
	fmt.Println()
	return nil
}
