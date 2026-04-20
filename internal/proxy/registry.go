package proxy

// AllRuntimes returns one instance of every known Runtime. The instances are
// constructed with a nil config — callers that only need backend-agnostic
// metadata (HFAppName, Kind) can use them directly; anything that touches
// options or starts a process should go through selectRuntime instead.
//
// This is the extension point for a new backend: implement Runtime, add a
// constructor call here, and discovery/search picks it up automatically.
func AllRuntimes() []Runtime {
	return []Runtime{
		NewLlamaRuntime(nil),
		NewSwiftLMRuntime(nil),
	}
}

// HFAppNames returns the HuggingFace "apps=" filter values for every
// registered runtime, in registration order. Used by search and trending
// to avoid hardcoding "llama.cpp" at the call site.
func HFAppNames() []string {
	rts := AllRuntimes()
	out := make([]string, 0, len(rts))
	for _, r := range rts {
		if name := r.HFAppName(); name != "" {
			out = append(out, name)
		}
	}
	return out
}
