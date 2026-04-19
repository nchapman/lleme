# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Build & Test Commands

```bash
make build                    # Build binary to ./lleme
make test                     # Run all tests
make check                    # Format + vet + test (run before committing)
go test ./cmd -run TestName   # Run single test (add -v for verbose)
go test ./internal/proxy      # Test specific package
```

Linting uses golangci-lint with `errcheck` and `unused` disabled.

## Architecture Overview

**lleme** is a Go CLI for running local LLMs via llama.cpp with Hugging Face model management. Built on Charmbracelet (bubbletea, lipgloss, glamour) for TUI and Cobra for CLI.

### Multi-Model Proxy (Core Architecture)

The central design is a reverse-proxy that manages multiple llama.cpp backend instances:

```
CLI/API → Proxy (port 11313) → Routes by model name → llama-server instances (:11314+)
```

Key packages in `internal/proxy/`:
- `server.go` - HTTP routing, reverse-proxy to backends
- `manager.go` - Model lifecycle (start/stop llama-server, LRU eviction, max 3 models)
- `idle.go` - Background monitor for auto-unloading idle models
- `ports.go` - Dynamic port allocation for backends

### Package Structure

- `cmd/` - Cobra CLI commands (run, pull, list, serve, status, etc.)
- `internal/config/` - Config loading/saving, personas (user-saved model settings)
- `internal/hf/` - Hugging Face API client, model downloads, quantization detection
- `internal/llama/` - llama.cpp binary management
- `internal/options/` - Settings resolver with layered precedence
- `internal/presets/` - Built-in curated sampling defaults per model family (embedded YAML)
- `internal/proxy/` - Multi-model proxy server
- `internal/server/` - Backend API client (OpenAI-compatible)
- `internal/tui/` - Bubbletea TUI (chat model, components, styles)
- `internal/ui/` - CLI utilities (spinner, progress, table, logger)

### Data Storage

All data lives in `~/.lleme/`:
- `config.yaml` - User configuration
- `models/` - Downloaded GGUF files (`user/repo/quant.gguf`)
- `bin/` - llama.cpp binaries
- `logs/` - Rotating log files

### Settings Resolver & Presets

Inference options flow through a layered resolver in `internal/options/resolver.go`. Precedence (highest to lowest):

**session > persona > preset > config > llama-server default**

- **session**: CLI flags for the current invocation.
- **persona**: user-saved named bundle of options (`config.Persona`).
- **preset**: built-in defaults curated per model family, matched by HuggingFace repo name.
- **config**: `~/.lleme/config.yaml` global defaults.
- **llama-server default**: the binary's own default if nothing is set.

The resolver uses **key-existence semantics**: a key with value `0` is distinct from an absent key. Explicit zeros (e.g. `min-p: 0.0`) are honored at every layer. Adjacent helpers: `ResolveFloat` / `ResolveInt` for the session-flag case (zero still means "not set" because CLI flag defaults are zero), `GetConfigFloat` / `GetConfigInt` for the persona-and-below case.

**Presets** live in `internal/presets/data/*.yaml`, embedded via `go:embed`. Each file defines sampling defaults for a model family and a list of `path.Match` globs against `user/repo`. Matching is case-insensitive and first-match-wins in **alphabetical order** of filename, so more specific files must sort before more general ones (e.g. `qwen3-coder.yaml` before `qwen3.yaml`).

When adding or editing a preset, read `internal/presets/data/README.md` first — it documents pattern conventions, ordering rules, what belongs under `options` (sampling params only, no runtime/hardware knobs), and when to split a family into multiple files. The `presets_test.go` table should include a realistic HF repo name per new preset, plus an ordering test if the preset could be masked by a more general one.

## Code Patterns

### Constructors
All major types use `NewXxx() *Type` pattern.

### Imports
Standard library → external deps → internal packages (blank lines between groups).

### Error Handling
Always wrap errors with context using `fmt.Errorf("context: %w", err)`.

### Testing
Table-driven tests with subtests:
```go
tests := []struct{ name, input, expected string }{...}
for _, tt := range tests {
    t.Run(tt.name, func(t *testing.T) {...})
}
```

### TUI
Bubbletea Model interface: `Init()`, `Update()`, `View()`. Components in `internal/tui/components/`.

### No Unnecessary Comments
Code should be self-documenting through clear naming and structure.
