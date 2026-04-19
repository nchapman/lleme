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

**lleme** is a Go CLI for running local LLMs via llama.cpp (GGUF) and SwiftLM (MLX, Apple Silicon only), with Hugging Face model management. Built on Charmbracelet (bubbletea, lipgloss, glamour) for TUI and Cobra for CLI.

### Multi-Model Proxy (Core Architecture)

The central design is a reverse-proxy that manages multiple backend server instances:

```
CLI/API → Proxy (port 11313) → Routes by model name → backend servers (:49152+)
                                                      ├─ llama-server (GGUF)
                                                      └─ SwiftLM (MLX, darwin/arm64)
```

Key packages in `internal/proxy/`:
- `server.go` - HTTP routing, reverse-proxy to backends, `/v1/chat/completions` pass-through, shared body/model helpers (`readOpenAIBody`/`readAnthropicBody`)
- `anthropic.go` - In-proxy translation of `/v1/messages` ⇄ `/v1/chat/completions` so backends only need an OpenAI surface. Covers tool-use (request + response + streaming), stop-sequence forwarding, image base64 (URL-source refused with 401 since backends don't fetch), error-type normalization, periodic `event: ping` frames (15s default) to survive nginx-style idle timeouts
- `manager.go` - Model lifecycle (start/stop backend, LRU eviction, max 3 models). `optionsChanged` delegates to `Runtime.SignificantOptions()` so reload-worthy key sets are a per-backend concern
- `backend.go` - `Runtime` interface: `Kind`, `HFAppName`, `BinaryPath`, `WorkingDir`, `BuildArgs`, `HealthURL`, `IsStartupError`, `SignificantOptions`
- `backend_llama.go`, `backend_swiftlm.go` - Concrete runtimes
- `backend_select.go` - Reads `metadata.yaml` backend kind → picks the runtime (fails closed on unknown kinds via `hf.parseBackendKind`)
- `registry.go` - `AllRuntimes` / `HFAppNames` (single registration point for a new backend)
- `idle.go` - Background monitor for auto-unloading idle models
- `ports.go` - Dynamic port allocation for backends

### Backends

Each backend is a `Runtime` that owns its binary, args, health probe, and error log scan. Adding a new backend means (1) implement `Runtime`, (2) add its constructor to `AllRuntimes()` in `registry.go`, (3) add a `case` in `selectRuntime` for its `metadata.yaml` kind, (4) optionally a config section under a new YAML key to mirror `llamacpp:` / `swiftlm:`. Nothing in `cmd/` or the discovery surface hardcodes a backend list — `cmd/search.go` joins every registered `HFAppName` for the `?apps=` filter.

**Security-sensitive install code is shared**: `internal/binaryrelease/` owns URL host/scheme allow-lists, download size caps, atomic symlink swap (`swiftlm-current` / `llama-current`), and a validating tar extractor (no path traversal, no escaping symlinks). `binaryrelease.SafeJoin` is also used by `internal/hf/mlx.go` to guard against malicious HuggingFace tree entries writing outside the model directory. Each backend installer (`internal/llama/binary.go`, `internal/swiftlm/binary.go`) wraps these primitives with its own repo URL, platform matrix, and `version.json` sibling file.

HuggingFace downloads enforce their own redirect allow-list (`hfAllowedHosts` in `internal/hf/client.go` — huggingface.co + the LFS / xethub CDN hosts) and a per-file size cap derived from the manifest-declared size (2× tolerance). A compromised HF response can't redirect the Authorization-bearing download off-domain, and a server lying about Content-Length can't exhaust the disk.

MLX is **macOS/arm64 only**; `swiftlm.IsSupported()` gates both install and pull. On unsupported platforms, pulling an MLX repo exits with a clear message.

### Package Structure

- `cmd/` - Cobra CLI commands (run, pull, list, serve, status, etc.)
- `internal/binaryrelease/` - Shared GitHub-release installer primitives (download, URL allow-list, tar extraction, symlink swap)
- `internal/config/` - Config loading/saving, personas (user-saved model settings)
- `internal/hf/` - Hugging Face API client, model downloads, GGUF + MLX detection, `inventory.go` metadata-driven listing
- `internal/llama/` - llama.cpp binary management
- `internal/swiftlm/` - SwiftLM binary management (MLX runtime)
- `internal/options/` - Settings resolver with layered precedence
- `internal/presets/` - Built-in curated sampling defaults per model family (embedded YAML)
- `internal/proxy/` - Multi-model proxy server + runtime registry + Anthropic translation
- `internal/server/` - Backend API client (OpenAI-compatible)
- `internal/tui/` - Bubbletea TUI (chat model, components, styles)
- `internal/ui/` - CLI utilities (spinner, progress, table, logger)

### Data Storage

All data lives in `~/.lleme/`:
- `config.yaml` - User configuration (`llamacpp:` and `swiftlm:` sections)
- `models/user/repo/metadata.yaml` - Per-repo record of each quant's backend kind; `lleme list`/`remove` walk these, not `*.gguf` globs
- `models/user/repo/<quant>.gguf` - Single-file GGUF layout
- `models/user/repo/<quant>/` - Directory layout (GGUF split shards OR full MLX tree with safetensors + tokenizer files)
- `bin/llama-current/` - Active llama.cpp symlink; `bin/version.json` tracks installed tag
- `bin/swiftlm-current/` - Active SwiftLM symlink; `bin/swiftlm-version.json` tracks installed tag
- `logs/` - Rotating log files (one per loaded model)

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

**Backend-specific preset/persona options**: both presets and personas support optional `llamacpp:` / `swiftlm:` sub-blocks that layer on top of the shared `options:` map for the matching runtime only. Merge order for a given backend kind: preset-shared → preset-backend → persona-shared → persona-backend (later wins). The client resolves kind via `hf.BackendKindForModelName` before calling `Persona.GetServerOptions(kind)` and `presets.MergeServerOptions(preset, personaOpts, kind)`.

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
