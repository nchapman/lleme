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

**lleme** is a Go CLI for running local LLMs via llama.cpp (GGUF), with Hugging Face model management. Built on Charmbracelet (bubbletea, lipgloss, glamour) for TUI and Cobra for CLI.

### Multi-Model Proxy (Core Architecture)

The central design is a reverse-proxy that manages multiple backend server instances:

```
CLI/API → Proxy (port 11313) → Routes by model name → backend servers (:49152+)
                                                      └─ llama-server (GGUF)
```

Key packages in `internal/proxy/`:
- `server.go` - HTTP routing, reverse-proxy to backends, `/v1/chat/completions` pass-through, shared body/model helpers (`readOpenAIBody`/`readAnthropicBody`)
- `anthropic.go` - In-proxy translation of `/v1/messages` ⇄ `/v1/chat/completions` so backends only need an OpenAI surface. Covers tool-use (request + response + streaming), stop-sequence forwarding, image base64 (URL-source refused with 401 since backends don't fetch), error-type normalization, periodic `event: ping` frames (15s default) to survive nginx-style idle timeouts
- `manager.go` - Model lifecycle (start/stop backend, LRU eviction, max 3 models). `optionsChanged` delegates to `Runtime.SignificantOptions()` so reload-worthy key sets are a per-backend concern
- `backend.go` - `Runtime` interface: `Kind`, `HFAppName`, `BinaryPath`, `WorkingDir`, `BuildArgs`, `HealthURL`, `IsStartupError`, `SignificantOptions`
- `backend_llama.go` - The llama-server runtime
- `backend_select.go` - Reads `metadata.yaml` backend kind → picks the runtime (fails closed on unknown kinds via `hf.parseBackendKind`)
- `registry.go` - `AllRuntimes` / `HFAppNames` (single registration point for a new backend)
- `idle.go` - Background monitor for auto-unloading idle models
- `ports.go` - Dynamic port allocation for backends

### Backends

lleme is llama.cpp-only today; the SwiftLM/MLX backend was removed. A single `Runtime` (`backend_llama.go`) owns the binary, args, health probe, and error log scan. Adding a new backend means (1) implement `Runtime`, (2) add its constructor to `AllRuntimes()` in `registry.go`, (3) add a `case` in `selectRuntime` for its `metadata.yaml` kind, (4) optionally a config section under a new YAML key to mirror `llamacpp:`. Nothing in `cmd/` or the discovery surface hardcodes a backend list — `cmd/search.go` joins every registered `HFAppName` for the `?apps=` filter. `metadata.yaml` entries with `backend: mlx` (written by SwiftLM-era versions) still list and remove cleanly; loading one fails closed with a clear error in `selectRuntime`.

**Security-sensitive install code is shared**: `internal/binaryrelease/` owns URL host/scheme allow-lists, download size caps, atomic symlink swap (`llama-current`), and a validating tar extractor (no path traversal, no escaping symlinks). The llama.cpp installer (`internal/llama/binary.go`) wraps these primitives with its repo URL, platform matrix, and `version.json` sibling file.

HuggingFace downloads enforce their own redirect allow-list (`hfAllowedDomains` in `internal/hf/client.go` — any subdomain of the HF-owned apex domains `hf.co`/`huggingface.co`, covering the LFS and regional xet CDN hosts) and a per-file size cap derived from the manifest-declared size (2× tolerance). A compromised HF response can't redirect the Authorization-bearing download off-domain, and a server lying about Content-Length can't exhaust the disk.

### Package Structure

- `cmd/` - Cobra CLI commands (run, pull, list, serve, status, etc.)
- `internal/binaryrelease/` - Shared GitHub-release installer primitives (download, URL allow-list, tar extraction, symlink swap)
- `internal/config/` - Config loading/saving, personas (user-saved model settings)
- `internal/hf/` - Hugging Face API client, model downloads, GGUF detection, `inventory.go` metadata-driven listing
- `internal/llama/` - llama.cpp binary management
- `internal/options/` - Settings resolver with layered precedence
- `internal/presets/` - Built-in curated sampling defaults per model family (embedded YAML)
- `internal/proxy/` - Multi-model proxy server + runtime registry + Anthropic translation
- `internal/server/` - Backend API client (OpenAI-compatible)
- `internal/tui/` - Bubbletea TUI (chat model, components, styles)
- `internal/ui/` - CLI utilities (spinner, progress, table, logger)

### Data Storage

All data lives in `~/.lleme/`:
- `config.yaml` - User configuration (`llamacpp:` section)
- `models/user/repo/metadata.yaml` - Per-repo record of each quant's backend kind; `lleme list`/`remove` walk these, not `*.gguf` globs
- `models/user/repo/<quant>.gguf` - Single-file GGUF layout
- `models/user/repo/<quant>/` - Directory layout (GGUF split shards; or a legacy MLX tree from SwiftLM-era pulls)
- `bin/llama-current/` - Active llama.cpp symlink; `bin/version.json` tracks installed tag
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

When adding or editing a preset, read `internal/presets/data/README.md` first — it documents pattern conventions, ordering rules, what belongs under `options` (sampling params only, no runtime/hardware knobs), and when to split a family into multiple files. The `presets_test.go` table should include a realistic HF repo name per new preset, plus an ordering test if the preset could be masked by a more general one. Preset options merge under persona options via `presets.MergeServerOptions(preset, personaOpts)`; persona options come from `Persona.GetServerOptions()`.

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
