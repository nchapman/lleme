# lleme

Run local LLMs from Hugging Face with a single command.
Drop-in replacement for OpenAI **and** Anthropic APIs — works with Claude Code and any other tool that speaks either protocol.

[![Release](https://img.shields.io/github/v/release/nchapman/lleme?color=blue)](https://github.com/nchapman/lleme/releases)
[![License](https://img.shields.io/github/license/nchapman/lleme)](./LICENSE)
[![Go Reference](https://pkg.go.dev/badge/github.com/nchapman/lleme.svg)](https://pkg.go.dev/github.com/nchapman/lleme)

<video src="https://tilde.quest/~nchapman/media/videos/lleme-demo.mp4"
       autoplay loop muted playsinline
       width="720">
  <img src="https://tilde.quest/~nchapman/media/videos/lleme-demo.gif"
       alt="lleme demo" width="720">
</video>

- **Any GGUF on Hugging Face, directly.** `lleme run user/repo` — no custom registry, no republishing.
- **One port, two protocols.** OpenAI- and Anthropic-compatible endpoints on `:11313`.
- **Multi-model proxy.** Loads models on demand, unloads when idle, evicts the least-recently-used model when the limit is reached.
- **Curated sampling per family.** Sensible defaults for Qwen, DeepSeek, gpt-oss, Llama, Gemma, Kimi, Phi, and more — applied automatically by matching the HF repo name.
- **TUI built for reasoning models.** Thinking traces render separately from the response.

## Installation

**Homebrew (macOS/Linux):**
```bash
brew install nchapman/tap/lleme
```

**Go** (requires Go 1.25+):
```bash
go install github.com/nchapman/lleme@latest
```

**Build from source:**
```bash
git clone https://github.com/nchapman/lleme
cd lleme
go build -o lleme .
```

`llama.cpp` is downloaded and installed automatically on first run.

## Quickstart

Run any GGUF model from Hugging Face:

```bash
lleme run unsloth/gemma-4-E2B-it-GGUF
```

That's it. lleme picks a sensible quantization (`Q4_K_M` by default, ~3 GB plus a ~1 GB vision projector for this model), starts a proxy, and drops you into an interactive chat.

One-shot prompts and piped input work too:

```bash
lleme run unsloth/gemma-4-E2B-it-GGUF "Explain quantum computing in one sentence"

cat bug-report.md | lleme run unsloth/gemma-4-E2B-it-GGUF "summarize this"
```

Partial names resolve automatically — `lleme run gemma-4` matches `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` as long as the name is unique. (The `:quant` suffix selects a specific quantization; omit it and lleme picks the best one available locally.)

## Use with Claude Code

lleme's Anthropic-compatible endpoint makes it a drop-in backend for [Claude Code](https://docs.anthropic.com/en/docs/claude-code):

```bash
lleme pull unsloth/Qwen3.6-35B-A3B-GGUF
lleme server start -d
ANTHROPIC_BASE_URL=http://127.0.0.1:11313 \
  claude --model unsloth/Qwen3.6-35B-A3B-GGUF
```

Claude Code issues requests to lleme, which loads the model on demand.

## Features

### One port, two APIs

OpenAI and Anthropic protocols live on the same endpoint. Point any existing client at `http://localhost:11313` — no other changes needed.

**OpenAI:**
```bash
curl http://localhost:11313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/gemma-4-E2B-it-GGUF",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Anthropic:**
```bash
curl http://localhost:11313/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/gemma-4-E2B-it-GGUF",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Model-family presets

lleme ships with curated sampling defaults for every major model family. When you run a model, lleme matches the repo name against known patterns and applies the right settings automatically. No more hunting "best sampler settings for Qwen3-Coder" on Reddit.

```
$ lleme run unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
Using preset: Qwen3-Coder (matched */Qwen3-Coder-*)
...
```

Presets are overridden by config, personas, and CLI flags — see [Settings resolution](#settings-resolution) below.

### Multi-model serving

A reverse-proxy manages multiple `llama.cpp` backends. Models load on demand, unload after a configurable idle timeout, and the least-recently-used model is evicted once the max is reached (3 by default).

```bash
lleme server start          # start proxy on :11313
lleme server start -d       # ... detached
lleme status                # show loaded models and idle times
lleme unload <model>        # evict from memory immediately
```

### Terminal UI

A Bubbletea-powered chat with markdown rendering, streaming, and native handling of reasoning-model thinking tokens — thinking renders separately from the final response so you can watch the model work.

### Web UI

Visit `http://localhost:11313` in your browser once the server is running. Same API, same models, same personas — built on [assistant-ui](https://github.com/Yonom/assistant-ui).

### Personas

Save a system prompt, a model, and sampling overrides under a named persona. Use it anywhere you'd use a model name:

```bash
lleme persona create life-coach   # opens $EDITOR
```

```yaml
# ~/.lleme/personas/life-coach.yaml
model: unsloth/gemma-4-E2B-it-GGUF
system: |
  You are a pragmatic life coach. Keep responses short, specific, and
  actionable. Ask a clarifying question when it helps; otherwise give
  concrete next steps rather than general advice.
options:
  temp: 0.7
  top-p: 0.9
```

```bash
lleme run life-coach "help me plan this week"
```

### Model discovery

```bash
lleme search mistral                 # search HF for GGUF models
lleme trending                       # trending GGUF models
lleme info unsloth/gemma-4-E2B-it-GGUF   # downloads, likes, quants, sizes
lleme list                           # list downloaded models
lleme remove --older-than 30d        # prune stale models
lleme remove --larger-than 10GB      # ... or large ones
```

## FAQ

### How does this compare to Ollama or LM Studio?

- **Hugging Face native.** Any GGUF, any quant, no republishing. If it was uploaded this morning, you can `lleme run` it tonight.
- **Anthropic API alongside OpenAI.** On the same port, so Claude Code (and anything else that speaks Messages) works locally with one env var.
- **Curated sampling per family.** Built-in defaults that track upstream recommendations rather than user-written Modelfiles.
- **CLI-first.** One Go binary — no desktop app (LM Studio), no tray process, no background services you didn't start.

### Does it support vision models?

Yes — lleme pulls the `mmproj` projector file alongside the GGUF when present and passes `--mmproj` to `llama-server`. Tested on the Gemma 3, Gemma 4, and Qwen3-VL families.

### Do I need a Hugging Face account?

Only for gated models (Llama, some Google releases). Get a token at https://huggingface.co/settings/tokens and either run `hf auth login` or export `HF_TOKEN=hf_xxxxx`.

## Commands

| Category | Command | Description |
|---|---|---|
| Model | `run <model\|persona> [prompt]` | Chat with a model (auto-downloads if needed) |
| Model | `pull <model>` | Download a model from Hugging Face |
| Model | `list` / `ls` | List downloaded models |
| Model | `remove [pattern]` / `rm` | Delete models by name, pattern, age, or size |
| Model | `unload <model>` | Unload a running model |
| Model | `status` / `ps` | Show server status and loaded models |
| Personas | `persona list/show/create/edit/rm` | Manage personas |
| Server | `server start/stop/restart` | Manage the proxy server |
| Discovery | `search <query>` | Search Hugging Face for GGUF models |
| Discovery | `trending` | Show trending GGUF models |
| Discovery | `info <model>` / `show` | Show model details |
| Config | `config show/edit/path/get/set/reset` | Manage configuration |
| Other | `update` | Update lleme and llama.cpp |
| Other | `version` | Show version information |

Run `lleme <command> --help` for detailed flags.

### Removing models

The `remove` command supports patterns and filters, which can be combined:

```bash
lleme remove user/repo:quant         # specific quantization
lleme remove user/repo               # all quantizations of a model
lleme remove user/*                  # all models from a user
lleme remove *                       # everything
lleme remove --older-than 30d        # unused in 30 days (4w, 1y also work)
lleme remove --larger-than 10GB      # larger than 10GB
lleme remove user/* --older-than 7d  # combine pattern and filter
```

Use `-f` / `--force` to skip the confirmation prompt.

## Configuration

Config lives at `~/.lleme/config.yaml`. View with `lleme config show`, edit with `lleme config edit`, or set keys directly:

```bash
lleme config set server.port 11314
lleme config set huggingface.default_quant Q6_K
lleme config set huggingface.token hf_xxxxx   # or export HF_TOKEN
```

Any `llama-server` flag can be set under `llamacpp.options`:

```yaml
huggingface:
  token: ""              # or set HF_TOKEN (required for gated models)
  default_quant: Q4_K_M

server:
  host: 127.0.0.1        # bind address (0.0.0.0 for all interfaces)
  port: 11313
  max_models: 3          # concurrent models in memory
  idle_timeout: 10m      # unload after this duration (30s, 10m, 1h)

llamacpp:
  options:
    ctx-size: 8192       # context size
    gpu-layers: -1       # -1 = all layers on GPU
    flash-attn: auto
    parallel: 4          # concurrent requests per backend
```

See [`llama-server` docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) for the complete option list.

### Settings resolution

Inference options cascade from most to least specific:

**session flags → persona → preset → config → `llama-server` defaults**

A CLI flag always wins; a persona overrides a preset; presets only apply where you haven't configured something yourself. This means you can rely on built-in presets for most models and override only when needed.

## Requirements

- **macOS** (Apple Silicon or Intel) — Metal GPU acceleration via `llama.cpp`
- **Linux** (x86_64, ARM64) — CPU by default; CUDA available in `llama.cpp` builds
- **Disk**: models range from ~1 GB (small quants) to 100 GB+ (frontier). Check `lleme info <model>` before pulling.
- **RAM**: Q4 of a 7B model needs ~5 GB; Q4 of a 70B model needs 40 GB+.

## Troubleshooting

**"model requires authentication"** — the repo is gated. Get a token at https://huggingface.co/settings/tokens and either run `hf auth login` or export `HF_TOKEN=hf_xxxxx`.

**Port 11313 already in use** — change it with `lleme config set server.port <port>`, or stop whatever is bound via `lleme server stop`.

**Model keeps getting unloaded** — raise the idle timeout: `lleme config set server.idle_timeout 1h`.

**Out of memory while loading** — pick a smaller quant (`lleme info <model>` lists available quants) or lower `ctx-size` in config.

**Slow on CPU** — confirm GPU offload is active: `lleme config set llamacpp.options.gpu-layers -1`.

## Data and logs

All data lives in `~/.lleme/` (override with `LLEME_HOME`):

- `config.yaml` — user configuration
- `models/` — downloaded GGUF files (`user/repo/quant.gguf`)
- `personas/` — saved personas
- `bin/` — `llama.cpp` binaries
- `logs/` — rotating log files (10 MB, 3 generations)
  - `proxy.log` — proxy server
  - `<model>.log` — per-model backend

## Privacy

lleme sends no telemetry. Network calls happen only when you explicitly pull a model (Hugging Face), run `lleme update` (GitHub releases), or on first run when `llama.cpp` is downloaded.

## Contributing

Bug reports and PRs are welcome. For larger changes, please open an issue first to discuss direction. See [`AGENTS.md`](./AGENTS.md) for project conventions.

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the inference engine
- [Hugging Face](https://huggingface.co) — model hosting and discovery
- [Charmbracelet](https://charm.sh) — `bubbletea`, `lipgloss`, and `glamour` power the TUI
- [assistant-ui](https://github.com/Yonom/assistant-ui) — the web chat interface
- [Unsloth](https://unsloth.ai) — high-quality GGUF quantizations referenced in the examples

## License

MIT — see [LICENSE](./LICENSE).
