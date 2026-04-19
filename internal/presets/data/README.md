# Model Presets

Each `.yaml` file in this directory defines curated inference defaults for a model family. Presets are matched by glob patterns against the HuggingFace `user/repo` name and slot into the settings resolver below persona and global config:

**session > persona > preset > config > llama-server default**

## Format

```yaml
name: Human-readable name shown in logs
source: https://link-to-official-docs-or-paper
match:
  - "*/ModelFamily-*"   # path.Match globs against user/repo
options:
  temp: 0.7
  top-p: 0.8
  top-k: 20
  min-p: 0.0
```

Keys under `options` mirror `llama-server` CLI flags in kebab-case (`--temp` → `temp`, `--top-k` → `top-k`, `--presence-penalty` → `presence-penalty`, etc.). Unknown keys are passed through to the server verbatim.

## Match patterns

- Patterns use Go's [`path.Match`](https://pkg.go.dev/path#Match) syntax (`*` matches any sequence excluding `/`, `?` matches one char, `[abc]` matches a character class).
- Patterns are matched against the `user/repo` portion only — the `:quant` suffix is stripped first.
- Matching is **case-insensitive** — both the pattern and repo name are lowercased before comparison. `*/Qwen3-*` and `*/qwen3-*` are equivalent.
- Files are evaluated in **alphabetical order**; the first matching pattern in the first matching file wins.
- A preset matches if **any** of its patterns matches.

### Pattern conventions

- **Prefer `*/<Model>-*` over explicit namespace prefixes.** One broad pattern covers `unsloth/`, `bartowski/`, `ggml-org/`, the upstream repo, and third-party forks in a single line. Only use an explicit namespace when you need to exclude look-alikes (rare).
- **Don't list redundant patterns.** If `*/Foo-*` is in the list, a more specific `*/Foo-Bar-*` is already covered and should be omitted.
- **Design patterns to avoid cross-matching between variants.** Use the structure of the repo name to encode the version boundary:
  - `*/Qwen3-*` (dash) deliberately won't match `Qwen3.5-*` (dot).
  - `*/Gemma-3-*` (trailing dash) won't match `Gemma-3n-*`.
  - Cross-contamination is bad — a Qwen3.5 repo matching the Qwen3 preset means the user gets the wrong settings.

### Ordering for specificity

When two presets could both match the same repo (e.g. `DeepSeek-R1-0528` could match both `*/DeepSeek-R1-0528-*` and `*/DeepSeek-R1-*`), the **more specific file must sort alphabetically first** so it wins first-match. Two facts that make this usually-automatic:

- `-` (0x2D) sorts before `.` (0x2E), which sorts before letters. So `qwen3-coder.yaml` naturally sorts before `qwen3.yaml`, and `deepseek-r1-0528.yaml` before `deepseek-r1.yaml`.
- A date-style suffix like `0528` sorts before any letter-prefixed filename for the same base name.

If your new preset could be masked by an existing more-general one, check the alphabetical ordering manually before committing.

## What belongs in `options`

**Include**: sampling and decoding parameters the model's official docs explicitly call out as non-default recommendations — `temp`, `top-p`, `top-k`, `min-p`, `presence-penalty`, `repeat-penalty`, etc.

**Omit**:
- Values that match the llama.cpp default (e.g. don't set `repeat-penalty: 1.0` unless the docs specifically flag it as a recommendation worth overriding a future default).
- Runtime/hardware knobs: `ctx-size`, `gpu-layers`, `flash-attn`, `cache-type-k`, `threads`. These depend on the user's machine and use case — leave them to `~/.lleme/config.yaml` or persona.
- Operational flags (`--jinja`, `--seed`, `--prio`). Not sampling behavior.

**Explicit zeros are valid.** The resolver uses key-existence semantics, so `min-p: 0.0` is distinct from "not set" and will disable min-p sampling at the server. Include explicit zeros when the upstream docs recommend them.

## When to split a family into multiple files

Split when the same model family ships variants with different recommended sampling:

- **Mode-specific variants**: Qwen3-Next-*-Instruct vs Qwen3-Next-*-Thinking; Ministral-3-*-Instruct vs Ministral-3-*-Reasoning.
- **Version bumps with different settings**: GLM-5 vs GLM-5.1, Qwen3 vs Qwen3.5 vs Qwen3.6.

Keep a single file when all variants in a family share the same recommendations (e.g. `nemotron-3.yaml` covers both the base and Super variants).

## Contributing

1. Create a new `.yaml` file named `<family>[-<variant>].yaml`. Lowercase, hyphen-separated.
2. `source` should link to the official documentation for the recommended settings (prefer the `unsloth.ai/docs/models/...md` URL).
3. Confirm your match patterns don't overlap with an existing preset (unless the alphabetical ordering deliberately routes the overlap).
4. Add a test case to `presets_test.go` using a realistic HF repo name. If your preset could be masked by a more general one, include a test that proves it wins.
5. Open a PR — one file per PR keeps reviews focused.

## Skipping a model

If the upstream docs don't recommend any non-default sampling settings (e.g. a blog post focused on fine-tuning only), skip creating a preset. An empty-options preset provides no value and just adds maintenance surface.
