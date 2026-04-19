# Model Presets

Each `.yaml` file in this directory defines curated inference defaults for a model family. Presets are matched by glob patterns against the HuggingFace `user/repo` name and slot into the settings resolver below persona and global config:

**session > persona > preset > config > llama-server default**

## Format

```yaml
name: Human-readable name shown in logs
source: https://link-to-official-docs-or-paper
match:
  - "*/ModelFamily*-Instruct*"   # path.Match patterns against user/repo
  - "specific-user/ModelRepo*"
options:
  temp: 0.7
  top-p: 0.8
  top-k: 20
  presence-penalty: 1.5
  # Any llama-server flag is valid here (sampling params or server-startup options).
  # Keys mirror llama-server CLI flags: --temp, --top-k, --presence-penalty, etc.
  # Server-startup keys (ctx-size, gpu-layers, cache-type-k, chat-template-kwargs, …)
  # are passed to llama-server at model load time.
```

## Match patterns

- Patterns use Go's `path.Match` syntax (`*` matches any sequence excluding `/`, `?` matches one char, `[abc]` matches character class).
- Patterns are matched against the `user/repo` portion only — the `:quant` suffix is stripped before matching.
- Files are evaluated in **alphabetical order**; the first matching preset wins. Name more specific files with an earlier sort key (e.g. `qwen3-thinking.yaml` before `qwen3.yaml`).
- A preset matches if **any** of its patterns matches.

## Contributing

1. Create a new `.yaml` file named `<family>[-<variant>].yaml`.
2. Verify the `source` URL links to official documentation for the recommended settings.
3. Add a test case to `presets_test.go` asserting your new preset is returned for a representative model name.
4. Open a PR — one file per PR keeps reviews focused.

## Notes

- Setting a value to `0` has no effect because zero is treated as "not set" by the resolver. If a model requires disabling a parameter (e.g. `min-p: 0.0`), users should override via persona or `~/.lleme/config.yaml` for now.
- `options` keys must be valid `llama-server` flags. Unknown keys are passed through to the server; llama-server will warn if it doesn't recognise them.
