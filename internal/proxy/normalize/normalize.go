// Package normalize patches subtle divergences in backend OpenAI surfaces
// (llama-server, SwiftLM) into a single canonical shape before responses
// reach OpenAI clients or the in-proxy Anthropic translator.
//
// Design rules — all four normalizers obey these:
//
//   - Idempotent: running a second pass on already-normalized output is a
//     no-op. Tests assert this directly so we don't drift.
//   - Self-disabling: when a backend stops emitting the bad pattern (e.g.
//     SwiftLM lands its prefill_progress filtering, or starts populating
//     system_fingerprint), the normalizer sees the new shape, finds no
//     work to do, and passes the bytes through. We never have to remove
//     a normalizer just because upstream caught up.
//   - Additive only: never overwrite a field the backend provided. The
//     single exception is `model`, which we always rewrite to the name
//     the client requested — that's the contract callers depend on.
//
// Frames the normalizer can't parse (malformed JSON, unrecognized SSE
// shapes) pass through verbatim. The proxy mirrors the Anthropic
// translator's posture: never abort a stream over one bad chunk.
package normalize

import (
	"io"
)

// Options configures a Wrap. RequestedModel is the model name from the
// client's request — we rewrite responses to echo it back, since SwiftLM
// substitutes its own internal modelId. Streaming switches between the
// SSE-aware reader and the buffered JSON reader.
type Options struct {
	RequestedModel string
	Streaming      bool
	// Fingerprint is the value injected into responses when the backend
	// omits system_fingerprint. Defaults to "lleme-dev" if empty.
	Fingerprint string
}

// Wrap returns an io.Reader that yields normalized bytes. The wrapper
// does not own the upstream's lifecycle: the caller is responsible for
// closing the original response body. Returning io.Reader (not
// io.ReadCloser) keeps that ownership boundary explicit and lets
// callers keep their existing `defer resp.Body.Close()` patterns
// without risking double-close.
func Wrap(r io.Reader, opts Options) io.Reader {
	if opts.Fingerprint == "" {
		opts.Fingerprint = "lleme-dev"
	}
	if opts.Streaming {
		return newStreamReader(r, opts)
	}
	return newBufferedReader(r, opts)
}
