package normalize

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// ErrResponseTooLarge is returned when a backend response body exceeds
// maxBufferedResponseBytes. Surfaced rather than silently truncated so
// the caller can return a clear error instead of a corrupt payload.
var ErrResponseTooLarge = errors.New("normalize: backend response exceeded size cap")

// maxBufferedResponseBytes caps the size of a non-streaming backend
// response body the proxy will buffer in memory. A misbehaving or
// adversarial backend that returns a multi-GB body would otherwise
// OOM the proxy. 64 MiB comfortably exceeds any realistic single
// chat-completion JSON (long conversations stream; non-stream
// responses with reasoning + tool calls top out around 1-2 MiB in
// practice).
const maxBufferedResponseBytes = 64 * 1024 * 1024

// bufferedReader handles non-streaming /v1/chat/completions responses.
// Reads the full body once on first Read (capped via io.LimitReader),
// runs response normalizers on the parsed top-level object,
// re-marshals, then serves bytes from an internal buffer.
type bufferedReader struct {
	upstream io.Reader
	opts     Options
	buf      *bytes.Reader
	loaded   bool
}

func newBufferedReader(r io.Reader, opts Options) *bufferedReader {
	return &bufferedReader{upstream: r, opts: opts}
}

func (b *bufferedReader) Read(p []byte) (int, error) {
	if !b.loaded {
		if err := b.load(); err != nil {
			return 0, err
		}
	}
	return b.buf.Read(p)
}

func (b *bufferedReader) load() error {
	b.loaded = true
	// Read one byte past the cap so we can detect overflow. Plain
	// io.LimitReader silently truncates, which would yield a partial
	// (likely invalid) JSON body served with a 200 status.
	body, err := io.ReadAll(io.LimitReader(b.upstream, maxBufferedResponseBytes+1))
	if err != nil {
		return err
	}
	if len(body) > maxBufferedResponseBytes {
		return fmt.Errorf("%w (limit: %d bytes)", ErrResponseTooLarge, maxBufferedResponseBytes)
	}
	b.buf = bytes.NewReader(normalizeResponseBody(body, b.opts))
	return nil
}

// normalizeResponseBody parses, transforms, and re-marshals a JSON
// response body. Returns the original bytes unchanged if parsing or
// re-marshaling fails — better to ship the upstream's body verbatim
// than to return a mangled response.
func normalizeResponseBody(body []byte, opts Options) []byte {
	var obj chunkObject
	if err := json.Unmarshal(body, &obj); err != nil {
		return body
	}
	applyResponseNormalizers(obj, opts)
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}
