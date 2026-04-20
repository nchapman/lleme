package normalize

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"
)

func TestBufferedNormalizesNonStreamResponse(t *testing.T) {
	// Minimal SwiftLM-shaped non-streaming response: model is the
	// backend's internal id, no system_fingerprint, usage missing
	// details. All three should be patched.
	in := `{"id":"chatcmpl-1","object":"chat.completion","model":"backend-id","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`
	r := newBufferedReader(strings.NewReader(in), Options{
		RequestedModel: "user/repo",
		Fingerprint:    "lleme-test",
	})
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal: %v\nbody=%s", err, out)
	}
	if got["model"] != "user/repo" {
		t.Errorf("model = %v, want user/repo", got["model"])
	}
	if got["system_fingerprint"] != "lleme-test" {
		t.Errorf("system_fingerprint = %v, want lleme-test", got["system_fingerprint"])
	}
	usage := got["usage"].(map[string]any)
	if _, ok := usage["prompt_tokens_details"]; !ok {
		t.Errorf("prompt_tokens_details missing")
	}
	if _, ok := usage["completion_tokens_details"]; !ok {
		t.Errorf("completion_tokens_details missing")
	}
}

// TestWrapDispatch verifies the public Wrap entrypoint: streaming
// option picks the SSE-aware reader, non-streaming picks the buffered
// JSON reader, and the empty Fingerprint default is applied. The
// internal constructors are exercised everywhere else; this test
// pins down the public API surface.
func TestWrapDispatch(t *testing.T) {
	t.Run("streaming dispatches to stream reader", func(t *testing.T) {
		// A complete SSE frame survives the stream reader and gets its
		// model rewritten. The buffered reader would have JSON-parsed
		// the entire body (including the SSE framing) and produced
		// nonsense — observable difference confirms dispatch.
		in := "data: {\"object\":\"chat.completion.chunk\",\"model\":\"backend\"}\n\n"
		got, err := io.ReadAll(Wrap(strings.NewReader(in), Options{
			Streaming:      true,
			RequestedModel: "user/repo",
			Fingerprint:    "lleme-test",
		}))
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !strings.Contains(string(got), `"model":"user/repo"`) {
			t.Errorf("streaming wrap did not rewrite model: %s", got)
		}
		if !strings.HasSuffix(string(got), "\n\n") {
			t.Errorf("streaming wrap dropped SSE frame terminator: %q", got)
		}
	})

	t.Run("non-streaming dispatches to buffered reader", func(t *testing.T) {
		// Buffered reader will parse JSON and rewrite model. The
		// streaming reader would not (no data: line).
		in := `{"model":"backend"}`
		out, err := io.ReadAll(Wrap(strings.NewReader(in), Options{RequestedModel: "user/repo"}))
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !strings.Contains(string(out), `"model":"user/repo"`) {
			t.Errorf("buffered wrap did not rewrite model: %s", out)
		}
	})

	t.Run("empty fingerprint defaults to lleme-dev", func(t *testing.T) {
		in := `{}`
		out, err := io.ReadAll(Wrap(strings.NewReader(in), Options{}))
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if !strings.Contains(string(out), `"system_fingerprint":"lleme-dev"`) {
			t.Errorf("default fingerprint not applied: %s", out)
		}
	})
}

// TestBufferedRespectsResponseSizeCap — verifies the io.LimitReader
// guard against a backend returning an oversized non-streaming
// response. Without the cap, io.ReadAll would buffer indefinitely.
// We don't try to assert a specific byte count post-cap; just that
// the loader returns without consuming the entire infinite source.
func TestBufferedRespectsResponseSizeCap(t *testing.T) {
	// infiniteReader yields 'x' forever — io.ReadAll without a cap
	// would never return.
	r := newBufferedReader(infiniteReader{}, Options{})
	done := make(chan struct{})
	go func() {
		_, _ = io.ReadAll(r)
		close(done)
	}()
	select {
	case <-done:
		// load() returned: cap worked.
	case <-time.After(2 * time.Second):
		t.Fatal("buffered loader did not honor size cap (infinite read)")
	}
}

type infiniteReader struct{}

func (infiniteReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 'x'
	}
	return len(p), nil
}

func TestBufferedPassesMalformedJSON(t *testing.T) {
	// Defensive posture: never replace the upstream body with a parse
	// error. Better to ship the original bytes than a mangled response.
	in := "not json at all"
	r := newBufferedReader(strings.NewReader(in), Options{RequestedModel: "x", Fingerprint: "y"})
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(out) != in {
		t.Errorf("malformed body changed: got %q, want %q", out, in)
	}
}
