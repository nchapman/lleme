package normalize

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func runStream(t *testing.T, in string, opts Options) string {
	t.Helper()
	r := newStreamReader(strings.NewReader(in), opts)
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return string(out)
}

func TestStreamDropsPrefillProgress(t *testing.T) {
	in := strings.Join([]string{
		`data: {"object":"prefill_progress","fraction":0.25}`, "",
		`data: {"object":"prefill_progress","fraction":0.75}`, "",
		`data: {"object":"chat.completion.chunk","id":"abc","model":"backend","choices":[{"delta":{"content":"hi"}}]}`, "",
		`data: [DONE]`, "",
		"",
	}, "\n")
	got := runStream(t, in, Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"})
	if strings.Contains(got, "prefill_progress") {
		t.Errorf("prefill_progress leaked into output:\n%s", got)
	}
	if !strings.Contains(got, `"chat.completion.chunk"`) {
		t.Errorf("real chunk dropped:\n%s", got)
	}
	if !strings.Contains(got, "[DONE]") {
		t.Errorf("[DONE] terminator dropped:\n%s", got)
	}
}

func TestStreamRewritesModel(t *testing.T) {
	in := "data: {\"object\":\"chat.completion.chunk\",\"model\":\"backend-id\"}\n\n"
	got := runStream(t, in, Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"})
	if !strings.Contains(got, `"model":"user/repo"`) {
		t.Errorf("model not rewritten:\n%s", got)
	}
	if strings.Contains(got, "backend-id") {
		t.Errorf("backend-id leaked:\n%s", got)
	}
}

func TestStreamPreservesDONE(t *testing.T) {
	in := "data: [DONE]\n\n"
	got := runStream(t, in, Options{RequestedModel: "x", Fingerprint: "y"})
	if got != in {
		t.Errorf("got %q, want %q", got, in)
	}
}

func TestStreamPreservesUnknownLines(t *testing.T) {
	in := "event: ping\n: heartbeat comment\nid: 17\ndata: {\"object\":\"chat.completion.chunk\",\"model\":\"x\"}\n\n"
	got := runStream(t, in, Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"})
	for _, want := range []string{"event: ping", ": heartbeat comment", "id: 17"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing preserved line %q in output:\n%s", want, got)
		}
	}
}

func TestStreamPreservesCRLF(t *testing.T) {
	in := "data: {\"object\":\"chat.completion.chunk\",\"model\":\"x\"}\r\n\r\n"
	got := runStream(t, in, Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"})
	if !strings.HasSuffix(got, "\r\n\r\n") {
		t.Errorf("CRLF framing not preserved:\n%q", got)
	}
}

// TestStreamDropsPartialFrameOnEOF: per WHATWG SSE §9.2.6,
// in-progress events on EOF are discarded. If we re-emitted a
// half-frame the downstream client could concatenate it with bytes
// from a reconnect and parse a corrupted JSON payload. The complete
// frames before the partial one must still make it through.
func TestStreamDropsPartialFrameOnEOF(t *testing.T) {
	in := strings.Join([]string{
		`data: {"object":"chat.completion.chunk","model":"backend","choices":[{"delta":{"content":"first"}}]}`, "",
		`data: {"object":"chat.completion.chunk","model":"backend","choices":[{"delta":{"content":"partial"}}]}`,
	}, "\n") // note: no trailing blank line on the second frame
	got := runStream(t, in, Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"})
	if !strings.Contains(got, `"first"`) {
		t.Errorf("complete frame missing:\n%q", got)
	}
	if strings.Contains(got, `"partial"`) {
		t.Errorf("partial frame should have been dropped:\n%q", got)
	}
}

func TestStreamPassesMalformedJSON(t *testing.T) {
	in := "data: {not valid json\n\ndata: {\"object\":\"chat.completion.chunk\",\"model\":\"x\"}\n\n"
	got := runStream(t, in, Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"})
	if !strings.Contains(got, "{not valid json") {
		t.Errorf("malformed frame should pass through unchanged; got:\n%s", got)
	}
	if !strings.Contains(got, `"model":"user/repo"`) {
		t.Errorf("subsequent valid frame should still normalize:\n%s", got)
	}
}

func TestStreamSynthesizesFingerprintAndUsageDetails(t *testing.T) {
	// SwiftLM-shaped final chunk: no system_fingerprint, usage missing details.
	in := "data: {\"object\":\"chat.completion.chunk\",\"model\":\"backend\",\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n"
	got := runStream(t, in, Options{RequestedModel: "user/repo", Fingerprint: "lleme-vX"})
	if !strings.Contains(got, `"system_fingerprint":"lleme-vX"`) {
		t.Errorf("system_fingerprint not synthesized:\n%s", got)
	}
	if !strings.Contains(got, `"prompt_tokens_details":{"cached_tokens":0}`) {
		t.Errorf("prompt_tokens_details not synthesized:\n%s", got)
	}
	if !strings.Contains(got, `"completion_tokens_details":{"reasoning_tokens":0}`) {
		t.Errorf("completion_tokens_details not synthesized:\n%s", got)
	}
}

// TestStreamIdempotent: feed normalized output back through the
// pipeline and confirm the second pass is a no-op semantically (parsed
// chunks identical). String comparison is fragile because Go's
// json.Marshal sorts map keys but the stream's framing/whitespace is
// stable across passes — we compare parsed JSON to be safe.
func TestStreamIdempotent(t *testing.T) {
	in := strings.Join([]string{
		`data: {"object":"prefill_progress","fraction":0.5}`, "",
		`data: {"object":"chat.completion.chunk","model":"backend","choices":[{"delta":{"content":"hi"}}]}`, "",
		`data: {"object":"chat.completion.chunk","model":"backend","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`, "",
		`data: [DONE]`, "",
		"",
	}, "\n")
	opts := Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"}
	first := runStream(t, in, opts)
	second := runStream(t, first, opts)
	if !semanticallyEqualSSE(t, first, second) {
		t.Errorf("not idempotent:\nfirst:\n%s\nsecond:\n%s", first, second)
	}
}

// TestStreamRejectsRunawayFrame — a frame whose lines individually
// fit under the per-line cap but whose total exceeds the per-frame
// cap must still be rejected. Without the per-frame total bound, a
// backend could pack many near-max lines into one event and OOM us
// before any individual line tripped maxStreamLineBytes.
func TestStreamRejectsRunawayFrame(t *testing.T) {
	// Each line ~6 MiB (under 8 MiB per-line cap). Three of them
	// sum to ~18 MiB (over 16 MiB per-frame cap). No blank line —
	// we want to trip the frame cap before EOF or terminator.
	chunk := strings.Repeat("x", 6*1024*1024)
	in := "data: " + chunk + "\n" +
		"data: " + chunk + "\n" +
		"data: " + chunk + "\n"
	r := newStreamReader(strings.NewReader(in), Options{})
	_, err := io.ReadAll(r)
	if err == nil {
		t.Fatal("expected error on oversized frame, got nil")
	}
	if !strings.Contains(err.Error(), "frame exceeded") {
		t.Errorf("error should mention frame size cap; got: %v", err)
	}
}

// TestStreamRejectsRunawayLine — a single SSE line with no
// terminator forever (or one larger than the cap) must be rejected
// rather than buffered to OOM. We use a small synthetic cap test by
// constructing a line that exceeds the per-line cap; the reader
// should return an error rather than allocate without bound.
func TestStreamRejectsRunawayLine(t *testing.T) {
	// 9 MiB of 'x' followed by no newline — exceeds the 8 MiB cap.
	huge := strings.Repeat("x", 9*1024*1024)
	r := newStreamReader(strings.NewReader(huge), Options{})
	_, err := io.ReadAll(r)
	if err == nil {
		t.Fatal("expected error on oversized line, got nil")
	}
	if !strings.Contains(err.Error(), "exceeded") {
		t.Errorf("error should mention size cap; got: %v", err)
	}
}

// semanticallyEqualSSE compares two SSE streams by extracting each
// data: payload, parsing JSON, and re-serializing — so map-key
// ordering differences don't trigger false negatives.
func semanticallyEqualSSE(t *testing.T, a, b string) bool {
	t.Helper()
	return canonicalizeSSE(t, a) == canonicalizeSSE(t, b)
}

func canonicalizeSSE(t *testing.T, s string) string {
	t.Helper()
	var out strings.Builder
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimRight(line, "\r")
		if strings.HasPrefix(trimmed, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			if payload == "" || payload == "[DONE]" {
				out.WriteString(trimmed + "\n")
				continue
			}
			var v any
			if err := json.Unmarshal([]byte(payload), &v); err != nil {
				out.WriteString(trimmed + "\n")
				continue
			}
			b, _ := json.Marshal(v)
			out.WriteString("data: " + string(b) + "\n")
			continue
		}
		out.WriteString(trimmed + "\n")
	}
	return out.String()
}
