package normalize

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"
)

// runStreamTimed feeds in an SSE stream split into `pre` (the
// content frames) and `final` (the chunk carrying usage), with a
// controlled gap between them so we can assert a specific
// predicted_per_second falls in a known range.
func runStreamTimed(t *testing.T, pre, final string, gap time.Duration) string {
	t.Helper()
	r := newStreamReader(&pausedReader{
		parts: []string{pre, final},
		gaps:  []time.Duration{gap},
	}, Options{})
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return string(out)
}

func TestStreamSynthesizesTimingsFromWallClock(t *testing.T) {
	// 4 completion tokens generated over ~100ms of wall-clock should
	// land at roughly 40 tok/s. We assert a wide band (10-200 tok/s)
	// to keep the test stable across CI noise — the point is to
	// prove a sensible value gets emitted, not to pin a number.
	pre := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
	final := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\n\n"
	got := runStreamTimed(t, pre, final, 100*time.Millisecond)

	// Find the line containing usage; its timings.predicted_per_second
	// should be set.
	frames := splitFrames(got)
	var usageChunk map[string]any
	for _, f := range frames {
		if !strings.Contains(f, `"usage"`) {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(strings.TrimRight(f, "\r\n"), "data:"))
		if err := json.Unmarshal([]byte(payload), &usageChunk); err != nil {
			t.Fatalf("parse usage chunk: %v\nraw=%s", err, payload)
		}
	}
	if usageChunk == nil {
		t.Fatalf("no usage chunk in output:\n%s", got)
	}
	timings, ok := usageChunk["timings"].(map[string]any)
	if !ok {
		t.Fatalf("timings missing or wrong shape: %v", usageChunk["timings"])
	}
	rate, ok := timings["predicted_per_second"].(float64)
	if !ok {
		t.Fatalf("predicted_per_second missing or wrong type: %v", timings["predicted_per_second"])
	}
	if rate < 10 || rate > 200 {
		t.Errorf("predicted_per_second = %v, want 10..200 for 4 tokens / ~100ms", rate)
	}
	if pn, _ := timings["predicted_n"].(float64); pn != 4 {
		t.Errorf("predicted_n = %v, want 4", pn)
	}
}

func TestStreamTimingsOverwritesBackendValues(t *testing.T) {
	// llama.cpp ships its own predicted_per_second in the final
	// chunk. We deliberately overwrite it so cross-backend
	// comparisons are apples-to-apples (see timings.go contract).
	// Backend prompt-side fields must survive untouched.
	pre := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
	final := `data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":42,"prompt_ms":500,"prompt_per_second":84,"cache_n":7,"predicted_n":4,"predicted_ms":1,"predicted_per_second":99999},"usage":{"completion_tokens":4,"prompt_tokens":42,"total_tokens":46}}` + "\n\n"
	got := runStreamTimed(t, pre, final, 50*time.Millisecond)

	chunk := findUsageChunk(t, got)
	timings := chunk["timings"].(map[string]any)

	// Predicted_per_second must NOT be the backend's 99999 — proves
	// we overwrote with our own measurement.
	if rate, _ := timings["predicted_per_second"].(float64); rate >= 99999 {
		t.Errorf("predicted_per_second = %v, expected proxy-measured value (overwrite contract)", rate)
	}
	// Prompt-side fields must survive — we can't measure those from
	// outside the inference engine, so backend values are ground truth.
	if pn, _ := timings["prompt_n"].(float64); pn != 42 {
		t.Errorf("prompt_n = %v, want 42 (backend-supplied, must not be touched)", pn)
	}
	if cn, _ := timings["cache_n"].(float64); cn != 7 {
		t.Errorf("cache_n = %v, want 7 (backend-supplied, must not be touched)", cn)
	}
}

func TestStreamTimingsSkipsWhenNoFirstContent(t *testing.T) {
	// A stream that goes straight to a final chunk with usage and no
	// preceding content frame can't have a meaningful tok/s — we
	// don't have a first-token timestamp. Must leave timings absent
	// rather than divide by ~0.
	final := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2}}\n\n"
	r := newStreamReader(strings.NewReader(final), Options{})
	out, _ := io.ReadAll(r)
	if strings.Contains(string(out), `"timings"`) {
		t.Errorf("timings should be absent when no content frame preceded usage:\n%s", out)
	}
}

func TestStreamTimingsSkipsRoleOnlyPreamble(t *testing.T) {
	// llama.cpp's first delta is often role-only ({"role":"assistant"}
	// with empty content). That isn't a real decoded token; using its
	// timestamp would inflate the elapsed window and underreport
	// tok/s. The clock must anchor on the first actual content delta.
	preamble := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
	content := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
	final := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4}}\n\n"

	// Long pause between preamble and content; short pause from
	// content to final. If the clock anchored to the preamble, the
	// reported tok/s would be dominated by the long pause.
	r := newStreamReader(&pausedReader{
		parts:  []string{preamble, content, final},
		gaps:   []time.Duration{300 * time.Millisecond, 50 * time.Millisecond},
		cursor: 0,
	}, Options{})
	out, _ := io.ReadAll(r)

	chunk := findUsageChunk(t, string(out))
	timings := chunk["timings"].(map[string]any)
	rate := timings["predicted_per_second"].(float64)
	// 4 tokens / 0.05s = 80 tok/s if clock is correct.
	// 4 tokens / 0.35s = 11 tok/s if clock anchored to preamble.
	if rate < 30 {
		t.Errorf("predicted_per_second = %.1f, looks like the clock anchored to the role-only preamble", rate)
	}
}

// --- test helpers ---

func splitFrames(s string) []string {
	return strings.Split(s, "\n\n")
}

func findUsageChunk(t *testing.T, stream string) map[string]any {
	t.Helper()
	for _, f := range splitFrames(stream) {
		if !strings.Contains(f, `"usage"`) {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(strings.TrimRight(f, "\r\n"), "data:"))
		var chunk map[string]any
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("parse usage chunk: %v\nraw=%s", err, payload)
		}
		return chunk
	}
	t.Fatalf("no usage chunk found in stream:\n%s", stream)
	return nil
}

// pausedReader serves byte chunks separated by configurable sleeps.
// Used to control the wall-clock interval between specific frames.
type pausedReader struct {
	parts  []string
	gaps   []time.Duration // gaps[i] = sleep BEFORE parts[i+1]
	cursor int
}

func (p *pausedReader) Read(b []byte) (int, error) {
	if p.cursor >= len(p.parts) {
		return 0, io.EOF
	}
	if p.cursor > 0 {
		time.Sleep(p.gaps[p.cursor-1])
	}
	n := copy(b, p.parts[p.cursor])
	p.cursor++
	return n, nil
}
