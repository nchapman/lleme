package normalize

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// streamReader is an SSE-aware io.ReadCloser. It pulls one frame
// (lines until a blank terminator) from upstream at a time, runs
// chunk normalizers against the JSON payload of any data: line, and
// emits the rewritten frame.
//
// Pattern mirrors translateAnthropicStream's bufio.Reader.ReadBytes
// loop in internal/proxy/anthropic.go: a Scanner's 64 KiB token cap
// would silently truncate long reasoning or tool-call payloads, and
// raising it isn't enough — single chunks from real models can be
// surprisingly large.
type streamReader struct {
	reader *bufio.Reader
	opts   Options
	out    bytes.Buffer
	done   bool
	err    error
}

func newStreamReader(r io.Reader, opts Options) *streamReader {
	return &streamReader{
		reader: bufio.NewReader(r),
		opts:   opts,
	}
}

// Read pulls and processes frames lazily. Each call may produce zero
// frames (a dropped prefill_progress yields only the upstream advance,
// no output bytes), so we loop until either the output buffer has
// data, the upstream signals done, or we hit a fatal error.
func (s *streamReader) Read(p []byte) (int, error) {
	for s.out.Len() == 0 && !s.done {
		if err := s.pumpFrame(); err != nil {
			if err == io.EOF {
				s.done = true
				break
			}
			s.err = err
			s.done = true
			return 0, err
		}
	}
	if s.out.Len() == 0 {
		if s.err != nil {
			return 0, s.err
		}
		return 0, io.EOF
	}
	return s.out.Read(p)
}

// Frame-buffer caps. A misbehaving backend that emits a single
// multi-GB line, or an event with a runaway number of lines, would
// otherwise grow our per-frame buffer without bound. The numbers are
// generous against real traffic — a long reasoning trace or a tool
// call with a serialized 1 MiB JSON argument blob still fits — but
// finite enough to keep one bad backend from killing the proxy.
const (
	maxStreamLineBytes  = 8 * 1024 * 1024
	maxStreamFrameLines = 1024
)

// pumpFrame reads one SSE frame from upstream, normalizes it, and
// writes the resulting bytes (possibly zero) into s.out. Returns
// io.EOF when upstream is fully drained.
//
// SSE line terminators per WHATWG §9.2.6 are CR, LF, or CRLF. We
// only split on LF: both backends in scope (llama.cpp llama-server,
// SwiftLM) emit LF/CRLF — a bare-CR-only stream would be read as one
// giant line and tripped by the per-line cap. Document this here so
// a future contributor knows the spec gap is deliberate, not missed.
//
// An SSE frame is a sequence of non-empty lines terminated by a blank
// line. We capture each line verbatim (including its line ending) so
// we can re-emit non-data lines (event:, id:, comments) without
// touching them. The terminating blank line is preserved too.
//
// Mid-stream EOF (no trailing blank) drops the partial frame: per
// spec, in-progress events on EOF are discarded. Re-emitting half a
// JSON payload would corrupt downstream parsing.
func (s *streamReader) pumpFrame() error {
	var lines [][]byte
	sawAnyLine := false
	for {
		line, err := readLineCapped(s.reader, maxStreamLineBytes)
		if len(line) > 0 {
			sawAnyLine = true
			if isBlankLine(line) {
				lines = append(lines, line)
				return s.emitFrame(lines)
			}
			lines = append(lines, line)
			if len(lines) > maxStreamFrameLines {
				return fmt.Errorf("normalize: SSE frame exceeded %d lines", maxStreamFrameLines)
			}
		}
		if err == io.EOF {
			if !sawAnyLine {
				return io.EOF
			}
			// Drop the partial frame; do not re-emit. Surfacing the EOF
			// lets the caller see the stream end cleanly.
			return io.EOF
		}
		if err != nil {
			return err
		}
	}
}

// readLineCapped reads up to a single '\n' (inclusive) but caps the
// total bytes consumed. Returns an error if the cap is exceeded
// before a newline is seen — preferable to silently truncating into a
// shorter "line" that wouldn't round-trip through SSE parsing.
func readLineCapped(r *bufio.Reader, max int) ([]byte, error) {
	var out []byte
	for {
		chunk, err := r.ReadSlice('\n')
		// chunk is only valid until the next ReadSlice — copy.
		out = append(out, chunk...)
		if len(out) > max {
			return nil, fmt.Errorf("normalize: SSE line exceeded %d bytes", max)
		}
		if err == bufio.ErrBufferFull {
			// Line longer than bufio's internal buffer: keep reading.
			continue
		}
		return out, err
	}
}

// emitFrame applies normalization to a captured frame's data: payload
// (if any) and writes the result to s.out. Frames with no data: line,
// or with a [DONE] / empty / malformed payload, pass through untouched.
func (s *streamReader) emitFrame(lines [][]byte) error {
	dataIdx, payload := findDataPayload(lines)
	if dataIdx < 0 {
		// No data: line at all (e.g. a comment or event-only frame).
		writeLines(&s.out, lines)
		return nil
	}
	// Special payloads that must not be JSON-parsed: empty keepalive
	// and the [DONE] terminator. Mirror anthropic.go:1433.
	trimmed := bytes.TrimSpace(payload)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("[DONE]")) {
		writeLines(&s.out, lines)
		return nil
	}

	var obj chunkObject
	if err := json.Unmarshal(trimmed, &obj); err != nil {
		// Malformed payload: pass through unchanged. Mirrors the
		// Anthropic translator's never-abort-the-stream posture.
		writeLines(&s.out, lines)
		return nil
	}

	if !applyChunkNormalizers(obj, s.opts) {
		// Frame dropped (prefill_progress). Emit nothing.
		return nil
	}

	encoded, err := json.Marshal(obj)
	if err != nil {
		writeLines(&s.out, lines)
		return nil
	}

	// Replace the data: line in place, preserving its original line
	// ending so we don't flip \r\n to \n mid-stream.
	rewritten := rewriteDataLine(lines[dataIdx], encoded)
	lines[dataIdx] = rewritten
	writeLines(&s.out, lines)
	return nil
}

// findDataPayload returns the index of the last data: line in a frame
// and its payload bytes (with the data: prefix and trailing newline
// stripped). Returns -1 if no data: line exists.
//
// Both backends (llama-server, SwiftLM) emit exactly one data: line
// per frame for chat completions. SSE itself permits multiple data:
// lines per frame, in which case spec-compliant clients concatenate
// them with `\n` before parsing. We do *not* support that case: only
// the last line is rewritten, earlier data: lines pass through
// unchanged, and a multi-line concatenation would yield malformed
// JSON to the client. This is a deliberate scope choice — if a future
// backend ever splits a single JSON payload across multiple data:
// lines we'd need to coalesce-then-rewrite. Add a regression test
// against the new backend at that point.
func findDataPayload(lines [][]byte) (int, []byte) {
	for i := len(lines) - 1; i >= 0; i-- {
		// SSE field is the substring before the first colon (§9.2.6
		// "Process the field"). HasPrefix("data:") would also match a
		// hypothetical "data-foo:" field; equality on the field name
		// avoids that.
		colon := bytes.IndexByte(lines[i], ':')
		if colon < 0 {
			continue
		}
		if bytes.Equal(lines[i][:colon], []byte("data")) {
			return i, dataLinePayload(lines[i])
		}
	}
	return -1, nil
}

// dataLinePayload extracts the payload from a "data: ..." line,
// stripping the prefix, an optional single space, and the trailing
// line terminator.
func dataLinePayload(line []byte) []byte {
	body := bytes.TrimPrefix(line, []byte("data:"))
	body = bytes.TrimRight(body, "\r\n")
	// SSE allows a single optional space after the colon; strip it
	// without eating leading whitespace inside the JSON itself.
	if len(body) > 0 && body[0] == ' ' {
		body = body[1:]
	}
	return body
}

// rewriteDataLine produces a replacement for an existing data: line,
// preserving its original line ending so the emitted stream's framing
// matches the upstream's choice (\n vs \r\n). The leading "data: "
// prefix is fixed since both backends use it.
func rewriteDataLine(original, payload []byte) []byte {
	ending := lineEnding(original)
	out := make([]byte, 0, len("data: ")+len(payload)+len(ending))
	out = append(out, "data: "...)
	out = append(out, payload...)
	out = append(out, ending...)
	return out
}

// lineEnding returns the trailing newline bytes from a line. Default
// to "\n" for lines without an explicit terminator (mid-frame
// upstream-EOF case).
func lineEnding(line []byte) []byte {
	if bytes.HasSuffix(line, []byte("\r\n")) {
		return []byte("\r\n")
	}
	if bytes.HasSuffix(line, []byte("\n")) {
		return []byte("\n")
	}
	return []byte("\n")
}

// isBlankLine reports whether a line consists only of a line
// terminator. SSE frame separator.
func isBlankLine(line []byte) bool {
	return bytes.Equal(line, []byte("\n")) || bytes.Equal(line, []byte("\r\n"))
}

func writeLines(w *bytes.Buffer, lines [][]byte) {
	for _, l := range lines {
		w.Write(l)
	}
}
