package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Tests for the small pure helpers added in handleChatCompletions and
// the readBodyRaw / parseModelField extractions. These are the
// surface that the rest of the chat-completions path depends on, so
// they get direct coverage rather than relying on full handler
// integration tests (which would need a Manager + fake backend).

func TestParseModelField(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		want    string
		wantErr string
	}{
		{"valid", `{"model":"user/repo"}`, "user/repo", ""},
		{"with extra fields", `{"model":"x","stream":true,"messages":[]}`, "x", ""},
		{"missing field", `{}`, "", "model: Field required"},
		{"empty model", `{"model":""}`, "", "model: Field required"},
		{"malformed JSON", `not json`, "", "Failed to parse request body as JSON"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseModelField([]byte(tt.body))
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if got != tt.want {
					t.Errorf("got %q, want %q", got, tt.want)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error %q, got nil (model=%q)", tt.wantErr, got)
			}
			if err.Error() != tt.wantErr {
				t.Errorf("got error %q, want %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestRequestWantsStream(t *testing.T) {
	tests := []struct {
		body string
		want bool
	}{
		{`{"stream":true}`, true},
		{`{"stream":false}`, false},
		{`{}`, false},
		{`{"messages":[],"stream":true}`, true},
		{`not json`, false}, // defensive default — caller already validated body
	}
	for _, tt := range tests {
		t.Run(tt.body, func(t *testing.T) {
			if got := requestWantsStream([]byte(tt.body)); got != tt.want {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOpenAIErrorTypeForStatus(t *testing.T) {
	tests := []struct {
		status int
		want   string
	}{
		{http.StatusBadRequest, "invalid_request_error"},
		{http.StatusUnauthorized, "authentication_error"},
		{http.StatusForbidden, "permission_error"},
		{http.StatusNotFound, "not_found_error"},
		{http.StatusRequestEntityTooLarge, "invalid_request_error"},
		{http.StatusTooManyRequests, "rate_limit_error"},
		{http.StatusServiceUnavailable, "server_error"},
		{http.StatusInternalServerError, "server_error"},
		{http.StatusBadGateway, "server_error"},
		{418, "invalid_request_error"}, // unmapped 4xx → invalid_request_error
		{599, "server_error"},          // unmapped 5xx → server_error
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.status), func(t *testing.T) {
			if got := openAIErrorTypeForStatus(tt.status); got != tt.want {
				t.Errorf("status %d: got %q, want %q", tt.status, got, tt.want)
			}
		})
	}
}

func TestReadBodyRaw(t *testing.T) {
	t.Run("happy path", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(`{"a":1}`))
		w := httptest.NewRecorder()
		body, rbe := readBodyRaw(w, req)
		if rbe != nil {
			t.Fatalf("unexpected error: %+v", rbe)
		}
		if string(body) != `{"a":1}` {
			t.Errorf("body = %q, want %q", body, `{"a":1}`)
		}
	})

	t.Run("rejects non-POST", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		w := httptest.NewRecorder()
		_, rbe := readBodyRaw(w, req)
		if rbe == nil || rbe.status != http.StatusMethodNotAllowed {
			t.Errorf("expected MethodNotAllowed, got %+v", rbe)
		}
	})

	t.Run("oversized body", func(t *testing.T) {
		oversized := strings.Repeat("a", maxRequestBodyBytes+1)
		req := httptest.NewRequest(http.MethodPost, "/", bytes.NewBufferString(oversized))
		w := httptest.NewRecorder()
		_, rbe := readBodyRaw(w, req)
		if rbe == nil || rbe.status != http.StatusRequestEntityTooLarge {
			t.Errorf("expected RequestEntityTooLarge, got %+v", rbe)
		}
	})
}

// TestReadOpenAIBodyTranslatesError — readBodyRaw is shared between
// the OpenAI and Anthropic surfaces; verify the OpenAI wrapper picks
// the right error type for each status.
func TestReadOpenAIBodyTranslatesError(t *testing.T) {
	s := &Server{}

	t.Run("non-POST -> method_not_allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
		w := httptest.NewRecorder()
		_, ok := s.readOpenAIBody(w, req)
		if ok {
			t.Fatal("expected ok=false")
		}
		var resp OpenAIError
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if resp.Error.Type != "method_not_allowed" {
			t.Errorf("error type = %q, want method_not_allowed", resp.Error.Type)
		}
	})

	t.Run("oversize -> invalid_request", func(t *testing.T) {
		oversized := strings.Repeat("a", maxRequestBodyBytes+1)
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(oversized))
		w := httptest.NewRecorder()
		_, ok := s.readOpenAIBody(w, req)
		if ok {
			t.Fatal("expected ok=false")
		}
		var resp OpenAIError
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if resp.Error.Type != "invalid_request" {
			t.Errorf("error type = %q, want invalid_request", resp.Error.Type)
		}
	})
}

func TestCopyResponseHeaders(t *testing.T) {
	t.Run("strips hop-by-hop and Content-Length", func(t *testing.T) {
		resp := &http.Response{Header: http.Header{}}
		resp.Header.Set("Content-Type", "application/json")
		resp.Header.Set("Content-Length", "42")
		resp.Header.Set("Connection", "keep-alive")
		resp.Header.Set("Transfer-Encoding", "chunked")
		resp.Header.Set("X-Custom", "preserved")
		resp.Header.Set("Request-Id", "req_abc")

		w := httptest.NewRecorder()
		copyResponseHeaders(w, resp, false)

		got := w.Header()
		for _, h := range []string{"Content-Length", "Connection", "Transfer-Encoding"} {
			if v := got.Get(h); v != "" {
				t.Errorf("expected %s stripped, got %q", h, v)
			}
		}
		if got.Get("Content-Type") != "application/json" {
			t.Errorf("Content-Type lost: %q", got.Get("Content-Type"))
		}
		if got.Get("X-Custom") != "preserved" {
			t.Errorf("X-Custom lost: %q", got.Get("X-Custom"))
		}
		if got.Get("Request-Id") != "req_abc" {
			t.Errorf("Request-Id lost: %q", got.Get("Request-Id"))
		}
	})

	t.Run("streaming skips Content-Type and Cache-Control", func(t *testing.T) {
		// The streaming path sets these explicitly to text/event-stream
		// and no-cache; passing through backend values would double-set.
		resp := &http.Response{Header: http.Header{}}
		resp.Header.Set("Content-Type", "application/json")
		resp.Header.Set("Cache-Control", "max-age=300")
		resp.Header.Set("X-Custom", "preserved")

		w := httptest.NewRecorder()
		copyResponseHeaders(w, resp, true)

		got := w.Header()
		if v := got.Get("Content-Type"); v != "" {
			t.Errorf("streaming should skip backend Content-Type, got %q", v)
		}
		if v := got.Get("Cache-Control"); v != "" {
			t.Errorf("streaming should skip backend Cache-Control, got %q", v)
		}
		if got.Get("X-Custom") != "preserved" {
			t.Errorf("X-Custom lost: %q", got.Get("X-Custom"))
		}
	})
}

// TestWriteNormalizedResponseSetsContentLength — Content-Length must
// match the bytes actually written, since the normalize layer changes
// payload size from upstream's. Getting this wrong would cause clients
// to hang waiting for bytes that never come, or to truncate.
func TestWriteNormalizedResponseSetsContentLength(t *testing.T) {
	body := []byte(`{"normalized":true}`)
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{},
	}
	resp.Header.Set("Content-Type", "application/json")
	resp.Header.Set("Content-Length", "999") // wrong — should be replaced

	w := httptest.NewRecorder()
	writeNormalizedResponse(w, resp, bytes.NewReader(body))

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	gotLen := w.Header().Get("Content-Length")
	wantLen := "19"
	if gotLen != wantLen {
		t.Errorf("Content-Length = %q, want %q", gotLen, wantLen)
	}
	if w.Body.Len() != len(body) {
		t.Errorf("body len = %d, want %d", w.Body.Len(), len(body))
	}
}

// TestWriteNormalizedStreamSetsSSEHeaders — confirms the streaming
// path overrides backend headers with SSE-correct values, regardless
// of what the backend sent.
func TestWriteNormalizedStreamSetsSSEHeaders(t *testing.T) {
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{},
	}
	// Backend sent something else; we must override.
	resp.Header.Set("Content-Type", "application/json")
	resp.Header.Set("Cache-Control", "max-age=300")

	body := strings.NewReader("data: {\"object\":\"chat.completion.chunk\"}\n\n")
	w := httptest.NewRecorder()
	writeNormalizedStream(w, resp, body)

	if got := w.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", got)
	}
	if got := w.Header().Get("Cache-Control"); got != "no-cache" {
		t.Errorf("Cache-Control = %q, want no-cache", got)
	}
	// Connection is hop-by-hop; net/http drops it. We deliberately do
	// not set it from the handler — verify it's absent.
	if got := w.Header().Get("Connection"); got != "" {
		t.Errorf("Connection should not be set by handler; got %q", got)
	}
	if !strings.Contains(w.Body.String(), "chat.completion.chunk") {
		t.Errorf("body did not include upstream content: %q", w.Body.String())
	}
}
