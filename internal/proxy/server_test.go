package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestGenerateRequestID(t *testing.T) {
	id1 := generateRequestID()
	id2 := generateRequestID()

	if !strings.HasPrefix(id1, "req_") {
		t.Errorf("request ID should start with 'req_', got %s", id1)
	}
	if len(id1) != 28 { // "req_" + 24 hex chars
		t.Errorf("request ID should be 28 chars, got %d", len(id1))
	}
	if id1 == id2 {
		t.Errorf("request IDs should be unique")
	}
}

func TestWriteAnthropicError(t *testing.T) {
	s := &Server{}
	w := httptest.NewRecorder()
	requestID := "req_test123"

	s.writeAnthropicError(w, requestID, http.StatusBadRequest, AnthropicInvalidRequest, "test error message")

	// Check status code
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}

	// Check headers
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got '%s'", ct)
	}
	if rid := w.Header().Get("request-id"); rid != requestID {
		t.Errorf("expected request-id '%s', got '%s'", requestID, rid)
	}

	// Check body structure
	var resp AnthropicError
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if resp.Type != "error" {
		t.Errorf("expected type 'error', got '%s'", resp.Type)
	}
	if resp.Error.Type != AnthropicInvalidRequest {
		t.Errorf("expected error type '%s', got '%s'", AnthropicInvalidRequest, resp.Error.Type)
	}
	if resp.Error.Message != "test error message" {
		t.Errorf("expected message 'test error message', got '%s'", resp.Error.Message)
	}
	if resp.RequestID != requestID {
		t.Errorf("expected request_id '%s', got '%s'", requestID, resp.RequestID)
	}
}

func TestAnthropicErrorTypes(t *testing.T) {
	tests := []struct {
		errType    AnthropicErrorType
		httpStatus int
	}{
		{AnthropicInvalidRequest, http.StatusBadRequest},
		{AnthropicAuthentication, http.StatusUnauthorized},
		{AnthropicPermission, http.StatusForbidden},
		{AnthropicNotFound, http.StatusNotFound},
		{AnthropicRequestTooLarge, http.StatusRequestEntityTooLarge},
		{AnthropicRateLimit, http.StatusTooManyRequests},
		{AnthropicAPIError, http.StatusInternalServerError},
		{AnthropicOverloaded, 529},
	}

	s := &Server{}
	for _, tt := range tests {
		t.Run(string(tt.errType), func(t *testing.T) {
			w := httptest.NewRecorder()
			s.writeAnthropicError(w, "req_test", tt.httpStatus, tt.errType, "test")

			var resp AnthropicError
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}
			if resp.Error.Type != tt.errType {
				t.Errorf("expected error type %s, got %s", tt.errType, resp.Error.Type)
			}
		})
	}
}

func TestAnthropicEndpointMethodValidation(t *testing.T) {
	s := &Server{config: DefaultConfig()}

	tests := []struct {
		method   string
		endpoint string
		handler  func(http.ResponseWriter, *http.Request)
	}{
		{http.MethodGet, "/v1/messages", s.handleAnthropicMessages},
		{http.MethodPut, "/v1/messages", s.handleAnthropicMessages},
		{http.MethodDelete, "/v1/messages", s.handleAnthropicMessages},
		{http.MethodGet, "/v1/messages/count_tokens", s.handleAnthropicCountTokens},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.endpoint, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.endpoint, nil)
			w := httptest.NewRecorder()
			tt.handler(w, req)

			if w.Code != http.StatusMethodNotAllowed {
				t.Errorf("expected status %d, got %d", http.StatusMethodNotAllowed, w.Code)
			}

			// Verify Anthropic error format
			var resp AnthropicError
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}
			if resp.Type != "error" {
				t.Errorf("expected type 'error', got '%s'", resp.Type)
			}
			if resp.Error.Type != AnthropicInvalidRequest {
				t.Errorf("expected error type '%s', got '%s'", AnthropicInvalidRequest, resp.Error.Type)
			}
		})
	}
}

func TestAnthropicEndpointBodyValidation(t *testing.T) {
	s := &Server{config: DefaultConfig()}

	tests := []struct {
		name        string
		body        string
		expectError AnthropicErrorType
		expectMsg   string
	}{
		{
			name:        "invalid json",
			body:        "not json",
			expectError: AnthropicInvalidRequest,
			expectMsg:   "Failed to parse request body as JSON",
		},
		{
			name:        "empty json",
			body:        "{}",
			expectError: AnthropicInvalidRequest,
			expectMsg:   "model: Field required",
		},
		{
			name:        "empty model",
			body:        `{"model": ""}`,
			expectError: AnthropicInvalidRequest,
			expectMsg:   "model: Field required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			s.handleAnthropicMessages(w, req)

			if w.Code != http.StatusBadRequest {
				t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
			}

			var resp AnthropicError
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}
			if resp.Error.Type != tt.expectError {
				t.Errorf("expected error type '%s', got '%s'", tt.expectError, resp.Error.Type)
			}
			if resp.Error.Message != tt.expectMsg {
				t.Errorf("expected message '%s', got '%s'", tt.expectMsg, resp.Error.Message)
			}
			// Verify request-id header is present
			if rid := w.Header().Get("request-id"); !strings.HasPrefix(rid, "req_") {
				t.Errorf("expected request-id header with 'req_' prefix, got '%s'", rid)
			}
		})
	}
}

func TestOpenAIEndpointReturnsOpenAIErrors(t *testing.T) {
	s := &Server{config: DefaultConfig()}

	// Test that OpenAI endpoints still return OpenAI-formatted errors
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString("{}"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	s.handleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}

	var resp OpenAIError
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal as OpenAI error: %v", err)
	}
	if resp.Error.Type != "invalid_request" {
		t.Errorf("expected OpenAI error type 'invalid_request', got '%s'", resp.Error.Type)
	}
}

// TestHandleModelsSortedByLastUsed verifies that /v1/models returns downloaded
// models ordered by most recently used first, falling back to the GGUF file's
// mtime when no persisted last-used metadata exists.
func TestHandleModelsSortedByLastUsed(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("LLEME_HOME", tmpDir)

	modelsDir := filepath.Join(tmpDir, "models")
	// Create three models and set distinct mtimes so the fallback path is ordered.
	specs := []struct {
		user, repo, quant string
		mtime             time.Time
	}{
		{"u", "old", "Q4_K_M", time.Now().Add(-72 * time.Hour)},
		{"u", "recent", "Q4_K_M", time.Now().Add(-1 * time.Hour)},
		{"u", "middle", "Q4_K_M", time.Now().Add(-24 * time.Hour)},
	}
	for _, sp := range specs {
		dir := filepath.Join(modelsDir, sp.user, sp.repo)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, sp.quant+".gguf")
		if err := os.WriteFile(path, []byte("fake"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, sp.mtime, sp.mtime); err != nil {
			t.Fatal(err)
		}
	}

	s := &Server{
		manager:   NewModelManager(DefaultConfig(), nil),
		config:    DefaultConfig(),
		startedAt: time.Now(),
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	s.handleModels(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp OpenAIModelsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	wantOrder := []string{
		"u/recent:Q4_K_M",
		"u/middle:Q4_K_M",
		"u/old:Q4_K_M",
	}
	if len(resp.Data) != len(wantOrder) {
		t.Fatalf("got %d models, want %d", len(resp.Data), len(wantOrder))
	}
	for i, want := range wantOrder {
		if resp.Data[i].ID != want {
			t.Errorf("position %d: got %s, want %s", i, resp.Data[i].ID, want)
		}
	}
}
