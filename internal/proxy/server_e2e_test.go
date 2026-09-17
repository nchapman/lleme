package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/nchapman/lleme/internal/config"
)

// newTestServer builds a full proxy Server (real mux, real manager) backed
// by the fake backend runtime, served via httptest.
func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	m := newLifecycleManager(t, 3)

	appCfg := &config.Config{}
	cfg := DefaultConfig()
	cfg.CORSOrigins = []string{"http://localhost:3000"}

	s := &Server{
		manager:   m,
		config:    cfg,
		appConfig: appCfg,
		startedAt: time.Now(),
	}
	m.SetStateChangeCallback(func() {
		s.saveState()
	})

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/stop", s.handleStopModel)

	ts := httptest.NewServer(CORSMiddleware(cfg.CORSOrigins)(mux))
	t.Cleanup(func() {
		ts.Close()
		_ = m.StopAllBackends()
	})
	return ts
}

func postJSON(t *testing.T, url, body string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	data, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	return resp, string(data)
}

func TestProxyChatCompletionsEndToEnd(t *testing.T) {
	ts := newTestServer(t)

	resp, body := postJSON(t, ts.URL+"/v1/chat/completions",
		`{"model":"testuser/alpha","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if !strings.Contains(body, "hello from fake backend") {
		t.Errorf("response did not pass through the backend: %s", body)
	}
}

func TestProxyChatCompletionsUnknownModel(t *testing.T) {
	ts := newTestServer(t)

	resp, body := postJSON(t, ts.URL+"/v1/chat/completions",
		`{"model":"testuser/gamma","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("expected error status, got 200: %s", body)
	}
	var errResp OpenAIError
	if err := json.Unmarshal([]byte(body), &errResp); err != nil {
		t.Fatalf("error body is not OpenAI-shaped: %v (%s)", err, body)
	}
	if errResp.Error.Message == "" {
		t.Errorf("error message empty: %s", body)
	}
}

func TestProxyAnthropicMessagesEndToEnd(t *testing.T) {
	ts := newTestServer(t)

	resp, body := postJSON(t, ts.URL+"/v1/messages",
		`{"model":"testuser/alpha","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	var result map[string]any
	if err := json.Unmarshal([]byte(body), &result); err != nil {
		t.Fatalf("invalid JSON: %v (%s)", err, body)
	}
	if result["type"] != "message" {
		t.Errorf("type = %v, want message", result["type"])
	}
	if result["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", result["role"])
	}
	content, _ := json.Marshal(result["content"])
	if !strings.Contains(string(content), "hello from fake backend") {
		t.Errorf("translated content missing backend text: %s", body)
	}
	if usage, ok := result["usage"].(map[string]any); !ok || usage["input_tokens"] == nil {
		t.Errorf("usage.input_tokens missing: %s", body)
	}
}

func TestProxyStatusAndHealth(t *testing.T) {
	ts := newTestServer(t)

	resp, body := postJSON(t, ts.URL+"/v1/chat/completions",
		`{"model":"testuser/alpha","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("warm-up request failed: %d %s", resp.StatusCode, body)
	}

	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("/health = %d, want 200", resp.StatusCode)
	}

	resp, err = http.Get(ts.URL + "/api/status")
	if err != nil {
		t.Fatal(err)
	}
	var status ProxyStatus
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/api/status = %d: %s", resp.StatusCode, data)
	}
	if err := json.Unmarshal(data, &status); err != nil {
		t.Fatalf("decode status: %v (%s)", err, data)
	}
	if status.LoadedCount != 1 || len(status.Models) != 1 {
		t.Errorf("status reports %d models, want 1: %s", status.LoadedCount, data)
	}
	if status.Models[0].Status != "ready" || status.Models[0].PID == 0 {
		t.Errorf("model status incomplete: %+v", status.Models[0])
	}

	// /v1/models must list the loaded model with lleme metadata.
	resp, err = http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	data, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Contains(data, []byte("testuser/alpha")) {
		t.Errorf("/v1/models missing loaded model: %s", data)
	}
}

func TestProxyStopModel(t *testing.T) {
	ts := newTestServer(t)

	resp, body := postJSON(t, ts.URL+"/v1/chat/completions",
		`{"model":"testuser/alpha","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("warm-up request failed: %d %s", resp.StatusCode, body)
	}

	resp, body = postJSON(t, ts.URL+"/api/stop", `{"model":"testuser/alpha"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/api/stop = %d: %s", resp.StatusCode, body)
	}

	resp, body = postJSON(t, ts.URL+"/api/stop", `{"model":"testuser/gamma"}`)
	if resp.StatusCode == http.StatusOK {
		t.Errorf("stopping unknown model returned 200: %s", body)
	}
}

func TestCORSMiddleware(t *testing.T) {
	ts := newTestServer(t)

	t.Run("allowed origin", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodOptions, ts.URL+"/v1/models", nil)
		req.Header.Set("Origin", "http://localhost:3000")
		req.Header.Set("Access-Control-Request-Method", "POST")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusNoContent {
			t.Errorf("preflight = %d, want 204", resp.StatusCode)
		}
		if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "http://localhost:3000" {
			t.Errorf("ACAO = %q, want echo of allowed origin", got)
		}
	})

	t.Run("disallowed origin gets no CORS grant", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodGet, ts.URL+"/health", nil)
		req.Header.Set("Origin", "http://evil.example")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "" {
			t.Errorf("ACAO = %q for disallowed origin, want empty", got)
		}
	})
}
