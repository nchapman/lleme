package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewAPIClientFromURL(t *testing.T) {
	t.Run("creates client with provided URL", func(t *testing.T) {
		api := NewAPIClientFromURL("http://localhost:9000")
		if api == nil {
			t.Fatal("Expected non-nil APIClient")
		}
		if api.baseURL != "http://localhost:9000" {
			t.Errorf("Expected baseURL http://localhost:9000, got %s", api.baseURL)
		}
		if api.client == nil {
			t.Error("Expected non-nil HTTP client")
		}
	})

	t.Run("preserves URL with path", func(t *testing.T) {
		api := NewAPIClientFromURL("http://example.com/api/v1")
		if api.baseURL != "http://example.com/api/v1" {
			t.Errorf("Expected baseURL http://example.com/api/v1, got %s", api.baseURL)
		}
	})
}

func TestHealth(t *testing.T) {
	t.Run("returns nil on successful health check", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/health" {
				t.Errorf("Expected path /health, got %s", r.URL.Path)
			}
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status":"ok"}`))
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		err := api.Health()
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("returns error on failed health check", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		err := api.Health()
		if err == nil {
			t.Error("Expected error for failed health check, got nil")
		}
	})
}

func TestChatCompletion(t *testing.T) {
	t.Run("successful chat completion", func(t *testing.T) {
		expectedReq := ChatCompletionRequest{
			Model: "test-model",
			Messages: []ChatMessage{
				{Role: "user", Content: "Hello"},
			},
			Stream: false,
		}

		expectedResp := ChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "test-model",
			Choices: []Choice{
				{
					Index:        0,
					Message:      ChatMessage{Role: "assistant", Content: "Hello! How can I help?"},
					FinishReason: "stop",
				},
			},
		}

		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("Expected POST method, got %s", r.Method)
			}
			if r.URL.Path != "/v1/chat/completions" {
				t.Errorf("Expected path /v1/chat/completions, got %s", r.URL.Path)
			}

			var req ChatCompletionRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("Failed to decode request: %v", err)
			}

			if req.Model != expectedReq.Model {
				t.Errorf("Expected model %s, got %s", expectedReq.Model, req.Model)
			}
			if len(req.Messages) != 1 {
				t.Fatalf("Expected 1 message, got %d", len(req.Messages))
			}
			if req.Messages[0].Content != expectedReq.Messages[0].Content {
				t.Errorf("Expected message %s, got %s", expectedReq.Messages[0].Content, req.Messages[0].Content)
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(expectedResp)
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		resp, err := api.ChatCompletion(&expectedReq)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if resp.ID != expectedResp.ID {
			t.Errorf("Expected ID %s, got %s", expectedResp.ID, resp.ID)
		}
		if resp.Model != expectedResp.Model {
			t.Errorf("Expected model %s, got %s", expectedResp.Model, resp.Model)
		}
		if len(resp.Choices) != 1 {
			t.Fatalf("Expected 1 choice, got %d", len(resp.Choices))
		}
		if resp.Choices[0].Message.Content != expectedResp.Choices[0].Message.Content {
			t.Errorf("Expected content %s, got %s", expectedResp.Choices[0].Message.Content, resp.Choices[0].Message.Content)
		}
	})

	t.Run("returns error on failed request", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":"invalid request"}`))
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		req := &ChatCompletionRequest{
			Model:    "test-model",
			Messages: []ChatMessage{{Role: "user", Content: "Hello"}},
			Stream:   false,
		}

		_, err := api.ChatCompletion(req)
		if err == nil {
			t.Error("Expected error for failed request, got nil")
		}
	})

	t.Run("returns error on invalid JSON response", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`invalid json`))
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		req := &ChatCompletionRequest{
			Model:    "test-model",
			Messages: []ChatMessage{{Role: "user", Content: "Hello"}},
			Stream:   false,
		}

		_, err := api.ChatCompletion(req)
		if err == nil {
			t.Error("Expected error for invalid JSON, got nil")
		}
	})
}

func TestStreamChatCompletion(t *testing.T) {
	t.Run("successful streaming chat completion", func(t *testing.T) {
		expectedReq := ChatCompletionRequest{
			Model: "test-model",
			Messages: []ChatMessage{
				{Role: "user", Content: "Hello"},
			},
			Stream: true,
		}

		chunks := []string{
			"Hello",
			" there",
			"!",
		}

		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("Expected POST method, got %s", r.Method)
			}
			if r.URL.Path != "/v1/chat/completions" {
				t.Errorf("Expected path /v1/chat/completions, got %s", r.URL.Path)
			}

			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatal("Expected streaming support")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(http.StatusOK)

			for i, chunk := range chunks {
				streamChunk := StreamChunk{
					ID:      "test-id",
					Object:  "chat.completion.chunk",
					Created: 1234567890,
					Model:   "test-model",
					Choices: []StreamChoice{
						{
							Index: 0,
							Delta: StreamDelta{Role: "assistant", Content: chunk},
						},
					},
				}

				jsonData, _ := json.Marshal(streamChunk)
				fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
				flusher.Flush()

				if i == len(chunks)-1 {
					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
				}
			}
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		var receivedChunks []string
		err := api.StreamChatCompletion(context.Background(), &expectedReq, StreamCallback{
			ContentCallback: func(content string) {
				receivedChunks = append(receivedChunks, content)
			},
		})

		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if len(receivedChunks) != len(chunks) {
			t.Fatalf("Expected %d chunks, got %d", len(chunks), len(receivedChunks))
		}

		for i, expected := range chunks {
			if receivedChunks[i] != expected {
				t.Errorf("Chunk %d: expected %s, got %s", i, expected, receivedChunks[i])
			}
		}
	})

	t.Run("handles empty lines and DONE marker", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			fmt.Fprintln(w, "")
			fmt.Fprintln(w, "data: [DONE]")
			fmt.Fprintln(w, "")
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		callCount := 0
		err := api.StreamChatCompletion(context.Background(), &ChatCompletionRequest{}, StreamCallback{
			ContentCallback: func(string) {
				callCount++
			},
		})

		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if callCount != 0 {
			t.Errorf("Expected callback not to be called, got %d calls", callCount)
		}
	})

	t.Run("returns error on failed request", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":"invalid request"}`))
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		req := &ChatCompletionRequest{
			Model:    "test-model",
			Messages: []ChatMessage{{Role: "user", Content: "Hello"}},
			Stream:   true,
		}

		err := api.StreamChatCompletion(context.Background(), req, StreamCallback{})
		if err == nil {
			t.Error("Expected error for failed request, got nil")
		}
	})

	t.Run("skips invalid JSON chunks", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			validChunk := StreamChunk{
				ID:      "test-id",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "test-model",
				Choices: []StreamChoice{
					{
						Index: 0,
						Delta: StreamDelta{Role: "assistant", Content: "valid"},
					},
				},
			}

			jsonData, _ := json.Marshal(validChunk)
			fmt.Fprintf(w, "data: invalid json\n\n")
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		callCount := 0
		err := api.StreamChatCompletion(context.Background(), &ChatCompletionRequest{}, StreamCallback{
			ContentCallback: func(string) {
				callCount++
			},
		})

		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if callCount != 1 {
			t.Errorf("Expected 1 callback call (valid chunk), got %d", callCount)
		}
	})

	t.Run("handles reasoning_content separately from content", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatal("Expected streaming support")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			// First send reasoning content
			reasoningChunk := StreamChunk{
				ID:      "test-id",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "test-model",
				Choices: []StreamChoice{
					{
						Index: 0,
						Delta: StreamDelta{ReasoningContent: "Let me think..."},
					},
				},
			}
			jsonData, _ := json.Marshal(reasoningChunk)
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
			flusher.Flush()

			// Then send actual content
			contentChunk := StreamChunk{
				ID:      "test-id",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "test-model",
				Choices: []StreamChoice{
					{
						Index: 0,
						Delta: StreamDelta{Content: "The answer is 42"},
					},
				},
			}
			jsonData, _ = json.Marshal(contentChunk)
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
			flusher.Flush()

			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		var reasoningChunks []string
		var contentChunks []string
		err := api.StreamChatCompletion(context.Background(), &ChatCompletionRequest{}, StreamCallback{
			ReasoningCallback: func(reasoning string) {
				reasoningChunks = append(reasoningChunks, reasoning)
			},
			ContentCallback: func(content string) {
				contentChunks = append(contentChunks, content)
			},
		})

		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if len(reasoningChunks) != 1 || reasoningChunks[0] != "Let me think..." {
			t.Errorf("Expected reasoning chunk 'Let me think...', got %v", reasoningChunks)
		}

		if len(contentChunks) != 1 || contentChunks[0] != "The answer is 42" {
			t.Errorf("Expected content chunk 'The answer is 42', got %v", contentChunks)
		}
	})

	t.Run("handles reasoning_content with nil ReasoningCallback", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatal("Expected streaming support")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			// Send reasoning content (should be ignored since no callback)
			reasoningChunk := StreamChunk{
				ID:      "test-id",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "test-model",
				Choices: []StreamChoice{
					{
						Index: 0,
						Delta: StreamDelta{ReasoningContent: "Thinking..."},
					},
				},
			}
			jsonData, _ := json.Marshal(reasoningChunk)
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
			flusher.Flush()

			// Send actual content
			contentChunk := StreamChunk{
				ID:      "test-id",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "test-model",
				Choices: []StreamChoice{
					{
						Index: 0,
						Delta: StreamDelta{Content: "Hello"},
					},
				},
			}
			jsonData, _ = json.Marshal(contentChunk)
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
			flusher.Flush()

			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		var contentChunks []string
		// Only provide ContentCallback, no ReasoningCallback - should not panic
		err := api.StreamChatCompletion(context.Background(), &ChatCompletionRequest{}, StreamCallback{
			ContentCallback: func(content string) {
				contentChunks = append(contentChunks, content)
			},
		})

		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if len(contentChunks) != 1 || contentChunks[0] != "Hello" {
			t.Errorf("Expected content chunk 'Hello', got %v", contentChunks)
		}
	})

	t.Run("returns context.Canceled when context is cancelled", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			// Simulate slow streaming - will be interrupted by context cancellation
			flusher, ok := w.(http.Flusher)
			if !ok {
				return
			}

			// Send one chunk, then stall
			chunk := StreamChunk{
				ID:      "test-id",
				Object:  "chat.completion.chunk",
				Created: 1234567890,
				Model:   "test-model",
				Choices: []StreamChoice{
					{Index: 0, Delta: StreamDelta{Content: "Hello"}},
				},
			}
			jsonData, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
			flusher.Flush()

			// Block until request context is done (simulates slow server)
			<-r.Context().Done()
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		ctx, cancel := context.WithCancel(context.Background())

		// Cancel after a short delay
		go func() {
			cancel()
		}()

		err := api.StreamChatCompletion(ctx, &ChatCompletionRequest{}, StreamCallback{})

		if err == nil {
			t.Error("Expected error when context is cancelled, got nil")
		}
		// Error could be context.Canceled or a wrapped error from the HTTP client
		if !errors.Is(err, context.Canceled) && !strings.Contains(err.Error(), "context canceled") {
			t.Errorf("Expected context.Canceled error, got: %v", err)
		}
	})
}

func TestSetModel(t *testing.T) {
	t.Run("successful model load", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("Expected POST method, got %s", r.Method)
			}
			if r.URL.Path != "/v1/load" {
				t.Errorf("Expected path /v1/load, got %s", r.URL.Path)
			}

			var req struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("Failed to decode request: %v", err)
			}

			if req.Model != "/path/to/model.gguf" {
				t.Errorf("Expected model path /path/to/model.gguf, got %s", req.Model)
			}

			w.WriteHeader(http.StatusOK)
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		err := api.SetModel("/path/to/model.gguf")
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})

	t.Run("returns error on failed load", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":"failed to load model"}`))
		}))
		defer ts.Close()

		api := &APIClient{
			baseURL: ts.URL,
			client:  ts.Client(),
		}

		err := api.SetModel("/path/to/model.gguf")
		if err == nil {
			t.Error("Expected error for failed load, got nil")
		}
	})
}

func TestChatMessageSerialization(t *testing.T) {
	msg := ChatMessage{
		Role:    "user",
		Content: "Hello, world!",
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("Failed to marshal ChatMessage: %v", err)
	}

	var decoded ChatMessage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal ChatMessage: %v", err)
	}

	if decoded.Role != msg.Role {
		t.Errorf("Expected Role %s, got %s", msg.Role, decoded.Role)
	}
	if decoded.Content != msg.Content {
		t.Errorf("Expected Content %s, got %s", msg.Content, decoded.Content)
	}
}

func TestStreamChunkSerialization(t *testing.T) {
	chunk := StreamChunk{
		ID:      "test-id",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []StreamChoice{
			{
				Index: 0,
				Delta: StreamDelta{Role: "assistant", Content: "Hello"},
			},
		},
	}

	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("Failed to marshal StreamChunk: %v", err)
	}

	var decoded StreamChunk
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal StreamChunk: %v", err)
	}

	if decoded.ID != chunk.ID {
		t.Errorf("Expected ID %s, got %s", chunk.ID, decoded.ID)
	}
	if len(decoded.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(decoded.Choices))
	}
	if decoded.Choices[0].Delta.Content != "Hello" {
		t.Errorf("Expected content 'Hello', got %s", decoded.Choices[0].Delta.Content)
	}
}
