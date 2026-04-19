package proxy

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/logs"
	"github.com/nchapman/lleme/internal/version"
)

// maxRequestBodyBytes caps inbound chat/messages request bodies to protect against
// unbounded memory use. 10 MiB is far larger than any reasonable prompt + history.
const maxRequestBodyBytes = 10 * 1024 * 1024

// Server is the main proxy server that routes requests to backends
type Server struct {
	httpServer   *http.Server
	manager      *ModelManager
	idleMonitor  *IdleMonitor
	config       *Config
	startedAt    time.Time
	shutdownChan chan struct{}
	stateMu      sync.Mutex // protects state file writes
}

// NewServer creates a new proxy server
func NewServer(cfg *Config, appCfg *config.Config) *Server {
	// Clean up any orphaned backends from a previous crash
	CleanupOrphanedBackends()

	manager := NewModelManager(cfg, appCfg)

	s := &Server{
		manager:      manager,
		config:       cfg,
		startedAt:    time.Now(),
		shutdownChan: make(chan struct{}),
	}

	// Set up state persistence callback
	manager.SetStateChangeCallback(func() {
		s.saveState()
	})

	// Create idle monitor
	s.idleMonitor = NewIdleMonitor(manager, cfg.IdleTimeout, 60*time.Second)

	// Setup HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/completions", s.handleCompletions)
	mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("/v1/models", s.handleModels)

	// Anthropic Messages API
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/v1/messages/count_tokens", s.handleAnthropicCountTokens)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/run", s.handleRun)
	mux.HandleFunc("/api/stop", s.handleStopModel)
	mux.HandleFunc("/api/stop-all", s.handleStopAll)

	// Serve embedded web UI at root
	mux.Handle("/", newWebUIHandler())

	// Apply CORS middleware
	handler := CORSMiddleware(cfg.CORSOrigins)(mux)

	s.httpServer = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler: handler,
	}

	return s
}

// Start starts the proxy server
func (s *Server) Start() error {
	// Start idle monitor
	s.idleMonitor.Start()

	ln, err := net.Listen("tcp", s.httpServer.Addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", s.httpServer.Addr, err)
	}

	go func() {
		if err := s.httpServer.Serve(ln); err != nil && err != http.ErrServerClosed {
			fmt.Printf("Server error: %v\n", err)
		}
	}()

	// Save initial state (no backends yet)
	s.saveState()

	return nil
}

// Stop gracefully stops the proxy server
func (s *Server) Stop() error {
	close(s.shutdownChan)

	// Stop idle monitor
	s.idleMonitor.Stop()

	// Stop all backends
	s.manager.StopAllBackends()

	// Shutdown HTTP server with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	return s.httpServer.Shutdown(ctx)
}

// saveState persists the current proxy and backend state to disk
func (s *Server) saveState() {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	backends := s.manager.ListBackends()
	var backendStates []BackendState
	for _, b := range backends {
		if b.Status == "ready" || b.Status == "starting" {
			backendStates = append(backendStates, BackendState{
				ModelName: b.ModelName,
				PID:       b.PID,
				Port:      b.Port,
				StartedAt: b.StartedAt,
			})
		}
	}

	state := &ProxyState{
		PID:       os.Getpid(),
		Host:      s.config.Host,
		Port:      s.config.Port,
		StartedAt: s.startedAt,
		Backends:  backendStates,
	}

	if err := SaveProxyState(state); err != nil {
		logs.Warn("Failed to persist proxy state", "error", err)
	}
}

// handleChatCompletions proxies chat completion requests to the appropriate backend
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.proxyToBackend(w, r, "/v1/chat/completions")
}

// handleCompletions proxies completion requests to the appropriate backend
func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.proxyToBackend(w, r, "/v1/completions")
}

// handleEmbeddings proxies embedding requests to the appropriate backend
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	s.proxyToBackend(w, r, "/v1/embeddings")
}

// handleAnthropicMessages proxies Anthropic Messages API requests
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	s.proxyToBackendAnthropic(w, r, "/v1/messages")
}

// handleAnthropicCountTokens proxies Anthropic token counting requests
func (s *Server) handleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	s.proxyToBackendAnthropic(w, r, "/v1/messages/count_tokens")
}

// proxyToBackend handles the common logic of extracting model and proxying
func (s *Server) proxyToBackend(w http.ResponseWriter, r *http.Request, path string) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is allowed")
		return
	}

	// Cap inbound body to prevent unbounded memory use.
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			s.writeError(w, http.StatusRequestEntityTooLarge, "invalid_request",
				fmt.Sprintf("Request body exceeds %d bytes", maxRequestBodyBytes))
			return
		}
		s.writeError(w, http.StatusBadRequest, "invalid_request", "Failed to read request body")
		return
	}
	r.Body.Close()

	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid_request", "Failed to parse request body")
		return
	}

	if req.Model == "" {
		s.writeError(w, http.StatusBadRequest, "invalid_request", "Model field is required")
		return
	}

	// Get or load the backend (no options override for chat endpoint).
	// GetOrLoadBackend reserves the backend via AcquireRequest; we release on return.
	backend, err := s.manager.GetOrLoadBackend(req.Model, nil)
	if err != nil {
		s.handleModelError(w, err)
		return
	}
	defer backend.ReleaseRequest()

	// Proxy the request
	backendURL := fmt.Sprintf("http://%s:%d", s.config.Host, backend.Port)
	target, err := url.Parse(backendURL)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, "server_error", "invalid backend URL")
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)

	// Handle streaming responses properly
	proxy.FlushInterval = -1 // Flush immediately for SSE

	proxy.ModifyResponse = stripCORSHeaders

	// Restore the body for the proxied request
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.URL.Path = path

	proxy.ServeHTTP(w, r)
}

// proxyToBackendAnthropic handles Anthropic API requests with proper error format
func (s *Server) proxyToBackendAnthropic(w http.ResponseWriter, r *http.Request, path string) {
	requestID := generateRequestID()

	if r.Method != http.MethodPost {
		s.writeAnthropicError(w, requestID, http.StatusMethodNotAllowed, AnthropicInvalidRequest, "Only POST is allowed")
		return
	}

	// Cap inbound body to prevent unbounded memory use.
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			s.writeAnthropicError(w, requestID, http.StatusRequestEntityTooLarge, AnthropicInvalidRequest,
				fmt.Sprintf("Request body exceeds %d bytes", maxRequestBodyBytes))
			return
		}
		s.writeAnthropicError(w, requestID, http.StatusBadRequest, AnthropicInvalidRequest, "Failed to read request body")
		return
	}
	r.Body.Close()

	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		s.writeAnthropicError(w, requestID, http.StatusBadRequest, AnthropicInvalidRequest, "Failed to parse request body as JSON")
		return
	}

	if req.Model == "" {
		s.writeAnthropicError(w, requestID, http.StatusBadRequest, AnthropicInvalidRequest, "model: Field required")
		return
	}

	// Get or load the backend.
	// GetOrLoadBackend reserves the backend via AcquireRequest; we release on return.
	backend, err := s.manager.GetOrLoadBackend(req.Model, nil)
	if err != nil {
		s.handleAnthropicModelError(w, requestID, err)
		return
	}
	defer backend.ReleaseRequest()

	// Proxy the request
	backendURL := fmt.Sprintf("http://%s:%d", s.config.Host, backend.Port)
	target, err := url.Parse(backendURL)
	if err != nil {
		s.writeAnthropicError(w, requestID, http.StatusInternalServerError, AnthropicAPIError, "Internal server error")
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)

	// Handle streaming responses properly
	proxy.FlushInterval = -1 // Flush immediately for SSE

	proxy.ModifyResponse = func(resp *http.Response) error {
		resp.Header.Set("request-id", requestID)
		return stripCORSHeaders(resp)
	}

	// Handle backend errors
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		s.writeAnthropicError(w, requestID, http.StatusBadGateway, AnthropicAPIError, "Backend server error: "+err.Error())
	}

	// Restore the body for the proxied request
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.URL.Path = path

	// Strip Anthropic auth headers before forwarding (local server doesn't need them)
	r.Header.Del("x-api-key")

	proxy.ServeHTTP(w, r)
}

// generateRequestID creates a unique request ID in Anthropic format
func generateRequestID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "req_" + hex.EncodeToString(b)
}

// writeAnthropicError writes an Anthropic-compatible error response
func (s *Server) writeAnthropicError(w http.ResponseWriter, requestID string, status int, errType AnthropicErrorType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("request-id", requestID)
	w.WriteHeader(status)
	writeJSON(w, AnthropicError{
		Type: "error",
		Error: AnthropicErrorDetail{
			Type:    errType,
			Message: message,
		},
		RequestID: requestID,
	})
}

// handleAnthropicModelError converts model errors to Anthropic-formatted HTTP responses
func (s *Server) handleAnthropicModelError(w http.ResponseWriter, requestID string, err error) {
	switch e := err.(type) {
	case *AmbiguousModelError:
		msg := fmt.Sprintf("Ambiguous model name '%s'. Matches: %s",
			e.Query, strings.Join(e.Matches, ", "))
		s.writeAnthropicError(w, requestID, http.StatusBadRequest, AnthropicInvalidRequest, msg)
	case *ModelNotFoundError:
		msg := fmt.Sprintf("No downloaded model matches '%s'", e.Query)
		if len(e.Suggestions) > 0 {
			msg += fmt.Sprintf(". Did you mean: %s", strings.Join(e.Suggestions, ", "))
		}
		s.writeAnthropicError(w, requestID, http.StatusNotFound, AnthropicNotFound, msg)
	default:
		s.writeAnthropicError(w, requestID, http.StatusInternalServerError, AnthropicAPIError, err.Error())
	}
}

// handleModels returns the list of loaded models
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is allowed")
		return
	}

	backends := s.manager.ListBackends()

	var models []OpenAIModelInfo
	for _, b := range backends {
		models = append(models, OpenAIModelInfo{
			ID:      b.ModelName,
			Object:  "model",
			Created: b.StartedAt.Unix(),
			OwnedBy: "local",
			Lleme: &LlemeStatus{
				Status:       b.Status,
				Port:         b.Port,
				LastActivity: b.LastActivity,
				LoadedAt:     b.StartedAt,
			},
		})
	}

	// Also include downloaded but not loaded models
	downloaded, _ := s.manager.Resolver().ListDownloadedModels()
	loadedSet := make(map[string]bool)
	for _, b := range backends {
		loadedSet[b.ModelName] = true
	}
	for _, d := range downloaded {
		if !loadedSet[d.FullName] {
			models = append(models, OpenAIModelInfo{
				ID:      d.FullName,
				Object:  "model",
				Created: 0,
				OwnedBy: "local",
			})
		}
	}

	resp := OpenAIModelsResponse{
		Object: "list",
		Data:   models,
	}

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, resp)
}

// handleHealth returns basic health status
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, map[string]string{"status": "ok"})
}

// handleStatus returns detailed proxy status
func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is allowed")
		return
	}

	backends := s.manager.ListBackends()

	status := ProxyStatus{
		Version:       version.Version,
		UptimeSeconds: time.Since(s.startedAt).Seconds(),
		Host:          s.config.Host,
		Port:          s.config.Port,
		MaxModels:     s.config.MaxModels,
		LoadedCount:   len(backends),
		IdleTimeout:   s.config.IdleTimeout.String(),
		Models:        backends,
	}

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, status)
}

// handleModelError converts model errors to appropriate HTTP responses
func (s *Server) handleModelError(w http.ResponseWriter, err error) {
	switch e := err.(type) {
	case *AmbiguousModelError:
		msg := fmt.Sprintf("Ambiguous model name '%s'. Matches: %s",
			e.Query, strings.Join(e.Matches, ", "))
		s.writeError(w, http.StatusBadRequest, "invalid_request", msg)
	case *ModelNotFoundError:
		msg := fmt.Sprintf("No downloaded model matches '%s'", e.Query)
		if len(e.Suggestions) > 0 {
			msg += fmt.Sprintf(". Did you mean: %s", strings.Join(e.Suggestions, ", "))
		}
		s.writeError(w, http.StatusNotFound, "not_found", msg)
	default:
		s.writeError(w, http.StatusInternalServerError, "server_error", err.Error())
	}
}

// writeJSON encodes v as JSON to w. Errors are logged but not returned
// since callers are HTTP handlers where recovery is not possible.
func writeJSON(w http.ResponseWriter, v any) {
	if err := json.NewEncoder(w).Encode(v); err != nil {
		logs.Debug("failed to encode JSON response", "error", err)
	}
}

// writeError writes an OpenAI-compatible error response
func (s *Server) writeError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	writeJSON(w, OpenAIError{
		Error: OpenAIErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}

// decodeJSONBody reads r.Body under the shared size cap and decodes it into v.
// Returns a formatted OpenAI-style error response on failure and false; callers
// can simply `return` when false is returned.
func (s *Server) decodeJSONBody(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			s.writeError(w, http.StatusRequestEntityTooLarge, "invalid_request",
				fmt.Sprintf("Request body exceeds %d bytes", maxRequestBodyBytes))
			return false
		}
		s.writeError(w, http.StatusBadRequest, "invalid_request", "Failed to parse request body")
		return false
	}
	return true
}

// handleRun loads a model with optional server options
func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is allowed")
		return
	}

	var req RunRequest
	if !s.decodeJSONBody(w, r, &req) {
		return
	}

	if req.Model == "" {
		s.writeError(w, http.StatusBadRequest, "invalid_request", "Model field is required")
		return
	}

	// Build options map: start with additional options, then explicit fields override
	options := make(map[string]any)
	// First, copy additional options (e.g., from persona)
	// Normalize keys to hyphens (llama-server CLI format)
	for k, v := range req.Options {
		normalizedKey := strings.ReplaceAll(k, "_", "-")
		options[normalizedKey] = v
	}
	// Explicit fields override additional options (CLI flags > persona options)
	if req.CtxSize != nil {
		options["ctx-size"] = *req.CtxSize
	}
	if req.GpuLayers != nil {
		options["gpu-layers"] = *req.GpuLayers
	}
	if req.Threads != nil {
		options["threads"] = *req.Threads
	}

	// Load the backend with options.
	// GetOrLoadBackend reserves the backend; release it immediately since this
	// endpoint only reports status and does not proxy a request.
	backend, err := s.manager.GetOrLoadBackend(req.Model, options)
	if err != nil {
		s.handleModelError(w, err)
		return
	}
	defer backend.ReleaseRequest()

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, RunResponse{
		Success: true,
		Model:   backend.ModelName,
		Status:  backend.GetStatus().String(),
		Port:    backend.Port,
	})
}

// handleStopModel handles requests to unload a specific model
func (s *Server) handleStopModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is allowed")
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if !s.decodeJSONBody(w, r, &req) {
		return
	}

	if req.Model == "" {
		s.writeError(w, http.StatusBadRequest, "invalid_request", "Model field is required")
		return
	}

	// Resolve the model name
	result, err := s.manager.Resolver().Resolve(req.Model)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	if result.Model == nil {
		s.writeError(w, http.StatusNotFound, "not_found", fmt.Sprintf("Model '%s' is not loaded", req.Model))
		return
	}

	modelName := result.Model.FullName

	// Check if it's actually loaded
	if backend := s.manager.GetBackend(modelName); backend == nil {
		s.writeError(w, http.StatusNotFound, "not_found", fmt.Sprintf("Model '%s' is not loaded", req.Model))
		return
	}

	// Stop the backend
	if err := s.manager.StopBackend(modelName); err != nil {
		s.writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, map[string]any{
		"success": true,
		"model":   modelName,
	})
}

// handleStopAll handles requests to unload all models
func (s *Server) handleStopAll(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is allowed")
		return
	}

	// Cap any incoming body even though we ignore it, to prevent a large POST
	// from buffering into memory.
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)

	count := s.manager.LoadedCount()
	if err := s.manager.StopAllBackends(); err != nil {
		s.writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, map[string]any{
		"success": true,
		"stopped": count,
	})
}
