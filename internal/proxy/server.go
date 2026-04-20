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
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/logs"
	"github.com/nchapman/lleme/internal/proxy/normalize"
	"github.com/nchapman/lleme/internal/swiftlm"
	"github.com/nchapman/lleme/internal/version"
)

// maxRequestBodyBytes caps inbound chat/messages request bodies to protect against
// unbounded memory use. 10 MiB is far larger than any reasonable prompt + history.
const maxRequestBodyBytes = 10 * 1024 * 1024

// Server is the main proxy server that routes requests to backends
type Server struct {
	httpServer       *http.Server
	manager          *ModelManager
	idleMonitor      *IdleMonitor
	config           *Config
	appConfig        *config.Config
	startedAt        time.Time
	shutdownChan     chan struct{}
	autoUpdateCancel context.CancelFunc
	stateMu          sync.Mutex // protects state file writes
}

// NewServer creates a new proxy server
func NewServer(cfg *Config, appCfg *config.Config) *Server {
	// Clean up any orphaned backends from a previous crash
	CleanupOrphanedBackends()

	manager := NewModelManager(cfg, appCfg)

	s := &Server{
		manager:      manager,
		config:       cfg,
		appConfig:    appCfg,
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

	// Check for backend updates in the background. Running backends keep using
	// their currently-loaded binary; new versions are picked up on next model load.
	ctx, cancel := context.WithCancel(context.Background())
	s.autoUpdateCancel = cancel
	go s.autoUpdateLlamaCpp(ctx)
	go s.autoUpdateSwiftLM(ctx)

	return nil
}

// autoUpdateCheck bundles what one backend's NewerVersionAvailable +
// identifiers resolve to before runBackendAutoUpdate drives the flow.
type autoUpdateCheck struct {
	name         string
	enabled      bool
	installedTag string
	latestTag    string
	haveLatest   bool
	checkErr     error
	install      func(context.Context) (string, error) // returns newly-installed tag
}

// runBackendAutoUpdate drives the check → install → log flow for one
// backend. Failures never affect the proxy; the context is canceled on Stop
// so an in-flight download aborts rather than swapping a symlink after
// shutdown. Kept separate from the two per-backend constructors so the
// shared logic isn't flagged as a duplicate.
func runBackendAutoUpdate(ctx context.Context, c autoUpdateCheck) {
	if !c.enabled {
		return
	}
	if c.checkErr != nil {
		logs.Debug(c.name+" auto-update check failed", "error", c.checkErr)
		return
	}
	if !c.haveLatest {
		return
	}
	logs.Info("Auto-updating "+c.name+" in background", "from", c.installedTag, "to", c.latestTag)

	newTag, err := c.install(ctx)
	if err != nil {
		if ctx.Err() != nil {
			logs.Debug(c.name+" auto-update canceled on shutdown", "error", err)
			return
		}
		logs.Warn(c.name+" auto-update failed", "error", err)
		return
	}
	logs.Info(c.name+" auto-update installed; new model loads will use this version", "version", newTag)
}

// autoUpdateLlamaCpp runs the llama.cpp auto-update check + install. The
// structural twin autoUpdateSwiftLM below duplicates the glue deliberately
// — collapsing both into a generic helper would lose per-backend types at
// package boundaries; the shared flow lives in runBackendAutoUpdate.
//
//nolint:dupl // backend-identity repetition; see doc comment above.
func (s *Server) autoUpdateLlamaCpp(ctx context.Context) {
	latest, installed, err := llama.NewerVersionAvailable()
	installedTag, latestTag := "", ""
	if installed != nil {
		installedTag = installed.TagName
	}
	if latest != nil {
		latestTag = latest.TagName
	}
	runBackendAutoUpdate(ctx, autoUpdateCheck{
		name:         "llama.cpp",
		enabled:      s.appConfig != nil && s.appConfig.LlamaCpp.AutoUpdateEnabled(),
		installedTag: installedTag,
		latestTag:    latestTag,
		haveLatest:   latest != nil,
		checkErr:     err,
		install: func(ctx context.Context) (string, error) {
			info, err := llama.InstallReleaseForAutoUpdate(ctx, latest, nil)
			if err != nil {
				return "", err
			}
			return info.TagName, nil
		},
	})
}

// autoUpdateSwiftLM runs the SwiftLM auto-update check + install. Gated on
// the platform via swiftlm.NewerVersionAvailable returning nil on hosts
// SwiftLM doesn't support. See autoUpdateLlamaCpp for the design note.
//
//nolint:dupl // backend-identity repetition; see autoUpdateLlamaCpp.
func (s *Server) autoUpdateSwiftLM(ctx context.Context) {
	latest, installed, err := swiftlm.NewerVersionAvailable()
	installedTag, latestTag := "", ""
	if installed != nil {
		installedTag = installed.TagName
	}
	if latest != nil {
		latestTag = latest.TagName
	}
	runBackendAutoUpdate(ctx, autoUpdateCheck{
		name:         "SwiftLM",
		enabled:      s.appConfig != nil && s.appConfig.SwiftLM.AutoUpdateEnabled(),
		installedTag: installedTag,
		latestTag:    latestTag,
		haveLatest:   latest != nil,
		checkErr:     err,
		install: func(ctx context.Context) (string, error) {
			info, err := swiftlm.InstallReleaseForAutoUpdate(ctx, latest, nil)
			if err != nil {
				return "", err
			}
			return info.TagName, nil
		},
	})
}

// Stop gracefully stops the proxy server
func (s *Server) Stop() error {
	close(s.shutdownChan)

	// Cancel any in-flight background auto-update so its download aborts.
	if s.autoUpdateCancel != nil {
		s.autoUpdateCancel()
	}

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

// handleChatCompletions proxies an OpenAI /v1/chat/completions request to
// the resolved backend, then runs the response through the normalize layer
// before returning it to the client. Mirrors handleAnthropicMessages so
// both surfaces share the same forward-and-rewrite shape — the reverse-
// proxy approach in proxyToBackend can't interpose on the response body,
// which is what normalization needs.
//
// /v1/completions and /v1/embeddings still use the reverse-proxy
// proxyToBackend helper: they have unrelated response shapes and don't
// need this normalization layer.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, ok := s.readOpenAIBody(w, r)
	if !ok {
		return
	}
	modelName, ok := s.extractOpenAIModel(w, body)
	if !ok {
		return
	}
	stream := requestWantsStream(body)

	backend, err := s.manager.GetOrLoadBackend(modelName, nil)
	if err != nil {
		s.handleModelError(w, err)
		return
	}
	defer backend.ReleaseRequest()

	resp, err := s.forwardToBackendChat(r.Context(), backend, body, stream)
	if err != nil {
		s.writeError(w, http.StatusBadGateway, "server_error", "Backend request failed: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Short-circuit: don't run the backend's error body through the
		// normalize layer (which expects a chat-completion shape and
		// would silently rewrite fields like model/system_fingerprint
		// onto the error envelope). Cap the body we echo back so a
		// misbehaving backend can't push arbitrary memory through.
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4*1024))
		msg := strings.TrimSpace(string(errBody))
		if msg == "" {
			msg = fmt.Sprintf("backend returned %d", resp.StatusCode)
		}
		s.writeError(w, resp.StatusCode, openAIErrorTypeForStatus(resp.StatusCode), msg)
		return
	}

	// Apply the same CORS strip the reverse-proxy ModifyResponse path uses,
	// so the proxy's own CORS middleware isn't shadowed by backend headers.
	_ = stripCORSHeaders(resp)

	wrapped := normalize.Wrap(resp.Body, normalize.Options{
		RequestedModel: modelName,
		Streaming:      stream,
		Fingerprint:    "lleme-" + version.Version,
	})

	if stream {
		writeNormalizedStream(w, resp, wrapped)
		return
	}
	s.writeNormalizedResponse(w, resp, wrapped)
}

// requestWantsStream reads only the "stream" field from a request body.
// Defaults to false (matches OpenAI's spec) when absent or malformed —
// callers already validated the body shape via extractOpenAIModel.
func requestWantsStream(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(body, &req)
	return req.Stream
}

// writeNormalizedStream copies SSE bytes from the normalize wrapper to
// the client, flushing after each read so single-token deltas reach the
// browser/SDK without buffering. Mirrors the SSE response shape from
// writeAnthropicStream — same headers, same flush cadence.
func writeNormalizedStream(w http.ResponseWriter, resp *http.Response, body io.Reader) {
	copyResponseHeaders(w, resp, true)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(resp.StatusCode)

	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 8*1024)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				logs.Debug("openai stream write failed", "error", werr)
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err == io.EOF {
			return
		}
		if err != nil {
			logs.Debug("openai stream read failed", "error", err)
			return
		}
	}
}

// writeNormalizedResponse drains the (already-normalized) body, sets
// Content-Length to the rewritten length, and writes the response. The
// upstream Content-Length is intentionally discarded — re-marshaling
// changes byte count.
func (s *Server) writeNormalizedResponse(w http.ResponseWriter, resp *http.Response, body io.Reader) {
	out, err := io.ReadAll(body)
	if err != nil {
		// Headers haven't been written yet — emit a proper OpenAI
		// error envelope so clients always see a parseable error
		// shape, never a plain-text http.Error body.
		status := http.StatusBadGateway
		if errors.Is(err, normalize.ErrResponseTooLarge) {
			status = http.StatusBadGateway
		}
		s.writeError(w, status, "server_error", "Failed to read backend response: "+err.Error())
		return
	}
	copyResponseHeaders(w, resp, false)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(out)))
	w.WriteHeader(resp.StatusCode)
	if _, werr := w.Write(out); werr != nil {
		logs.Debug("openai response write failed", "error", werr)
	}
}

// copyResponseHeaders forwards backend response headers to the client,
// dropping headers that we control or that no longer apply after
// rewriting. CORS is stripped earlier via stripCORSHeaders. We always
// drop Content-Length (recomputed for non-stream, irrelevant for SSE)
// and hop-by-hop headers per RFC 7230.
func copyResponseHeaders(w http.ResponseWriter, resp *http.Response, streaming bool) {
	for k, vs := range resp.Header {
		switch http.CanonicalHeaderKey(k) {
		case "Content-Length",
			"Transfer-Encoding",
			"Connection",
			"Keep-Alive",
			"Proxy-Authenticate",
			"Proxy-Authorization",
			"Te",
			"Trailer",
			"Upgrade":
			continue
		case "Content-Type", "Cache-Control":
			if streaming {
				// Streaming path sets these explicitly; skip backend
				// values to avoid double-set.
				continue
			}
		}
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
}

// handleCompletions proxies completion requests to the appropriate backend
func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.proxyToBackend(w, r, "/v1/completions")
}

// handleEmbeddings proxies embedding requests to the appropriate backend
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	s.proxyToBackend(w, r, "/v1/embeddings")
}

// handleAnthropicMessages translates an Anthropic /v1/messages request into
// an OpenAI chat completion, forwards it to the selected backend, and
// translates the response back. Translation happens in the proxy so backends
// (llama-server, SwiftLM) only need to speak OpenAI.
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	requestID := generateRequestID()

	body, ok := s.readAnthropicBody(w, r, requestID)
	if !ok {
		return
	}

	modelName, ok := s.extractAnthropicModel(w, requestID, body)
	if !ok {
		return
	}

	openAIBody, stream, err := translateAnthropicRequest(body)
	if err != nil {
		s.writeAnthropicTranslationError(w, requestID, err)
		return
	}

	backend, err := s.manager.GetOrLoadBackend(modelName, nil)
	if err != nil {
		s.handleAnthropicModelError(w, requestID, err)
		return
	}
	defer backend.ReleaseRequest()

	resp, err := s.forwardToBackendChat(r.Context(), backend, openAIBody, stream)
	if err != nil {
		s.writeAnthropicError(w, requestID, http.StatusBadGateway, AnthropicAPIError, "Backend request failed: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Cap the error body we buffer & echo back — a misbehaving backend
		// shouldn't cause the proxy to hold a multi-MB error string in memory
		// or leak arbitrarily large internals to the caller.
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4*1024))
		msg := strings.TrimSpace(string(errBody))
		if msg == "" {
			msg = fmt.Sprintf("backend returned %d", resp.StatusCode)
		}
		s.writeAnthropicError(w, requestID, resp.StatusCode, anthropicErrorTypeForStatus(resp.StatusCode), msg)
		return
	}

	// Pass the upstream body through the normalize layer before the
	// Anthropic translator sees it. SwiftLM's prefill_progress frames
	// would otherwise reach handleStreamLine as malformed-but-parseable
	// JSON and emit phantom Anthropic events; the model rewrite and
	// envelope synthesis are no-ops on the Anthropic surface but keep
	// both code paths consuming an identical OpenAI shape.
	wrapped := normalize.Wrap(resp.Body, normalize.Options{
		RequestedModel: modelName,
		Streaming:      stream,
		Fingerprint:    "lleme-" + version.Version,
	})

	messageID := generateMessageID()
	if stream {
		s.writeAnthropicStream(w, wrapped, requestID, messageID, modelName)
		return
	}
	s.writeAnthropicNonStream(w, wrapped, requestID, messageID)
}

// extractAnthropicModel pulls the model field out of an Anthropic request
// body and writes an error response if missing or malformed.
func (s *Server) extractAnthropicModel(w http.ResponseWriter, requestID string, body []byte) (string, bool) {
	model, err := parseModelField(body)
	if err != nil {
		s.writeAnthropicError(w, requestID, http.StatusBadRequest, AnthropicInvalidRequest, err.Error())
		return "", false
	}
	return model, true
}

// forwardToBackendChat POSTs a translated OpenAI request to the backend's
// /v1/chat/completions endpoint and returns the raw response.
func (s *Server) forwardToBackendChat(ctx context.Context, backend *Backend, openAIBody []byte, stream bool) (*http.Response, error) {
	backendURL := fmt.Sprintf("http://%s:%d/v1/chat/completions", s.config.Host, backend.Port)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, backendURL, bytes.NewReader(openAIBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	return backendChatClient.Do(req)
}

// writeAnthropicStream sets SSE headers and pumps the translated stream.
// Errors after the header has been written can't be surfaced to the client
// as HTTP status; log them for debugging.
func (s *Server) writeAnthropicStream(w http.ResponseWriter, body io.Reader, requestID, messageID, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("request-id", requestID)
	w.WriteHeader(http.StatusOK)

	if err := translateAnthropicStream(w, body, messageID, model); err != nil {
		logs.Debug("anthropic stream translation ended with error", "error", err)
	}
}

// writeAnthropicNonStream reads the full backend response, translates to
// Anthropic format, and writes it.
func (s *Server) writeAnthropicNonStream(w http.ResponseWriter, body io.Reader, requestID, messageID string) {
	respBody, err := io.ReadAll(body)
	if err != nil {
		s.writeAnthropicError(w, requestID, http.StatusBadGateway, AnthropicAPIError, "Failed to read backend response")
		return
	}
	translated, err := translateOpenAIResponse(respBody, messageID)
	if err != nil {
		s.writeAnthropicError(w, requestID, http.StatusInternalServerError, AnthropicAPIError, "Translation error: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("request-id", requestID)
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(translated); err != nil {
		logs.Debug("failed to write anthropic response", "error", err)
	}
}

// handleAnthropicCountTokens returns an approximate input-token count for an
// Anthropic /v1/messages/count_tokens request. Backends don't expose an
// OpenAI-equivalent tokenize endpoint uniformly, so we approximate on the
// proxy side. The estimate leans toward overcounting so callers plan
// conservatively; see countAnthropicTokens for the ratio.
func (s *Server) handleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	requestID := generateRequestID()

	body, ok := s.readAnthropicBody(w, r, requestID)
	if !ok {
		return
	}

	count, err := countAnthropicTokens(body)
	if err != nil {
		s.writeAnthropicTranslationError(w, requestID, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("request-id", requestID)
	w.WriteHeader(http.StatusOK)
	writeJSON(w, map[string]int{"input_tokens": count})
}

// backendChatClient forwards translated requests to a local backend. No
// timeout: streaming completions can run for minutes, and request-context
// cancelation already covers the disconnect case. Named to avoid reading
// like an Anthropic cloud client — the target is always a local backend.
var backendChatClient = &http.Client{}

// readAnthropicBody reads and size-caps the request body, writing an Anthropic
// error response on failure. Returns (body, true) on success, or (nil, false)
// after writing an error response.
func (s *Server) readAnthropicBody(w http.ResponseWriter, r *http.Request, requestID string) ([]byte, bool) {
	body, rbe := readBodyRaw(w, r)
	if rbe != nil {
		s.writeAnthropicError(w, requestID, rbe.status, AnthropicInvalidRequest, rbe.message)
		return nil, false
	}
	// Strip Anthropic client auth headers before any downstream forwarding.
	r.Header.Del("x-api-key")
	return body, true
}

// readBodyErr carries the status + message for a request-read failure
// without committing to a surface-specific error shape. Each handler
// (OpenAI vs Anthropic) translates it to its own JSON envelope.
type readBodyErr struct {
	status  int
	message string
}

// readBodyRaw enforces POST + size cap and drains the request body.
// Surface-agnostic: callers translate the returned readBodyErr into
// their own error response. Centralizing this avoids drift between
// the two surfaces' size-limit and method-validation behavior.
func readBodyRaw(w http.ResponseWriter, r *http.Request) ([]byte, *readBodyErr) {
	if r.Method != http.MethodPost {
		return nil, &readBodyErr{http.StatusMethodNotAllowed, "Only POST is allowed"}
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return nil, &readBodyErr{http.StatusRequestEntityTooLarge,
				fmt.Sprintf("Request body exceeds %d bytes", maxRequestBodyBytes)}
		}
		return nil, &readBodyErr{http.StatusBadRequest, "Failed to read request body"}
	}
	r.Body.Close()
	return body, nil
}

// writeAnthropicTranslationError maps a translation error to an Anthropic
// HTTP response. translateError carries the intended status and error type;
// other errors surface as 500 api_error.
func (s *Server) writeAnthropicTranslationError(w http.ResponseWriter, requestID string, err error) {
	var te *translateError
	if errors.As(err, &te) {
		s.writeAnthropicError(w, requestID, te.status, te.errType, te.msg)
		return
	}
	s.writeAnthropicError(w, requestID, http.StatusInternalServerError, AnthropicAPIError, err.Error())
}

// proxyToBackend handles the common logic of extracting model and proxying
// an OpenAI-style request verbatim to the resolved backend.
func (s *Server) proxyToBackend(w http.ResponseWriter, r *http.Request, path string) {
	body, ok := s.readOpenAIBody(w, r)
	if !ok {
		return
	}
	modelName, ok := s.extractOpenAIModel(w, body)
	if !ok {
		return
	}

	// Get or load the backend (no options override for chat endpoint).
	// GetOrLoadBackend reserves the backend via AcquireRequest; we release on return.
	backend, err := s.manager.GetOrLoadBackend(modelName, nil)
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

// readOpenAIBody reads and size-caps the request body for an OpenAI-style
// endpoint, writing an OpenAI-shaped error on failure. Returns (body, true)
// on success, (nil, false) after writing an error.
func (s *Server) readOpenAIBody(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	body, rbe := readBodyRaw(w, r)
	if rbe != nil {
		errType := "invalid_request"
		if rbe.status == http.StatusMethodNotAllowed {
			errType = "method_not_allowed"
		}
		s.writeError(w, rbe.status, errType, rbe.message)
		return nil, false
	}
	return body, true
}

// extractOpenAIModel pulls the model field out of an OpenAI-style request
// body, writing an OpenAI-shaped error on failure.
func (s *Server) extractOpenAIModel(w http.ResponseWriter, body []byte) (string, bool) {
	model, err := parseModelField(body)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return "", false
	}
	return model, true
}

// parseModelField extracts the "model" string from a request body. Used
// by both the OpenAI and Anthropic surfaces — the request shape is the
// same; only the error envelope differs, which the callers handle.
//
// Error messages are capitalized because they're user-facing API
// response messages (rendered into the error.message JSON field), not
// idiomatic Go error chains.
func parseModelField(body []byte) (string, error) {
	var ref struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &ref); err != nil {
		return "", errors.New("Failed to parse request body as JSON") //nolint:staticcheck // user-facing message
	}
	if ref.Model == "" {
		return "", errors.New("model: Field required") //nolint:staticcheck // user-facing message
	}
	return ref.Model, nil
}

// generateRequestID creates a unique request ID in Anthropic format
func generateRequestID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "req_" + hex.EncodeToString(b)
}

// generateMessageID creates an Anthropic-style msg_<random> ID. Kept
// independent from requestID so clients can't use one to derive the other
// — the two live in different namespaces (request is ephemeral; message is
// a resource identifier).
func generateMessageID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "msg_" + hex.EncodeToString(b)
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

// handleModels returns all downloaded models sorted by most recently used first.
// Loaded backends use their live LastActivity; unloaded models use the persisted
// last-used metadata, falling back to the GGUF file's mtime.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is allowed")
		return
	}

	type modelEntry struct {
		info     OpenAIModelInfo
		lastUsed time.Time
	}

	var entries []modelEntry
	loadedSet := make(map[string]bool)

	for _, b := range s.manager.ListBackends() {
		loadedSet[b.ModelName] = true
		entries = append(entries, modelEntry{
			info: OpenAIModelInfo{
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
			},
			lastUsed: b.LastActivity,
		})
	}

	downloaded, err := s.manager.Resolver().ListDownloadedModels()
	if err != nil {
		logs.Warn("Failed to enumerate downloaded models", "error", err)
	}
	for _, d := range downloaded {
		if loadedSet[d.FullName] {
			continue
		}
		lastUsed := hf.GetLastUsed(d.User, d.Repo, d.Quant)
		if lastUsed.IsZero() {
			if fi, err := os.Stat(d.ModelPath); err == nil {
				lastUsed = fi.ModTime()
			}
		}
		entries = append(entries, modelEntry{
			info: OpenAIModelInfo{
				ID:      d.FullName,
				Object:  "model",
				Created: 0,
				OwnedBy: "local",
			},
			lastUsed: lastUsed,
		})
	}

	sort.SliceStable(entries, func(i, j int) bool {
		return entries[i].lastUsed.After(entries[j].lastUsed)
	})

	models := make([]OpenAIModelInfo, len(entries))
	for i, e := range entries {
		models[i] = e.info
	}

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, OpenAIModelsResponse{Object: "list", Data: models})
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

// openAIErrorTypeForStatus maps an upstream HTTP status to the
// short error-type string this proxy uses across all OpenAI error
// responses (writeError call sites use "invalid_request",
// "not_found", "server_error", "method_not_allowed", etc. — kept
// for consistency over OpenAI's canonical "*_error" suffix forms).
// A future canonicalization pass should align both this function
// and the existing call sites in one go.
func openAIErrorTypeForStatus(status int) string {
	switch status {
	case http.StatusBadRequest,
		http.StatusRequestEntityTooLarge,
		http.StatusUnprocessableEntity: // llama-server uses 422 for context overflow
		return "invalid_request"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found"
	case http.StatusTooManyRequests:
		return "rate_limit"
	case http.StatusMethodNotAllowed:
		return "method_not_allowed"
	default:
		if status >= 400 && status < 500 {
			return "invalid_request"
		}
		return "server_error"
	}
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
