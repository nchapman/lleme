package proxy

import (
	"io"
	"os"
	"sync"
	"time"

	"github.com/nchapman/lleme/internal/config"
)

// BackendStatus represents the current state of a backend server
type BackendStatus int

const (
	BackendStarting BackendStatus = iota
	BackendReady
	BackendStopping
	BackendStopped
)

func (s BackendStatus) String() string {
	switch s {
	case BackendStarting:
		return "starting"
	case BackendReady:
		return "ready"
	case BackendStopping:
		return "stopping"
	case BackendStopped:
		return "stopped"
	default:
		return "unknown"
	}
}

// Backend represents a running server instance for a specific model.
// The serving process is produced by Runtime (llama-server, SwiftLM, ...).
type Backend struct {
	mu           sync.RWMutex
	ModelName    string         // Full model reference: "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
	ModelPath    string         // Absolute path to the model (file for GGUF, directory for MLX)
	Runtime      Runtime        // Strategy for starting/health-checking this backend (exposes Kind())
	Port         int            // Port this backend is listening on
	Process      *os.Process    // The backend server process
	LogWriter    io.WriteCloser // Log file writer for this backend
	LastActivity time.Time      // Last time a request was made to this backend
	StartedAt    time.Time      // When this backend was started
	Status       BackendStatus  // Current status
	ReadyChan    chan struct{}  // Closed when backend is ready (for request coalescing)
	readyOnce    sync.Once      // Ensures ReadyChan is closed exactly once
	Options      map[string]any // Runtime options passed at load time (override config)
	inFlight     int            // In-flight request count; prevents idle/LRU eviction while > 0
}

// CloseReadyChan safely closes the ReadyChan exactly once
func (b *Backend) CloseReadyChan() {
	b.readyOnce.Do(func() {
		close(b.ReadyChan)
	})
}

// UpdateActivity updates the last activity time for this backend
func (b *Backend) UpdateActivity() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.LastActivity = time.Now()
}

// GetLastActivity returns the last activity time
func (b *Backend) GetLastActivity() time.Time {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.LastActivity
}

// GetStatus returns the current status
func (b *Backend) GetStatus() BackendStatus {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.Status
}

// SetStatus updates the status
func (b *Backend) SetStatus(status BackendStatus) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Status = status
}

// IdleDuration returns how long the backend has been idle
func (b *Backend) IdleDuration() time.Duration {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return time.Since(b.LastActivity)
}

// AcquireRequest marks the start of an in-flight request. Paired with ReleaseRequest.
// Backends with in-flight requests are skipped by idle and LRU eviction.
func (b *Backend) AcquireRequest() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.inFlight++
	b.LastActivity = time.Now()
}

// ReleaseRequest marks the end of an in-flight request.
func (b *Backend) ReleaseRequest() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.inFlight > 0 {
		b.inFlight--
	}
	b.LastActivity = time.Now()
}

// InFlight returns the current number of in-flight requests.
func (b *Backend) InFlight() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.inFlight
}

// PID returns the backend process PID, or 0 if not yet started.
// Guarded by b.mu because Process is assigned and torn down concurrently.
func (b *Backend) PID() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.Process == nil {
		return 0
	}
	return b.Process.Pid
}

// GetProcess returns the underlying *os.Process pointer (or nil). The returned
// pointer's methods (Signal/Wait/Kill) are safe to call without b.mu because
// os.Process is internally goroutine-safe; the lock only guards the pointer
// itself against concurrent reassignment.
func (b *Backend) GetProcess() *os.Process {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.Process
}

// SetProcess assigns the backend's process under the lock.
func (b *Backend) SetProcess(p *os.Process) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Process = p
}

// Config holds proxy configuration
type Config struct {
	Host           string        // Proxy host (default: "127.0.0.1")
	Port           int           // Proxy port (default: 11313)
	MaxModels      int           // Maximum concurrent models (0 = unlimited)
	IdleTimeout    time.Duration // How long before idle models are unloaded
	BackendPortMin int           // Minimum port for backends
	BackendPortMax int           // Maximum port for backends
	StartupTimeout time.Duration // How long to wait for backend startup
	CORSOrigins    []string      // Allowed CORS origins (empty = local only)
}

// DefaultConfig returns the default proxy configuration
func DefaultConfig() *Config {
	return &Config{
		Host:           "127.0.0.1",
		Port:           11313,
		MaxModels:      3,
		IdleTimeout:    10 * time.Minute,
		BackendPortMin: 49152,
		BackendPortMax: 49200,
		StartupTimeout: 120 * time.Second,
	}
}

// ConfigFromAppConfig creates a proxy Config from the app config
func ConfigFromAppConfig(s config.Server) *Config {
	cfg := DefaultConfig()

	if s.Host != "" {
		cfg.Host = s.Host
	}
	if s.Port > 0 {
		cfg.Port = s.Port
	}
	if s.MaxModels > 0 {
		cfg.MaxModels = s.MaxModels
	}
	// Duration zero-value means "not configured" → keep the proxy default.
	// A user who writes `idle_timeout: 0s` will get the default too; "never
	// unload" is not supported via zero — use a very large duration instead.
	if s.IdleTimeout > 0 {
		cfg.IdleTimeout = s.IdleTimeout.AsDuration()
	}
	if s.BackendPortMin > 0 {
		cfg.BackendPortMin = s.BackendPortMin
	}
	if s.BackendPortMax > 0 {
		cfg.BackendPortMax = s.BackendPortMax
	}
	if s.StartupTimeout > 0 {
		cfg.StartupTimeout = s.StartupTimeout.AsDuration()
	}
	if len(s.CORSOrigins) > 0 {
		cfg.CORSOrigins = s.CORSOrigins
	}

	return cfg
}

// BackendInfo contains information about a backend for API responses
type BackendInfo struct {
	ModelName    string    `json:"name"`
	Status       string    `json:"status"`
	Port         int       `json:"port"`
	PID          int       `json:"pid"`
	StartedAt    time.Time `json:"started_at"`
	LastActivity time.Time `json:"last_activity"`
	IdleMinutes  float64   `json:"idle_minutes"`
}

// ProxyStatus contains the full proxy status for API responses
type ProxyStatus struct {
	Version       string        `json:"version"`
	UptimeSeconds float64       `json:"uptime_seconds"`
	Host          string        `json:"host"`
	Port          int           `json:"port"`
	MaxModels     int           `json:"max_models"`
	LoadedCount   int           `json:"loaded_count"`
	IdleTimeout   string        `json:"idle_timeout"`
	Models        []BackendInfo `json:"models"`
}

// OpenAIError represents an OpenAI-compatible error response
type OpenAIError struct {
	Error OpenAIErrorDetail `json:"error"`
}

// OpenAIErrorDetail contains the error details
type OpenAIErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// OpenAIModelsResponse represents the /v1/models response
type OpenAIModelsResponse struct {
	Object string            `json:"object"`
	Data   []OpenAIModelInfo `json:"data"`
}

// OpenAIModelInfo represents a single model in the models list
type OpenAIModelInfo struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	OwnedBy string       `json:"owned_by"`
	Lleme   *LlemeStatus `json:"lleme,omitempty"`
}

// LlemeStatus contains lleme-specific model status
type LlemeStatus struct {
	Status       string    `json:"status"`
	Port         int       `json:"port"`
	LastActivity time.Time `json:"last_activity"`
	LoadedAt     time.Time `json:"loaded_at"`
}

// RunRequest is the request body for POST /api/run
// Field names use underscores to match CLI flag names (with hyphens converted)
// Pointer types allow distinguishing "not set" from "explicitly zero"
type RunRequest struct {
	Model string `json:"model"`

	// Server options (passed to llama-server at load time)
	// Use pointers so 0 can be explicitly set (e.g., gpu_layers: 0 for CPU-only)
	CtxSize   *int `json:"ctx_size,omitempty"`
	GpuLayers *int `json:"gpu_layers,omitempty"`
	Threads   *int `json:"threads,omitempty"`

	// Additional llama-server options can be passed as a map
	Options map[string]any `json:"options,omitempty"`
}

// RunResponse is the response for POST /api/run
type RunResponse struct {
	Success bool   `json:"success"`
	Model   string `json:"model"`
	Status  string `json:"status"`
	Port    int    `json:"port"`
}

// Anthropic API error types
// See: https://docs.anthropic.com/en/api/errors

// AnthropicErrorType represents the type of Anthropic API error
type AnthropicErrorType string

const (
	AnthropicInvalidRequest  AnthropicErrorType = "invalid_request_error"
	AnthropicAuthentication  AnthropicErrorType = "authentication_error"
	AnthropicPermission      AnthropicErrorType = "permission_error"
	AnthropicNotFound        AnthropicErrorType = "not_found_error"
	AnthropicRequestTooLarge AnthropicErrorType = "request_too_large"
	AnthropicRateLimit       AnthropicErrorType = "rate_limit_error"
	AnthropicAPIError        AnthropicErrorType = "api_error"
	AnthropicOverloaded      AnthropicErrorType = "overloaded_error"
)

// AnthropicError represents the full Anthropic error response
type AnthropicError struct {
	Type      string               `json:"type"` // Always "error"
	Error     AnthropicErrorDetail `json:"error"`
	RequestID string               `json:"request_id,omitempty"`
}

// AnthropicErrorDetail contains the error details
type AnthropicErrorDetail struct {
	Type    AnthropicErrorType `json:"type"`
	Message string             `json:"message"`
}
