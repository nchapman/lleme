package proxy

// Lock ordering (MUST be preserved):
//
//   ModelManager.mu  →  Backend.mu
//
// Any code that holds Backend.mu MUST NOT acquire ModelManager.mu. Reverse
// order would deadlock with GetIdleBackends, getLRUModel, GetOrLoadBackend,
// and StopBackend, all of which hold m.mu while calling Backend methods that
// take b.mu (InFlight, IdleDuration, GetStatus, SetStatus, AcquireRequest,
// etc.). When a function releases m.mu (e.g., shutdownBackend waiting on the
// process), it must not re-acquire it while holding b.mu.

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/logs"
)

// ModelManager manages the lifecycle of llama-server backend instances
type ModelManager struct {
	mu            sync.RWMutex
	backends      map[string]*Backend // model name -> backend
	lruOrder      []string            // for eviction ordering (front = most recent)
	portAllocator *PortAllocator
	resolver      *ModelResolver
	config        *Config
	appConfig     *config.Config
	onStateChange func() // called after backend start/stop to persist state
}

// NewModelManager creates a new model manager
func NewModelManager(cfg *Config, appCfg *config.Config) *ModelManager {
	return &ModelManager{
		backends:      make(map[string]*Backend),
		lruOrder:      make([]string, 0),
		portAllocator: NewPortAllocator(cfg.BackendPortMin, cfg.BackendPortMax),
		resolver:      NewModelResolver(),
		config:        cfg,
		appConfig:     appCfg,
	}
}

// SetStateChangeCallback sets a callback that is invoked after backends start or stop.
// This is used to persist backend state for crash recovery.
func (m *ModelManager) SetStateChangeCallback(fn func()) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onStateChange = fn
}

// GetOrLoadBackend returns a backend for the given model, loading it if necessary.
// Options override config defaults for this specific load (ctx-size, gpu-layers, etc.).
//
// On success, the returned backend has been reserved via AcquireRequest under m.mu,
// so idle-eviction and LRU-eviction cannot tear it down while the caller uses it.
// Callers MUST call backend.ReleaseRequest() when done (typically via defer).
func (m *ModelManager) GetOrLoadBackend(modelQuery string, options map[string]any) (*Backend, error) { //nolint:gocognit,cyclop // TODO: refactor after adding integration tests for locking behavior
	// First, resolve the model name
	result, err := m.resolver.Resolve(modelQuery)
	if err != nil {
		return nil, err
	}

	// Handle resolution errors
	if result.Model == nil {
		if len(result.Matches) > 1 {
			// Ambiguous match
			var names []string
			for _, match := range result.Matches {
				names = append(names, match.FullName)
			}
			return nil, &AmbiguousModelError{
				Query:   modelQuery,
				Matches: names,
			}
		}
		// No match
		var suggestions []string
		for _, s := range result.Suggestions {
			suggestions = append(suggestions, s.FullName)
		}
		return nil, &ModelNotFoundError{
			Query:       modelQuery,
			Suggestions: suggestions,
		}
	}

	modelName := result.Model.FullName
	modelPath := result.Model.ModelPath

	// Track model usage for cleanup purposes (non-critical)
	if err := hf.TouchLastUsed(result.Model.User, result.Model.Repo, result.Model.Quant); err != nil {
		logs.Debug("failed to update last used timestamp", "model", modelName, "error", err)
	}

	// Check if already loaded or loading
	m.mu.Lock()
	backend, exists := m.backends[modelName]
	if exists {
		switch status := backend.GetStatus(); status {
		case BackendReady:
			// Check if options changed - if so, reload the model
			if optionsChanged(backend.Options, options) {
				// Mark as stopping to prevent race conditions
				backend.SetStatus(BackendStopping)
				m.mu.Unlock()
				// Stop and let it fall through to reload with new options
				m.StopBackend(modelName)
				m.mu.Lock()
				// Fall through to create new backend with new options
			} else {
				// Same options - update LRU, reserve for this request, and return.
				// AcquireRequest runs while we still hold m.mu so any concurrent
				// getLRUModel or GetIdleBackends sees inFlight > 0 immediately.
				m.updateLRU(modelName)
				backend.AcquireRequest()
				m.mu.Unlock()
				return backend, nil
			}
		case BackendStarting:
			// Currently starting - wait for it, then reserve under m.mu.
			readyChan := backend.ReadyChan
			m.mu.Unlock()
			<-readyChan
			if backend.GetStatus() != BackendReady {
				return nil, fmt.Errorf("backend failed to start")
			}
			return m.finalizeReadyBackend(modelQuery, modelName, backend, options)
		}
	}

	// Need to start a new backend
	// Check if we need to evict
	if m.config.MaxModels > 0 && len(m.backends) >= m.config.MaxModels {
		lruModel := m.getLRUModel()
		if lruModel == "" {
			m.mu.Unlock()
			return nil, fmt.Errorf("cannot load %s: all %d model slots busy with in-flight requests", modelName, m.config.MaxModels)
		}
		// Mark as stopping to prevent concurrent eviction race
		if lruBackend := m.backends[lruModel]; lruBackend != nil {
			lruBackend.SetStatus(BackendStopping)
		}
		m.mu.Unlock()
		logs.Info("Evicting model to free slot", "model", lruModel)
		if err := m.StopBackend(lruModel); err != nil {
			return nil, fmt.Errorf("failed to evict model: %w", err)
		}
		m.mu.Lock()
	}

	// Allocate port
	port, err := m.portAllocator.Allocate()
	if err != nil {
		m.mu.Unlock()
		return nil, fmt.Errorf("failed to allocate port: %w", err)
	}

	// Create backend entry
	backend = &Backend{
		ModelName:    modelName,
		ModelPath:    modelPath,
		Runtime:      m.selectRuntime(modelPath, modelName),
		Port:         port,
		Status:       BackendStarting,
		StartedAt:    time.Now(),
		LastActivity: time.Now(),
		ReadyChan:    make(chan struct{}),
		Options:      options,
	}
	m.backends[modelName] = backend
	m.lruOrder = append([]string{modelName}, m.lruOrder...)
	callback := m.onStateChange
	m.mu.Unlock()

	// Persist "starting" state so we can clean up orphans if we crash
	if callback != nil {
		callback()
	}

	// Start the backend in background
	go m.startBackend(backend)

	// Wait for ready
	select {
	case <-backend.ReadyChan:
		if backend.GetStatus() != BackendReady {
			return nil, fmt.Errorf("backend failed to start")
		}
		return m.finalizeReadyBackend(modelQuery, modelName, backend, options)
	case <-time.After(m.config.StartupTimeout):
		m.StopBackend(modelName)
		return nil, fmt.Errorf("backend startup timeout after %v", m.config.StartupTimeout)
	}
}

// finalizeReadyBackend atomically verifies a ready backend is still the one
// registered under modelName, updates LRU, and reserves it for the caller's
// request. Returns a fresh GetOrLoadBackend call if options changed or the
// backend was replaced/evicted between startup and finalization.
func (m *ModelManager) finalizeReadyBackend(modelQuery, modelName string, backend *Backend, options map[string]any) (*Backend, error) {
	m.mu.Lock()
	current, ok := m.backends[modelName]
	if !ok || current != backend {
		// Evicted or replaced during handoff; retry from scratch.
		m.mu.Unlock()
		return m.GetOrLoadBackend(modelQuery, options)
	}
	if optionsChanged(backend.Options, options) {
		m.mu.Unlock()
		if err := m.StopBackend(modelName); err != nil {
			return nil, fmt.Errorf("reload with new options: %w", err)
		}
		return m.GetOrLoadBackend(modelQuery, options)
	}
	m.updateLRU(modelName)
	backend.AcquireRequest()
	m.mu.Unlock()
	return backend, nil
}

// GetBackend returns a backend if it exists and is ready
func (m *ModelManager) GetBackend(modelName string) *Backend {
	m.mu.RLock()
	defer m.mu.RUnlock()

	backend, exists := m.backends[modelName]
	if !exists {
		return nil
	}

	if backend.GetStatus() != BackendReady {
		return nil
	}

	return backend
}

// ListBackends returns info about all loaded backends
func (m *ModelManager) ListBackends() []BackendInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var infos []BackendInfo
	for _, backend := range m.backends {
		infos = append(infos, BackendInfo{
			ModelName:    backend.ModelName,
			Status:       backend.GetStatus().String(),
			Port:         backend.Port,
			PID:          backend.PID(),
			StartedAt:    backend.StartedAt,
			LastActivity: backend.GetLastActivity(),
			IdleMinutes:  backend.IdleDuration().Minutes(),
		})
	}
	return infos
}

// ErrBackendBusy is returned by StopIfIdle when the backend has in-flight requests.
var ErrBackendBusy = fmt.Errorf("backend busy with in-flight requests")

// StopIfIdle stops a backend only if it has no in-flight requests. Returns
// ErrBackendBusy otherwise. Used by the idle monitor to avoid tearing down
// a backend that a request acquired between the snapshot and the stop.
func (m *ModelManager) StopIfIdle(modelName string) error {
	m.mu.Lock()
	backend, exists := m.backends[modelName]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("backend not found: %s", modelName)
	}
	if backend.InFlight() > 0 {
		m.mu.Unlock()
		return ErrBackendBusy
	}
	backend.SetStatus(BackendStopping)
	m.mu.Unlock()

	return m.shutdownBackend(modelName, backend)
}

// StopBackend stops a specific backend unconditionally. Callers that must not
// kill an in-flight request (e.g., the idle monitor) should use StopIfIdle.
func (m *ModelManager) StopBackend(modelName string) error {
	m.mu.Lock()
	backend, exists := m.backends[modelName]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("backend not found: %s", modelName)
	}

	backend.SetStatus(BackendStopping)
	m.mu.Unlock()

	return m.shutdownBackend(modelName, backend)
}

// shutdownBackend performs graceful shutdown after a backend has been marked
// BackendStopping. Must be called without m.mu held.
func (m *ModelManager) shutdownBackend(modelName string, backend *Backend) error {
	// Graceful shutdown. Snapshot the process pointer under b.mu; after the
	// snapshot the os.Process methods are safe to call without holding the lock.
	if proc := backend.GetProcess(); proc != nil {
		proc.Signal(syscall.SIGTERM)

		// Wait for graceful exit (up to 5 seconds)
		done := make(chan struct{})
		go func() {
			proc.Wait()
			close(done)
		}()

		select {
		case <-done:
			// Process exited gracefully
		case <-time.After(5 * time.Second):
			// Force kill
			proc.Kill()
			proc.Wait()
		}
	}

	// Cleanup
	m.mu.Lock()
	backend.SetStatus(BackendStopped)
	backend.CloseReadyChan()
	if backend.LogWriter != nil {
		backend.LogWriter.Close()
	}
	m.portAllocator.Release(backend.Port)
	delete(m.backends, modelName)
	m.removeLRU(modelName)
	callback := m.onStateChange
	m.mu.Unlock()

	// Notify state change for persistence
	if callback != nil {
		callback()
	}

	return nil
}

// StopAllBackends stops all running backends
func (m *ModelManager) StopAllBackends() error {
	m.mu.RLock()
	names := make([]string, 0, len(m.backends))
	for name := range m.backends {
		names = append(names, name)
	}
	m.mu.RUnlock()

	var lastErr error
	for _, name := range names {
		if err := m.StopBackend(name); err != nil {
			lastErr = err
		}
	}

	return lastErr
}

// LoadedCount returns the number of loaded models
func (m *ModelManager) LoadedCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.backends)
}

// GetIdleBackends returns backends that have been idle longer than the timeout.
// Backends with in-flight requests are skipped regardless of idle duration.
func (m *ModelManager) GetIdleBackends(timeout time.Duration) []*Backend {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var idle []*Backend
	for _, backend := range m.backends {
		if backend.GetStatus() != BackendReady {
			continue
		}
		if backend.InFlight() > 0 {
			continue
		}
		if backend.IdleDuration() > timeout {
			idle = append(idle, backend)
		}
	}
	return idle
}

// Resolver returns the model resolver
func (m *ModelManager) Resolver() *ModelResolver {
	return m.resolver
}

// startBackend starts the server process for a backend via its Runtime.
func (m *ModelManager) startBackend(backend *Backend) {
	defer func() {
		// Ensure ReadyChan is closed even on error
		if backend.GetStatus() != BackendReady {
			backend.CloseReadyChan()
		}
	}()

	rt := backend.Runtime
	serverPath := rt.BinaryPath()
	args := rt.BuildArgs(backend, m.config.Host)

	cmd := exec.Command(serverPath, args...)
	cmd.Env = os.Environ()
	cmd.Dir = rt.WorkingDir()

	// Create rotating log writer for this backend
	logWriter, err := logs.NewRotatingWriter(logs.BackendLogPath(backend.ModelName))
	if err != nil {
		backend.SetStatus(BackendStopped)
		return
	}
	backend.LogWriter = logWriter

	cmd.Stdout = logWriter
	cmd.Stderr = logWriter

	if err := cmd.Start(); err != nil {
		logWriter.Close()
		backend.SetStatus(BackendStopped)
		return
	}

	backend.SetProcess(cmd.Process)

	// Wait for server to be ready
	if err := m.waitForReady(backend); err != nil {
		backend.SetStatus(BackendStopped)
		cmd.Process.Kill()
		logWriter.Close()
		return
	}

	backend.SetStatus(BackendReady)
	backend.CloseReadyChan()

	logs.Info("Model loaded", "model", backend.ModelName, "port", backend.Port)

	// Notify state change for persistence
	m.mu.RLock()
	callback := m.onStateChange
	m.mu.RUnlock()
	if callback != nil {
		callback()
	}
}

func (m *ModelManager) waitForReady(backend *Backend) error {
	healthURL := backend.Runtime.HealthURL(m.config.Host, backend.Port)
	client := &http.Client{Timeout: 2 * time.Second}

	logPath := logs.BackendLogPath(backend.ModelName)
	deadline := time.Now().Add(m.config.StartupTimeout)

	for time.Now().Before(deadline) {
		// Try health check
		resp, err := client.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		// Check log for errors
		if backend.Runtime.IsStartupError(logPath) {
			return fmt.Errorf("server startup failed (check %s)", logPath)
		}

		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("server did not become ready within %v", m.config.StartupTimeout)
}

func (m *ModelManager) updateLRU(modelName string) {
	// Move to front
	m.removeLRU(modelName)
	m.lruOrder = append([]string{modelName}, m.lruOrder...)
}

func (m *ModelManager) removeLRU(modelName string) {
	for i, name := range m.lruOrder {
		if name == modelName {
			m.lruOrder = append(m.lruOrder[:i], m.lruOrder[i+1:]...)
			return
		}
	}
}

// getLRUModel returns the least recently used model name that is eligible for eviction.
// Backends with in-flight requests are skipped. Returns "" if no candidate is eligible.
// Caller must hold m.mu.
func (m *ModelManager) getLRUModel() string {
	for i := len(m.lruOrder) - 1; i >= 0; i-- {
		name := m.lruOrder[i]
		backend, ok := m.backends[name]
		if !ok {
			continue
		}
		if backend.InFlight() > 0 {
			continue
		}
		return name
	}
	return ""
}

// AmbiguousModelError is returned when a query matches multiple models
type AmbiguousModelError struct {
	Query   string
	Matches []string
}

func (e *AmbiguousModelError) Error() string {
	return fmt.Sprintf("ambiguous model name '%s': matches %v", e.Query, e.Matches)
}

// ModelNotFoundError is returned when no model matches the query
type ModelNotFoundError struct {
	Query       string
	Suggestions []string
}

func (e *ModelNotFoundError) Error() string {
	if len(e.Suggestions) > 0 {
		return fmt.Sprintf("no model matches '%s'. Did you mean: %v", e.Query, e.Suggestions)
	}
	return fmt.Sprintf("no model matches '%s'", e.Query)
}

// optionsChanged returns true if the new options differ from the current options.
// Only compares options that affect model loading (server options).
// Returns false if both are nil/empty, or if they have the same values.
func optionsChanged(current, new map[string]any) bool {
	// If new options are nil/empty, no change requested
	if len(new) == 0 {
		return false
	}

	// If current is nil but new has values, that's a change
	if len(current) == 0 {
		return true
	}

	// Compare the options that matter for model loading
	serverOptions := []string{"ctx-size", "gpu-layers", "threads", "batch-size", "ubatch-size", "flash-attn", "mlock", "cache-type-k", "cache-type-v"}

	for _, key := range serverOptions {
		newVal, newExists := new[key]
		curVal, curExists := current[key]

		if newExists != curExists {
			return true
		}
		if newExists && !optionValuesEqual(newVal, curVal) {
			return true
		}
	}

	return false
}

// optionValuesEqual compares two option values, handling type coercion.
// YAML parses numbers as float64, but CLI produces int, so we normalize for comparison.
func optionValuesEqual(a, b any) bool {
	// Try numeric comparison first (handles int vs float64)
	aNum, aIsNum := toFloat64(a)
	bNum, bIsNum := toFloat64(b)
	if aIsNum && bIsNum {
		return aNum == bNum
	}

	// Fall back to direct comparison for non-numeric types (strings, bools)
	return a == b
}

// toFloat64 converts numeric types to float64 for comparison.
func toFloat64(v any) (float64, bool) {
	switch val := v.(type) {
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case float64:
		return val, true
	case float32:
		return float64(val), true
	}
	return 0, false
}
