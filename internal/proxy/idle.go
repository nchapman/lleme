package proxy

import (
	"errors"
	"time"

	"github.com/nchapman/lleme/internal/logs"
)

// IdleMonitor periodically checks for and shuts down idle backends
type IdleMonitor struct {
	manager       *ModelManager
	idleTimeout   time.Duration
	checkInterval time.Duration
	stopChan      chan struct{}
	stoppedChan   chan struct{}
}

// NewIdleMonitor creates a new idle monitor
func NewIdleMonitor(manager *ModelManager, idleTimeout, checkInterval time.Duration) *IdleMonitor {
	return &IdleMonitor{
		manager:       manager,
		idleTimeout:   idleTimeout,
		checkInterval: checkInterval,
		stopChan:      make(chan struct{}),
		stoppedChan:   make(chan struct{}),
	}
}

// Start begins the idle monitoring loop
func (m *IdleMonitor) Start() {
	go m.run()
}

// Stop stops the idle monitor
func (m *IdleMonitor) Stop() {
	close(m.stopChan)
	<-m.stoppedChan
}

func (m *IdleMonitor) run() {
	defer close(m.stoppedChan)

	ticker := time.NewTicker(m.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.checkAndEvict()
		}
	}
}

func (m *IdleMonitor) checkAndEvict() {
	idleBackends := m.manager.GetIdleBackends(m.idleTimeout)

	for _, backend := range idleBackends {
		modelName := backend.ModelName
		idleDuration := backend.IdleDuration()

		// StopIfIdle re-checks the in-flight counter under m.mu so a request
		// that arrived between GetIdleBackends and here isn't torn down.
		err := m.manager.StopIfIdle(modelName)
		switch {
		case err == nil:
			logs.Info("Unloaded idle model", "model", modelName, "idle", idleDuration.Round(time.Second))
		case errors.Is(err, ErrBackendBusy):
			logs.Debug("Skipping idle unload; request arrived", "model", modelName)
		default:
			logs.Warn("Failed to unload model", "model", modelName, "error", err)
		}
	}
}
