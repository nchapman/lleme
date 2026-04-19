package proxy

import (
	"errors"
	"testing"
	"time"
)

// newTestBackend builds a Backend with a known LastActivity.
// The status is set so GetStatus reflects the initial value.
func newTestBackend(name string, status BackendStatus, activityAgo time.Duration) *Backend {
	return &Backend{
		ModelName:    name,
		Port:         1,
		Status:       status,
		LastActivity: time.Now().Add(-activityAgo),
		ReadyChan:    make(chan struct{}),
	}
}

func TestBackendInFlightLifecycle(t *testing.T) {
	b := newTestBackend("m", BackendReady, 0)

	if got := b.InFlight(); got != 0 {
		t.Fatalf("initial InFlight() = %d, want 0", got)
	}

	b.AcquireRequest()
	if got := b.InFlight(); got != 1 {
		t.Errorf("after Acquire: InFlight() = %d, want 1", got)
	}

	b.AcquireRequest()
	if got := b.InFlight(); got != 2 {
		t.Errorf("after 2nd Acquire: InFlight() = %d, want 2", got)
	}

	b.ReleaseRequest()
	b.ReleaseRequest()
	if got := b.InFlight(); got != 0 {
		t.Errorf("after 2 Releases: InFlight() = %d, want 0", got)
	}

	// Defensive: extra Release should not drive the counter negative.
	b.ReleaseRequest()
	if got := b.InFlight(); got != 0 {
		t.Errorf("after extra Release: InFlight() = %d, want 0", got)
	}
}

func TestBackendAcquireRefreshesActivity(t *testing.T) {
	b := newTestBackend("m", BackendReady, 10*time.Minute)
	if idle := b.IdleDuration(); idle < 10*time.Minute {
		t.Fatalf("precondition: IdleDuration = %v, want >= 10m", idle)
	}

	b.AcquireRequest()
	if idle := b.IdleDuration(); idle > time.Second {
		t.Errorf("after Acquire: IdleDuration = %v, want near zero", idle)
	}
}

// newTestManagerWithBackends builds a ModelManager with the given backends
// already registered and ordered in lruOrder by insertion (first = most recent).
func newTestManagerWithBackends(backends ...*Backend) *ModelManager {
	m := &ModelManager{
		backends:      make(map[string]*Backend),
		lruOrder:      make([]string, 0, len(backends)),
		portAllocator: NewPortAllocator(49000, 49100),
		config:        &Config{MaxModels: len(backends)},
	}
	for _, b := range backends {
		m.backends[b.ModelName] = b
		m.lruOrder = append([]string{b.ModelName}, m.lruOrder...)
	}
	return m
}

func TestGetIdleBackendsSkipsInFlight(t *testing.T) {
	idle := newTestBackend("idle", BackendReady, 20*time.Minute)
	busy := newTestBackend("busy", BackendReady, 20*time.Minute)
	busy.AcquireRequest()

	m := newTestManagerWithBackends(idle, busy)

	got := m.GetIdleBackends(5 * time.Minute)
	if len(got) != 1 {
		t.Fatalf("GetIdleBackends returned %d backends, want 1", len(got))
	}
	if got[0].ModelName != "idle" {
		t.Errorf("GetIdleBackends returned %q, want idle", got[0].ModelName)
	}
}

func TestGetIdleBackendsSkipsNonReady(t *testing.T) {
	starting := newTestBackend("starting", BackendStarting, 20*time.Minute)
	ready := newTestBackend("ready", BackendReady, 20*time.Minute)

	m := newTestManagerWithBackends(starting, ready)

	got := m.GetIdleBackends(5 * time.Minute)
	if len(got) != 1 || got[0].ModelName != "ready" {
		t.Errorf("expected only 'ready' returned, got %+v", got)
	}
}

func TestGetLRUModelSkipsInFlight(t *testing.T) {
	// Helper prepends each arg, matching production's updateLRU semantics:
	// first arg ends up at the tail (LRU), last arg at the head (MRU).
	oldest := newTestBackend("oldest", BackendReady, 30*time.Minute)
	oldest.AcquireRequest() // busy - should be skipped
	middle := newTestBackend("middle", BackendReady, 20*time.Minute)
	newest := newTestBackend("newest", BackendReady, 1*time.Minute)

	m := newTestManagerWithBackends(oldest, middle, newest)

	got := m.getLRUModel()
	if got != "middle" {
		t.Errorf("getLRUModel = %q, want 'middle' (oldest is in-flight)", got)
	}
}

func TestGetLRUModelAllBusy(t *testing.T) {
	a := newTestBackend("a", BackendReady, 0)
	a.AcquireRequest()
	b := newTestBackend("b", BackendReady, 0)
	b.AcquireRequest()

	m := newTestManagerWithBackends(a, b)

	if got := m.getLRUModel(); got != "" {
		t.Errorf("getLRUModel with all in-flight = %q, want empty", got)
	}
}

// TestStopIfIdleRefusesBusyBackend verifies the idle monitor's path does not
// tear down a backend whose inFlight counter is > 0 — the race closed by
// moving AcquireRequest inside GetOrLoadBackend and having StopIfIdle
// re-check under m.mu.
func TestStopIfIdleRefusesBusyBackend(t *testing.T) {
	busy := newTestBackend("busy", BackendReady, 20*time.Minute)
	busy.AcquireRequest()

	m := newTestManagerWithBackends(busy)

	err := m.StopIfIdle("busy")
	if !errors.Is(err, ErrBackendBusy) {
		t.Errorf("StopIfIdle returned %v, want ErrBackendBusy", err)
	}

	// Backend must still be registered
	if _, ok := m.backends["busy"]; !ok {
		t.Error("busy backend was removed despite in-flight request")
	}
}
