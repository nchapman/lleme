package proxy

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/nchapman/lleme/internal/config"
)

// The lifecycle tests exercise the full backend machinery — process spawn,
// health polling, ready handoff, reuse, options reload, LRU eviction, idle
// unload, graceful stop — against a fake "llama-server". The fake is this
// test binary itself, re-exec'd with --fake-backend ADDR, which serves
// /health on the port the manager allocated for it.

const fakeBackendFlag = "--fake-backend"

func TestMain(m *testing.M) {
	for i, arg := range os.Args {
		if arg == fakeBackendFlag && i+1 < len(os.Args) {
			runFakeBackendProcess(os.Args[i+1])
			os.Exit(0)
		}
	}
	os.Exit(m.Run())
}

func runFakeBackendProcess(addr string) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	// Minimal OpenAI-compatible completion the proxy can pass through and
	// the Anthropic translator can convert back.
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl-fake","object":"chat.completion","created":1700000000,"model":"fake",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"hello from fake backend"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9}}`)
	})
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		// Manager treats an unbindable port as a startup failure.
		os.Exit(1)
	}
	// Blocks until the manager's SIGTERM kills us with the default
	// disposition (immediate exit), which cmd.Wait observes.
	_ = (&http.Server{Handler: mux}).Serve(ln)
}

// fakeRuntime is a Runtime whose "binary" is this test binary re-exec'd as
// the fake backend server.
type fakeRuntime struct{}

func (fakeRuntime) Kind() BackendKind          { return BackendKindLlama }
func (fakeRuntime) HFAppName() string          { return "llama.cpp" }
func (fakeRuntime) WorkingDir() string         { return "" }
func (fakeRuntime) IsStartupError(string) bool { return false }
func (fakeRuntime) SignificantOptions() []string {
	return []string{"ctx-size"}
}

func (fakeRuntime) BinaryPath() string {
	exe, err := os.Executable()
	if err != nil {
		panic(err)
	}
	return exe
}

func (fakeRuntime) BuildArgs(b *Backend, host string) []string {
	return []string{fakeBackendFlag, net.JoinHostPort(host, strconv.Itoa(b.Port))}
}

func (fakeRuntime) HealthURL(host string, port int) string {
	return fmt.Sprintf("http://%s:%d/health", host, port)
}

// writeFakeModel lays down a single-file GGUF model with gguf metadata under
// the given LLEME_HOME.
func writeFakeModel(t *testing.T, home, user, repo, quant string) {
	t.Helper()
	dir := fmt.Sprintf("%s/models/%s/%s", home, user, repo)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fmt.Sprintf("%s/%s.gguf", dir, quant), []byte("gguf"), 0644); err != nil {
		t.Fatal(err)
	}
	meta := fmt.Sprintf("quants:\n  %s:\n    backend: gguf\n", quant)
	if err := os.WriteFile(dir+"/metadata.yaml", []byte(meta), 0644); err != nil {
		t.Fatal(err)
	}
}

// newLifecycleManager builds a manager backed by the fake runtime with two
// local models (alpha, beta) under an isolated LLEME_HOME.
func newLifecycleManager(t *testing.T, maxModels int) *ModelManager {
	t.Helper()
	home := t.TempDir()
	t.Setenv("LLEME_HOME", home)
	writeFakeModel(t, home, "testuser", "alpha", "Q4_K_M")
	writeFakeModel(t, home, "testuser", "beta", "Q8_0")

	old := newLlamaRuntime
	newLlamaRuntime = func(*config.Config) Runtime { return fakeRuntime{} }
	t.Cleanup(func() { newLlamaRuntime = old })

	cfg := DefaultConfig()
	cfg.MaxModels = maxModels
	cfg.StartupTimeout = 15 * time.Second
	return NewModelManager(cfg, &config.Config{})
}

func mustLoad(t *testing.T, m *ModelManager, query string, options map[string]any) *Backend {
	t.Helper()
	b, err := m.GetOrLoadBackend(query, options)
	if err != nil {
		t.Fatalf("GetOrLoadBackend(%q): %v", query, err)
	}
	return b
}

func processAlive(pid int) bool {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

func TestBackendLoadAndReuse(t *testing.T) {
	m := newLifecycleManager(t, 3)
	t.Cleanup(func() { _ = m.StopAllBackends() })

	b1 := mustLoad(t, m, "testuser/alpha", nil)
	defer b1.ReleaseRequest()
	if b1.GetStatus() != BackendReady {
		t.Fatalf("status = %s, want ready", b1.GetStatus())
	}
	if b1.PID() == 0 {
		t.Error("PID not set after ready")
	}

	// Second load of the same model must reuse the running backend.
	b2 := mustLoad(t, m, "alpha:Q4_K_M", nil)
	b2.ReleaseRequest()
	if b2 != b1 {
		t.Error("second GetOrLoadBackend returned a different backend instance")
	}
	if m.LoadedCount() != 1 {
		t.Errorf("LoadedCount = %d, want 1", m.LoadedCount())
	}

	// The fake backend process must actually be serving health.
	resp, err := (&http.Client{Timeout: 2 * time.Second}).Get(b1.Runtime.HealthURL("127.0.0.1", b1.Port))
	if err != nil {
		t.Fatalf("backend health check: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("backend health = %d, want 200", resp.StatusCode)
	}
}

func TestBackendOptionsReload(t *testing.T) {
	m := newLifecycleManager(t, 3)
	t.Cleanup(func() { _ = m.StopAllBackends() })

	b1 := mustLoad(t, m, "testuser/alpha", map[string]any{"ctx-size": 100})
	oldPID := b1.PID()
	oldPort := b1.Port
	b1.ReleaseRequest()

	// ctx-size is a significant option for the fake runtime: loading again
	// with a different value must restart the backend process.
	b2 := mustLoad(t, m, "testuser/alpha", map[string]any{"ctx-size": 200})
	defer b2.ReleaseRequest()
	if b2.PID() == oldPID {
		t.Error("backend was not restarted for changed significant option")
	}
	_ = oldPort
	if m.LoadedCount() != 1 {
		t.Errorf("LoadedCount = %d, want 1 (replaced in place)", m.LoadedCount())
	}
	if processAlive(oldPID) {
		t.Error("old backend process still alive after options reload")
	}

	// Same significant value again must reuse, not restart.
	pid := b2.PID()
	b3 := mustLoad(t, m, "testuser/alpha", map[string]any{"ctx-size": 200})
	b3.ReleaseRequest()
	if b3.PID() != pid {
		t.Error("backend restarted although significant options were unchanged")
	}
}

func TestBackendLRUEviction(t *testing.T) {
	m := newLifecycleManager(t, 1)
	t.Cleanup(func() { _ = m.StopAllBackends() })

	alpha := mustLoad(t, m, "testuser/alpha", nil)
	alphaPID := alpha.PID()
	alpha.ReleaseRequest()

	// Slot limit is 1: loading beta must evict the idle alpha.
	beta := mustLoad(t, m, "testuser/beta", nil)
	defer beta.ReleaseRequest()

	if m.GetBackend("testuser/alpha:Q4_K_M") != nil {
		t.Error("alpha still loaded after LRU eviction")
	}
	if processAlive(alphaPID) {
		t.Error("evicted alpha process still alive")
	}
	if m.LoadedCount() != 1 {
		t.Errorf("LoadedCount = %d, want 1", m.LoadedCount())
	}
}

func TestBackendEvictionBlockedByInFlight(t *testing.T) {
	m := newLifecycleManager(t, 1)
	t.Cleanup(func() { _ = m.StopAllBackends() })

	alpha := mustLoad(t, m, "testuser/alpha", nil)
	defer alpha.ReleaseRequest() // hold the request slot for the whole test

	// All slots busy with an in-flight request: eviction must fail closed
	// rather than kill the backend mid-request.
	if _, err := m.GetOrLoadBackend("testuser/beta", nil); err == nil {
		t.Fatal("expected error when all model slots have in-flight requests")
	} else if !strings.Contains(err.Error(), "busy") {
		t.Errorf("err = %v, want slots-busy error", err)
	}
	if m.GetBackend("testuser/alpha:Q4_K_M") == nil {
		t.Error("alpha must stay loaded while its request is in flight")
	}
}

func TestBackendStopLifecycle(t *testing.T) {
	m := newLifecycleManager(t, 3)

	b := mustLoad(t, m, "testuser/alpha", nil)
	pid := b.PID()
	b.ReleaseRequest()

	if err := m.StopBackend("testuser/alpha:Q4_K_M"); err != nil {
		t.Fatalf("StopBackend: %v", err)
	}
	if processAlive(pid) {
		t.Error("backend process still alive after StopBackend")
	}
	if m.GetBackend("testuser/alpha:Q4_K_M") != nil {
		t.Error("backend still registered after StopBackend")
	}
	if m.LoadedCount() != 0 {
		t.Errorf("LoadedCount = %d, want 0", m.LoadedCount())
	}

	if err := m.StopBackend("testuser/alpha:Q4_K_M"); err == nil {
		t.Error("stopping an unknown backend must error")
	}
}

func TestStopIfIdle(t *testing.T) {
	m := newLifecycleManager(t, 3)

	b := mustLoad(t, m, "testuser/alpha", nil)

	// In-flight request pins the backend.
	if err := m.StopIfIdle("testuser/alpha:Q4_K_M"); err == nil {
		t.Fatal("StopIfIdle must refuse while a request is in flight")
	}
	b.ReleaseRequest()

	if err := m.StopIfIdle("testuser/alpha:Q4_K_M"); err != nil {
		t.Fatalf("StopIfIdle after release: %v", err)
	}
	if m.LoadedCount() != 0 {
		t.Errorf("LoadedCount = %d, want 0", m.LoadedCount())
	}
}

func TestIdleMonitorEvictsIdleBackend(t *testing.T) {
	m := newLifecycleManager(t, 3)

	b := mustLoad(t, m, "testuser/alpha", nil)
	b.ReleaseRequest()

	monitor := NewIdleMonitor(m, 50*time.Millisecond, 50*time.Millisecond)
	monitor.Start()
	defer monitor.Stop()

	deadline := time.Now().Add(5 * time.Second)
	for m.LoadedCount() != 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if m.LoadedCount() != 0 {
		t.Fatal("idle monitor did not unload the idle backend")
	}

	// A backend with an in-flight request must never be evicted by the monitor.
	busy := mustLoad(t, m, "testuser/beta", nil)
	defer busy.ReleaseRequest()
	time.Sleep(200 * time.Millisecond)
	if m.GetBackend("testuser/beta:Q8_0") == nil {
		t.Error("idle monitor evicted a backend with an in-flight request")
	}
}

func TestGetOrLoadBackendUnknownModel(t *testing.T) {
	m := newLifecycleManager(t, 3)

	_, err := m.GetOrLoadBackend("testuser/gamma", nil)
	if err == nil {
		t.Fatal("expected error for unknown model")
	}
	if _, ok := err.(*ModelNotFoundError); !ok {
		t.Errorf("err = %T (%v), want *ModelNotFoundError", err, err)
	}
}

func TestGetOrLoadBackendStartupFailure(t *testing.T) {
	home := t.TempDir()
	t.Setenv("LLEME_HOME", home)
	writeFakeModel(t, home, "testuser", "alpha", "Q4_K_M")

	// This runtime's child exits immediately (port 1 is unbindable), so the
	// manager must surface a startup failure rather than hang.
	old := newLlamaRuntime
	newLlamaRuntime = func(*config.Config) Runtime { return brokenRuntime{} }
	t.Cleanup(func() { newLlamaRuntime = old })

	cfg := DefaultConfig()
	cfg.StartupTimeout = 2 * time.Second
	m := NewModelManager(cfg, &config.Config{})

	if _, err := m.GetOrLoadBackend("testuser/alpha", nil); err == nil {
		t.Fatal("expected startup error for backend that never becomes healthy")
	}
	if m.LoadedCount() != 0 {
		t.Errorf("LoadedCount = %d, want 0 after failed start", m.LoadedCount())
	}
}

// brokenRuntime spawns a child that can never bind its port.
type brokenRuntime struct{ fakeRuntime }

func (brokenRuntime) BuildArgs(b *Backend, host string) []string {
	return []string{fakeBackendFlag, net.JoinHostPort(host, "1")}
}
