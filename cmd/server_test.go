package cmd

import (
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"

	"github.com/nchapman/lleme/internal/proxy"
)

func TestStopServerNotRunning(t *testing.T) {
	// Use temp directory to isolate from real system state
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	// Use a port that's definitely not in use to avoid finding real servers
	oldPort := serverPort
	defer func() { serverPort = oldPort }()
	serverPort = 59999

	// With a temp HOME and unused port, no server should be found
	stopped, err := stopServer()
	if err != nil {
		t.Errorf("stopServer() error = %v, want nil", err)
	}
	if stopped {
		t.Error("stopServer() returned true when server was not running")
	}
}

// TestStopServerPIDReuse guards the PID-reuse hazard: a stale state file
// whose PID now belongs to an unrelated live process must not be signaled.
func TestStopServerPIDReuse(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep not available")
	}
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	victim := exec.Command("sleep", "10")
	if err := victim.Start(); err != nil {
		t.Fatalf("spawn victim: %v", err)
	}
	defer func() {
		_ = victim.Process.Kill()
		_ = victim.Wait()
	}()

	state := &proxy.ProxyState{
		PID:  victim.Process.Pid, // Live PID that is NOT a lleme process
		Host: "127.0.0.1",
		Port: 11313,
	}
	if err := proxy.SaveProxyState(state); err != nil {
		t.Fatalf("Failed to save state: %v", err)
	}

	stopped, err := stopServer()
	if err == nil || !strings.Contains(err.Error(), "stale state file") {
		t.Errorf("stopServer() err = %v, want stale state file rejection", err)
	}
	if stopped {
		t.Error("stopServer() must not report stopping a reused PID")
	}
	if proxy.GetRunningProxyState() != nil {
		t.Error("stopServer() should clear the stale state file")
	}
	// The innocent process must have survived.
	if err := victim.Process.Signal(syscall.Signal(0)); err != nil {
		t.Error("reused-PID victim was killed by stopServer()")
	}
}

func TestStopServerStaleState(t *testing.T) {
	// Create temp directory for state file
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	// Save state with a PID that doesn't exist
	state := &proxy.ProxyState{
		PID:  99999999, // Very unlikely to be a real process
		Host: "127.0.0.1",
		Port: 11313,
	}
	if err := proxy.SaveProxyState(state); err != nil {
		t.Fatalf("Failed to save state: %v", err)
	}

	// stopServer should handle this gracefully
	stopped, err := stopServer()

	// On Unix, FindProcess always succeeds, but Signal will fail
	// Either way, state should be cleared
	if proxy.GetRunningProxyState() != nil {
		t.Error("stopServer() should clear stale state")
	}

	// We expect either an error (signal failed) or stopped=true (process killed)
	// The important thing is it doesn't panic
	_ = stopped
	_ = err
}
