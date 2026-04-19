package proxy

import (
	"encoding/json"
	"os"
	"testing"
	"time"
)

// useTestHome sets LLEME_HOME to a temp directory for the duration of the test.
func useTestHome(t *testing.T) {
	t.Helper()
	t.Setenv("LLEME_HOME", t.TempDir())
}

func TestBackendStateSerialization(t *testing.T) {
	now := time.Now().Truncate(time.Second)
	state := BackendState{
		ModelName: "test/model:Q4_K_M",
		PID:       12345,
		Port:      49152,
		StartedAt: now,
	}

	data, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("failed to marshal BackendState: %v", err)
	}

	var decoded BackendState
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal BackendState: %v", err)
	}

	if decoded.ModelName != state.ModelName {
		t.Errorf("ModelName: expected %s, got %s", state.ModelName, decoded.ModelName)
	}
	if decoded.PID != state.PID {
		t.Errorf("PID: expected %d, got %d", state.PID, decoded.PID)
	}
	if decoded.Port != state.Port {
		t.Errorf("Port: expected %d, got %d", state.Port, decoded.Port)
	}
	if !decoded.StartedAt.Equal(state.StartedAt) {
		t.Errorf("StartedAt: expected %v, got %v", state.StartedAt, decoded.StartedAt)
	}
}

func TestProxyStateWithBackends(t *testing.T) {
	now := time.Now().Truncate(time.Second)
	state := ProxyState{
		PID:       1000,
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: now,
		Backends: []BackendState{
			{
				ModelName: "model1:Q4",
				PID:       2001,
				Port:      49152,
				StartedAt: now,
			},
			{
				ModelName: "model2:Q8",
				PID:       2002,
				Port:      49153,
				StartedAt: now,
			},
		},
	}

	data, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("failed to marshal ProxyState: %v", err)
	}

	var decoded ProxyState
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal ProxyState: %v", err)
	}

	if len(decoded.Backends) != 2 {
		t.Fatalf("expected 2 backends, got %d", len(decoded.Backends))
	}
	if decoded.Backends[0].ModelName != "model1:Q4" {
		t.Errorf("Backend[0].ModelName: expected model1:Q4, got %s", decoded.Backends[0].ModelName)
	}
	if decoded.Backends[1].PID != 2002 {
		t.Errorf("Backend[1].PID: expected 2002, got %d", decoded.Backends[1].PID)
	}
}

func TestProxyStateBackendsOmitEmpty(t *testing.T) {
	state := ProxyState{
		PID:       1000,
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: time.Now(),
		Backends:  nil,
	}

	data, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	// Verify "backends" field is omitted when nil
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("failed to unmarshal to map: %v", err)
	}

	if _, exists := m["backends"]; exists {
		t.Error("expected 'backends' field to be omitted when nil")
	}
}

func TestSaveLoadClearProxyState(t *testing.T) {
	useTestHome(t)

	now := time.Now().Truncate(time.Second)
	state := &ProxyState{
		PID:       os.Getpid(),
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: now,
		Backends: []BackendState{
			{
				ModelName: "test/model:Q4",
				PID:       12345,
				Port:      49152,
				StartedAt: now,
			},
		},
	}

	if err := SaveProxyState(state); err != nil {
		t.Fatalf("SaveProxyState failed: %v", err)
	}

	loaded, err := LoadProxyState()
	if err != nil {
		t.Fatalf("LoadProxyState failed: %v", err)
	}
	if loaded == nil {
		t.Fatal("LoadProxyState returned nil")
	}

	if loaded.PID != state.PID {
		t.Errorf("PID: expected %d, got %d", state.PID, loaded.PID)
	}
	if loaded.Host != state.Host {
		t.Errorf("Host: expected %s, got %s", state.Host, loaded.Host)
	}
	if loaded.Port != state.Port {
		t.Errorf("Port: expected %d, got %d", state.Port, loaded.Port)
	}
	if len(loaded.Backends) != 1 {
		t.Fatalf("expected 1 backend, got %d", len(loaded.Backends))
	}
	if loaded.Backends[0].ModelName != "test/model:Q4" {
		t.Errorf("Backend ModelName: expected test/model:Q4, got %s", loaded.Backends[0].ModelName)
	}

	// Test clear
	if err := ClearProxyState(); err != nil {
		t.Fatalf("ClearProxyState failed: %v", err)
	}

	loaded, err = LoadProxyState()
	if err != nil {
		t.Fatalf("LoadProxyState after clear failed: %v", err)
	}
	if loaded != nil {
		t.Error("expected nil after ClearProxyState")
	}
}

func TestLoadProxyStateNonExistent(t *testing.T) {
	useTestHome(t)

	state, err := LoadProxyState()
	if err != nil {
		t.Fatalf("LoadProxyState should not error on non-existent file: %v", err)
	}
	if state != nil {
		t.Error("expected nil state for non-existent file")
	}
}

func TestIsProcessRunning(t *testing.T) {
	// Current process should be running
	if !isProcessRunning(os.Getpid()) {
		t.Error("current process should be detected as running")
	}

	// Invalid PIDs should not be running
	if isProcessRunning(0) {
		t.Error("PID 0 should not be running")
	}
	if isProcessRunning(-1) {
		t.Error("PID -1 should not be running")
	}

	// Very high PID unlikely to exist
	if isProcessRunning(9999999) {
		t.Error("PID 9999999 should not be running")
	}
}

func TestCmdlineMatchesBackend(t *testing.T) {
	tests := []struct {
		name     string
		cmdline  string
		expected bool
	}{
		{
			name:     "llama-server in path",
			cmdline:  "/usr/bin/llama-server\x00--model\x00test.gguf",
			expected: true,
		},
		{
			name:     "llama_server with underscore",
			cmdline:  "/opt/llama_server\x00--port\x0049152",
			expected: true,
		},
		{
			name:     "SwiftLM via ps -o comm= (basename + newline)",
			cmdline:  "SwiftLM\n",
			expected: true,
		},
		{
			name:     "SwiftLM no trailing whitespace",
			cmdline:  "SwiftLM",
			expected: true,
		},
		{
			name:     "SwiftLM substring must not false-positive",
			cmdline:  "MySwiftLMWrapper\n",
			expected: false,
		},
		{
			name:     "different process",
			cmdline:  "/usr/bin/python\x00script.py",
			expected: false,
		},
		{
			name:     "empty cmdline",
			cmdline:  "",
			expected: false,
		},
		{
			name:     "llama-server as argument",
			cmdline:  "/bin/sh\x00-c\x00llama-server --model test",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cmdlineMatchesBackend(tt.cmdline)
			if result != tt.expected {
				t.Errorf("cmdlineMatchesBackend(%q) = %v, want %v", tt.cmdline, result, tt.expected)
			}
		})
	}
}

func TestCleanupOrphanedBackendsNoState(t *testing.T) {
	useTestHome(t)

	killed := CleanupOrphanedBackends()
	if killed != 0 {
		t.Errorf("expected 0 killed with no state, got %d", killed)
	}
}

func TestCleanupOrphanedBackendsProxyStillRunning(t *testing.T) {
	useTestHome(t)

	state := &ProxyState{
		PID:       os.Getpid(), // Current process - "running"
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: time.Now(),
		Backends: []BackendState{
			{
				ModelName: "test:Q4",
				PID:       99999, // Fake backend
				Port:      49152,
				StartedAt: time.Now(),
			},
		},
	}
	if err := SaveProxyState(state); err != nil {
		t.Fatalf("SaveProxyState failed: %v", err)
	}

	// Should not kill anything since proxy is "running"
	killed := CleanupOrphanedBackends()
	if killed != 0 {
		t.Errorf("expected 0 killed when proxy is running, got %d", killed)
	}

	// State should still exist
	loaded, _ := LoadProxyState()
	if loaded == nil {
		t.Error("state should still exist when proxy is running")
	}
}

func TestCleanupOrphanedBackendsStaleState(t *testing.T) {
	useTestHome(t)

	state := &ProxyState{
		PID:       9999999, // Non-existent process
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: time.Now(),
		Backends: []BackendState{
			{
				ModelName: "test:Q4",
				PID:       9999998, // Also non-existent
				Port:      49152,
				StartedAt: time.Now(),
			},
		},
	}
	if err := SaveProxyState(state); err != nil {
		t.Fatalf("SaveProxyState failed: %v", err)
	}

	// Should clean up stale state
	killed := CleanupOrphanedBackends()
	// Backend PID doesn't exist, so nothing to kill
	if killed != 0 {
		t.Errorf("expected 0 killed for non-existent PIDs, got %d", killed)
	}

	// State should be cleared
	loaded, _ := LoadProxyState()
	if loaded != nil {
		t.Error("stale state should be cleared")
	}
}

func TestGetRunningProxyState(t *testing.T) {
	useTestHome(t)

	// No state should return nil
	if state := GetRunningProxyState(); state != nil {
		t.Error("expected nil with no state")
	}

	// Save state with current process PID
	state := &ProxyState{
		PID:       os.Getpid(),
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: time.Now(),
	}
	if err := SaveProxyState(state); err != nil {
		t.Fatalf("SaveProxyState failed: %v", err)
	}

	// Should return state since process is running
	got := GetRunningProxyState()
	if got == nil {
		t.Fatal("expected non-nil state")
	}
	if got.PID != os.Getpid() {
		t.Errorf("PID: expected %d, got %d", os.Getpid(), got.PID)
	}
}

func TestGetRunningProxyStateStale(t *testing.T) {
	useTestHome(t)

	state := &ProxyState{
		PID:       9999999,
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: time.Now(),
	}
	if err := SaveProxyState(state); err != nil {
		t.Fatalf("SaveProxyState failed: %v", err)
	}

	// Should return nil for stale state
	got := GetRunningProxyState()
	if got != nil {
		t.Error("expected nil for stale state")
	}

	// State file should still exist - cleanup happens via CleanupOrphanedBackends
	loaded, _ := LoadProxyState()
	if loaded == nil {
		t.Error("state file should still exist (cleanup is done by CleanupOrphanedBackends)")
	}
}

func TestIsProxyRunning(t *testing.T) {
	useTestHome(t)

	// No state
	if IsProxyRunning() {
		t.Error("expected false with no state")
	}

	// With running process
	state := &ProxyState{
		PID:       os.Getpid(),
		Host:      "127.0.0.1",
		Port:      11313,
		StartedAt: time.Now(),
	}
	SaveProxyState(state)

	if !IsProxyRunning() {
		t.Error("expected true with running process")
	}
}
