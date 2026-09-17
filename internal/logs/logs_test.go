package logs

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nchapman/lleme/internal/config"
)

func TestSanitizeModelName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
			expected: "llama-3.2-3b-instruct-q4_k_m",
		},
		{
			input:    "Qwen/Qwen2.5-7B-Instruct-GGUF:Q5_K_M",
			expected: "qwen2.5-7b-instruct-q5_k_m",
		},
		{
			input:    "simple-model",
			expected: "simple-model",
		},
		{
			input:    "Model-With-GGUF-Suffix-GGUF:Q4_K_M",
			expected: "model-with-gguf-suffix-q4_k_m",
		},
		{
			input:    "org/Model-Name-gguf:Q8_0",
			expected: "model-name-q8_0",
		},
		{
			input:    "UPPERCASE-MODEL",
			expected: "uppercase-model",
		},
		{
			input:    "model with spaces",
			expected: "model-with-spaces",
		},
		{
			input:    "model///multiple//slashes",
			expected: "slashes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := SanitizeModelName(tt.input)
			if result != tt.expected {
				t.Errorf("SanitizeModelName(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestRotateLogs(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "lleme-logs-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	basePath := filepath.Join(tmpDir, "test.log")

	// Create initial log file with content
	if err := os.WriteFile(basePath, []byte("original content"), 0644); err != nil {
		t.Fatalf("Failed to write initial file: %v", err)
	}

	// Rotate
	if err := rotateLogs(basePath); err != nil {
		t.Fatalf("rotateLogs failed: %v", err)
	}

	// Check that .log was moved to .log.1
	content, err := os.ReadFile(basePath + ".1")
	if err != nil {
		t.Fatalf("Failed to read rotated file: %v", err)
	}
	if string(content) != "original content" {
		t.Errorf("Rotated content = %q, want %q", string(content), "original content")
	}

	// Original should not exist
	if _, err := os.Stat(basePath); !os.IsNotExist(err) {
		t.Error("Original file should not exist after rotation")
	}
}

func TestRotateLogsMultiple(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "lleme-logs-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	basePath := filepath.Join(tmpDir, "test.log")

	// Create 3 generations of log files
	os.WriteFile(basePath+".2", []byte("oldest"), 0644)
	os.WriteFile(basePath+".1", []byte("older"), 0644)
	os.WriteFile(basePath, []byte("current"), 0644)

	// Rotate
	if err := rotateLogs(basePath); err != nil {
		t.Fatalf("rotateLogs failed: %v", err)
	}

	// .2 should have "older" (was .1)
	content2, _ := os.ReadFile(basePath + ".2")
	if string(content2) != "older" {
		t.Errorf(".log.2 content = %q, want %q", string(content2), "older")
	}

	// .1 should have "current" (was .log)
	content1, _ := os.ReadFile(basePath + ".1")
	if string(content1) != "current" {
		t.Errorf(".log.1 content = %q, want %q", string(content1), "current")
	}

	// Original should not exist
	if _, err := os.Stat(basePath); !os.IsNotExist(err) {
		t.Error("Original file should not exist after rotation")
	}

	// "oldest" should be gone (exceeded MaxRotations)
}

func TestRotatingWriter(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "lleme-logs-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	basePath := filepath.Join(tmpDir, "test.log")

	writer, err := NewRotatingWriter(basePath)
	if err != nil {
		t.Fatalf("NewRotatingWriter failed: %v", err)
	}

	// Write some data
	data := "Hello, World!\n"
	n, err := writer.Write([]byte(data))
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	if n != len(data) {
		t.Errorf("Write returned %d, want %d", n, len(data))
	}

	writer.Close()

	// Verify file content
	content, err := os.ReadFile(basePath)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}
	if string(content) != data {
		t.Errorf("File content = %q, want %q", string(content), data)
	}
}

func TestRotatingWriterSizeRotation(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "lleme-logs-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	basePath := filepath.Join(tmpDir, "test.log")

	writer, err := NewRotatingWriter(basePath)
	if err != nil {
		t.Fatalf("NewRotatingWriter failed: %v", err)
	}

	// Override internal bytesWritten to simulate near-limit
	writer.bytesWritten = MaxFileSize - 10

	// Write enough to trigger rotation
	data := strings.Repeat("X", 100)
	if _, err := writer.Write([]byte(data)); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	writer.Close()

	// .log.1 should exist (rotated file)
	if _, err := os.Stat(basePath + ".1"); os.IsNotExist(err) {
		t.Error("Rotated file .log.1 should exist after size-based rotation")
	}

	// Current file should have the new data
	content, _ := os.ReadFile(basePath)
	if string(content) != data {
		t.Errorf("Current file content = %q, want %q", string(content), data)
	}
}

func TestNewRotatingWriterRotatesExisting(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "lleme-logs-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	basePath := filepath.Join(tmpDir, "test.log")

	// Create existing log file
	os.WriteFile(basePath, []byte("previous session"), 0644)

	// Create new writer (should rotate)
	writer, err := NewRotatingWriter(basePath)
	if err != nil {
		t.Fatalf("NewRotatingWriter failed: %v", err)
	}
	writer.Write([]byte("new session"))
	writer.Close()

	// Previous content should be in .log.1
	content1, err := os.ReadFile(basePath + ".1")
	if err != nil {
		t.Fatalf("Failed to read rotated file: %v", err)
	}
	if string(content1) != "previous session" {
		t.Errorf(".log.1 content = %q, want %q", string(content1), "previous session")
	}

	// New content should be in .log
	content, _ := os.ReadFile(basePath)
	if string(content) != "new session" {
		t.Errorf(".log content = %q, want %q", string(content), "new session")
	}
}

func TestBackendLogPath(t *testing.T) {
	path := BackendLogPath("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
	expected := filepath.Join(config.LogsPath(), "llama-3.2-3b-instruct-q4_k_m.log")
	if path != expected {
		t.Errorf("BackendLogPath() = %q, want %q", path, expected)
	}
}

func TestProxyLogPath(t *testing.T) {
	path := ProxyLogPath()
	expected := filepath.Join(config.LogsPath(), "proxy.log")
	if path != expected {
		t.Errorf("ProxyLogPath() = %q, want %q", path, expected)
	}
}
