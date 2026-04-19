// Package logs provides log file management with rotation for lleme.
package logs

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/nchapman/lleme/internal/config"
)

const (
	// MaxRotations is the number of rotated files to keep (.log.1, .log.2)
	MaxRotations = 2
	// MaxFileSize is the maximum size of a log file before rotation (10MB)
	MaxFileSize = 10 * 1024 * 1024
)

// Pre-compiled regexes for model name sanitization
var (
	ggufSuffixRe      = regexp.MustCompile(`(?i)-gguf(:|$)`)
	unsafeCharsRe     = regexp.MustCompile(`[^a-z0-9._-]`)
	multipleHyphensRe = regexp.MustCompile(`-+`)
)

// SanitizeModelName converts a full model name to a safe filename.
// Example: "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M" -> "llama-3.2-3b-instruct-q4_k_m"
func SanitizeModelName(fullName string) string {
	// Extract just the model part after the last slash
	name := fullName
	if idx := strings.LastIndex(fullName, "/"); idx >= 0 {
		name = fullName[idx+1:]
	}

	// Remove -GGUF suffix at end of name or before colon (case insensitive)
	name = ggufSuffixRe.ReplaceAllString(name, "$1")

	// Replace colon with hyphen (for quant separator)
	name = strings.ReplaceAll(name, ":", "-")

	// Convert to lowercase
	name = strings.ToLower(name)

	// Replace any remaining unsafe characters with hyphens
	name = unsafeCharsRe.ReplaceAllString(name, "-")

	// Collapse multiple hyphens
	name = multipleHyphensRe.ReplaceAllString(name, "-")

	// Trim leading/trailing hyphens
	name = strings.Trim(name, "-")

	return name
}

// BackendLogPath returns the log file path for a backend with the given model name.
func BackendLogPath(modelName string) string {
	sanitized := SanitizeModelName(modelName)
	return filepath.Join(config.LogsPath(), sanitized+".log")
}

// ProxyLogPath returns the log file path for the proxy.
func ProxyLogPath() string {
	return filepath.Join(config.LogsPath(), "proxy.log")
}

// rotateLogs rotates log files: .log -> .log.1 -> .log.2
// Keeps MaxRotations backup files plus the current active log.
func rotateLogs(basePath string) error {
	// Delete the oldest rotated file
	oldestPath := fmt.Sprintf("%s.%d", basePath, MaxRotations)
	os.Remove(oldestPath)

	// Rotate existing files
	for i := MaxRotations; i >= 1; i-- {
		oldPath := basePath
		if i > 1 {
			oldPath = fmt.Sprintf("%s.%d", basePath, i-1)
		}
		newPath := fmt.Sprintf("%s.%d", basePath, i)

		// Rename if the old file exists
		if _, err := os.Stat(oldPath); err == nil {
			if err := os.Rename(oldPath, newPath); err != nil {
				return err
			}
		}
	}

	return nil
}

// RotatingWriter wraps a file and automatically rotates when size limit is exceeded.
type RotatingWriter struct {
	mu           sync.Mutex
	basePath     string
	file         *os.File
	bytesWritten int64
}

// NewRotatingWriter creates a new rotating writer for the given base path.
// It rotates any existing log file and opens a fresh one.
func NewRotatingWriter(basePath string) (*RotatingWriter, error) {
	// Ensure the logs directory exists
	dir := filepath.Dir(basePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	// Rotate existing logs
	if err := rotateLogs(basePath); err != nil {
		return nil, err
	}

	// Open a fresh log file
	file, err := os.OpenFile(basePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return nil, err
	}

	return &RotatingWriter{
		basePath: basePath,
		file:     file,
	}, nil
}

// Write writes data to the log file, rotating if necessary.
func (w *RotatingWriter) Write(p []byte) (n int, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Check if we need to rotate before writing
	if w.bytesWritten+int64(len(p)) > MaxFileSize {
		if err := w.rotateUnlocked(); err != nil {
			return 0, err
		}
	}

	n, err = w.file.Write(p)
	w.bytesWritten += int64(n)
	return n, err
}

// Close closes the underlying file.
func (w *RotatingWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file != nil {
		return w.file.Close()
	}
	return nil
}

// rotateUnlocked performs rotation without holding the lock.
// Caller must hold w.mu.
func (w *RotatingWriter) rotateUnlocked() error {
	// Close current file; a failure here is non-fatal (we still want to rotate
	// and reopen), but the error is worth logging to stderr so debugging isn't
	// silent.
	if w.file != nil {
		if err := w.file.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "logs: failed to close %s during rotation: %v\n", w.basePath, err)
		}
	}

	// Rotate files
	if err := rotateLogs(w.basePath); err != nil {
		return fmt.Errorf("rotate logs for %s: %w", w.basePath, err)
	}

	// Open a fresh log file
	file, err := os.OpenFile(w.basePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("open %s: %w", w.basePath, err)
	}

	w.file = file
	w.bytesWritten = 0
	return nil
}

// Path returns the base path of the log file.
func (w *RotatingWriter) Path() string {
	return w.basePath
}
