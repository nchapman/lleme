package proxy

// A Runtime implements the backend-specific parts of process lifecycle:
// which binary to run, how to build its CLI args, how to check readiness,
// and how to recognize fatal startup errors from its logs. ModelManager
// owns the generic parts (ports, logs, LRU, idle eviction, signal
// handling) and delegates the rest through this interface so llama-server
// and future backends (SwiftLM/MLX) can coexist.
type Runtime interface {
	Kind() BackendKind

	// HFAppName returns the HuggingFace "apps=" filter value that lists
	// models this backend can load (e.g. "llama.cpp" for GGUF,
	// "mlx-lm" for MLX). The search/trending UI joins the names of all
	// registered runtimes into one filter so adding a backend doesn't
	// require editing the discovery surface.
	HFAppName() string

	// BinaryPath returns the absolute path to the executable.
	BinaryPath() string

	// WorkingDir returns the directory to set as the process cwd. Some
	// backends locate sidecar files (templates, metallib) relative to the
	// binary and need a specific cwd.
	WorkingDir() string

	// BuildArgs returns the CLI arguments for starting this backend's
	// server against the given backend instance. host is the address the
	// backend should bind to; the port comes from backend.Port.
	BuildArgs(backend *Backend, host string) []string

	// HealthURL returns the URL the proxy polls to determine readiness.
	HealthURL(host string, port int) string

	// IsStartupError inspects the backend's log file and returns true when
	// a fatal error indicates the process will not become ready.
	IsStartupError(logPath string) bool
}

// BackendKind identifies a backend implementation.
type BackendKind string

const (
	BackendKindLlama BackendKind = "llama"
	BackendKindMLX   BackendKind = "mlx"
)
