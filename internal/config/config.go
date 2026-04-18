package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// DefaultSystemPrompt returns the default system prompt for chat sessions.
func DefaultSystemPrompt() string {
	return "You are a helpful assistant."
}

type Config struct {
	HuggingFace HuggingFace `yaml:"huggingface"`
	Server      Server      `yaml:"server"`
	LlamaCpp    LlamaCpp    `yaml:"llamacpp"`
}

type HuggingFace struct {
	Token        string `yaml:"token"`
	DefaultQuant string `yaml:"default_quant"`
}

type LlamaCpp struct {
	ServerPath string         `yaml:"server_path,omitempty"`
	Options    map[string]any `yaml:"options,omitempty"`
}

type Server struct {
	Host            string   `yaml:"host"`
	Port            int      `yaml:"port"`
	MaxModels       int      `yaml:"max_models"`
	IdleTimeoutMins int      `yaml:"idle_timeout_mins"`
	StartupTimeoutS int      `yaml:"startup_timeout_secs"`
	BackendPortMin  int      `yaml:"backend_port_min"`
	BackendPortMax  int      `yaml:"backend_port_max"`
	CORSOrigins     []string `yaml:"cors_origins,omitempty"`
}

const (
	configDir  = ".lleme"
	configFile = "config.yaml"
	modelsDir  = "models"
	binDir     = "bin"
	cacheDir   = "cache"
	logsDir    = "logs"
	pidsDir    = "pids"
)

// UserHomeDir returns the user's home directory.
func UserHomeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return home
}

// BaseDir returns the base directory for all lleme data.
// Uses LLEME_HOME environment variable if set, otherwise ~/.lleme
func BaseDir() string {
	if dir := os.Getenv("LLEME_HOME"); dir != "" {
		return dir
	}
	return filepath.Join(UserHomeDir(), configDir)
}

func ConfigPath() string {
	return filepath.Join(BaseDir(), configFile)
}

func ModelsPath() string {
	return filepath.Join(BaseDir(), modelsDir)
}

func BinPath() string {
	return filepath.Join(BaseDir(), binDir)
}

func CachePath() string {
	return filepath.Join(BaseDir(), cacheDir)
}

func LogsPath() string {
	return filepath.Join(BaseDir(), logsDir)
}

func PidsPath() string {
	return filepath.Join(BaseDir(), pidsDir)
}

func DefaultConfig() *Config {
	return &Config{
		HuggingFace: HuggingFace{
			Token:        "",
			DefaultQuant: "Q4_K_M",
		},
		LlamaCpp: LlamaCpp{},
		Server: Server{
			Host:            "127.0.0.1",
			Port:            11313,
			MaxModels:       3,
			IdleTimeoutMins: 10,
			StartupTimeoutS: 120,
			BackendPortMin:  49152,
			BackendPortMax:  49200,
			CORSOrigins: []string{
				"http://localhost",
				"http://127.0.0.1",
				"http://[::1]",
			},
		},
	}
}

// DefaultConfigTemplate returns a nicely formatted config with comments
// showing popular llama-server options and their defaults.
const DefaultConfigTemplate = `# Hugging Face settings
huggingface:
  # Access token for gated models (or set HF_TOKEN env var)
  token: ""
  # Default quantization when pulling models
  default_quant: Q4_K_M

# lleme server settings
server:
  host: 127.0.0.1
  port: 11313
  max_models: 3              # Max concurrent models in memory
  idle_timeout_mins: 10      # Unload idle models after this time
  startup_timeout_secs: 120  # Max time to wait for model to load
  backend_port_min: 49152    # Port range for llama-server backends
  backend_port_max: 49200
  cors_origins:              # Allowed CORS origins
    - http://localhost
    - http://127.0.0.1
    - http://[::1]

# llama.cpp server settings
# All options here are passed directly to llama-server.
# See 'llama-server --help' for the full list.
llamacpp:
  # Path to llama-server binary (empty = auto-detect)
  # server_path: ""

  # Any llama-server options can be added here.
  # Uncomment and modify as needed:
  options:
    # --- Performance ---
    # threads: -1              # CPU threads for generation (-1 = auto)
    # threads-batch: -1        # CPU threads for batch processing (-1 = same as threads)
    # ctx-size: 0              # Context size (0 = from model)
    # batch-size: 2048         # Logical batch size
    # ubatch-size: 512         # Physical batch size
    # parallel: -1             # Number of slots/concurrent requests (-1 = auto)

    # --- GPU ---
    # gpu-layers: auto         # Layers to offload to GPU (auto, all, or number)
    # split-mode: layer        # Multi-GPU split: none, layer, row
    # main-gpu: 0              # Primary GPU index
    # flash-attn: auto         # Flash attention (on, off, auto)

    # --- Memory ---
    # cache-type-k: f16        # KV cache type for K (f16, q8_0, q4_0, etc.)
    # cache-type-v: f16        # KV cache type for V
    # mlock: false             # Lock model in RAM (prevents swapping)

    # --- Sampling defaults ---
    # temp: 0.8                # Temperature
    # top-k: 40                # Top-k sampling (0 = disabled)
    # top-p: 0.9               # Top-p / nucleus sampling (1.0 = disabled)
    # min-p: 0.1               # Min-p sampling (0.0 = disabled)
    # repeat-penalty: 1.0      # Repetition penalty (1.0 = disabled)

    # --- Reasoning models ---
    # reasoning-format: auto   # Thinking token handling (auto, none, deepseek)
`

func Load() (*Config, error) {
	cfg := DefaultConfig()

	configPath := ConfigPath()
	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return cfg, nil
}

func Save(cfg *Config) error {
	configPath := ConfigPath()
	configDir := filepath.Dir(configPath)

	if err := os.MkdirAll(configDir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	return nil
}

// SaveDefault writes the default config template with comments.
// Use this for initial config creation or reset.
func SaveDefault() error {
	configPath := ConfigPath()
	configDir := filepath.Dir(configPath)

	if err := os.MkdirAll(configDir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	if err := os.WriteFile(configPath, []byte(DefaultConfigTemplate), 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	return nil
}

// GetOption returns a llama-server option value from the config.
// Returns the value and true if found, or nil and false if not set.
func (c *LlamaCpp) GetOption(key string) (any, bool) {
	if c.Options == nil {
		return nil, false
	}
	val, ok := c.Options[key]
	return val, ok
}

// GetIntOption returns an int option, with a default if not set.
func (c *LlamaCpp) GetIntOption(key string, defaultVal int) int {
	if val, ok := c.GetOption(key); ok {
		switch v := val.(type) {
		case int:
			return v
		case float64:
			return int(v)
		}
	}
	return defaultVal
}

// GetFloatOption returns a float option, with a default if not set.
func (c *LlamaCpp) GetFloatOption(key string, defaultVal float64) float64 {
	if val, ok := c.GetOption(key); ok {
		switch v := val.(type) {
		case float64:
			return v
		case int:
			return float64(v)
		}
	}
	return defaultVal
}

func EnsureDirectories() error {
	dirs := []string{
		ConfigPath(),
		ModelsPath(),
		BinPath(),
		CachePath(),
		LogsPath(),
		PidsPath(),
		PersonasPath(),
	}

	for _, dir := range dirs {
		if filepath.Ext(dir) != "" {
			dir = filepath.Dir(dir)
		}
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	return nil
}
