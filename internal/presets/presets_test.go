package presets

import (
	"testing"
)

func TestLoad(t *testing.T) {
	presets, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if len(presets) == 0 {
		t.Fatal("Load() returned no presets")
	}
	for _, p := range presets {
		if p.Name == "" {
			t.Errorf("preset %q has no name", p.filename)
		}
		if len(p.Match) == 0 {
			t.Errorf("preset %q has no match patterns", p.filename)
		}
		if p.Source == "" {
			t.Errorf("preset %q has no source URL", p.filename)
		}
	}
}

func TestLoadSorted(t *testing.T) {
	presets, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	for i := 1; i < len(presets); i++ {
		if presets[i].filename < presets[i-1].filename {
			t.Errorf("presets not sorted: %q before %q", presets[i-1].filename, presets[i].filename)
		}
	}
}

func TestFind(t *testing.T) {
	tests := []struct {
		name        string
		modelName   string
		wantName    string
		wantNoMatch bool
	}{
		{
			name:      "Qwen3.5 with Instruct suffix",
			modelName: "unsloth/Qwen3.5-4B-Instruct-GGUF:Q4_K_M",
			wantName:  "Qwen3.5",
		},
		{
			name:      "Qwen3 base bartowski",
			modelName: "bartowski/Qwen3-4B-Instruct-GGUF:Q4_K_M",
			wantName:  "Qwen3",
		},
		{
			name:      "Qwen3 thinking variant",
			modelName: "unsloth/Qwen3-4B-Thinking-GGUF:Q4_K_M",
			wantName:  "Qwen3",
		},
		{
			name:      "Gemma 4",
			modelName: "ggml-org/gemma-4-4b-it-GGUF:Q4_K_M",
			wantName:  "Gemma 4",
		},
		{
			name:        "unknown model returns nil",
			modelName:   "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
			wantNoMatch: true,
		},
		{
			name:      "no quant suffix still matches",
			modelName: "bartowski/Qwen3-4B-Instruct-GGUF",
			wantName:  "Qwen3",
		},
		{
			name:      "Qwen3.5 bare repo",
			modelName: "unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL",
			wantName:  "Qwen3.5",
		},
		{
			name:      "Qwen3.6 MoE",
			modelName: "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
			wantName:  "Qwen3.6",
		},
		{
			name:      "upstream Qwen namespace — Qwen3 base",
			modelName: "Qwen/Qwen3-8B-GGUF:Q4_K_M",
			wantName:  "Qwen3",
		},
		{
			name:      "upstream Qwen namespace — Qwen3.5",
			modelName: "Qwen/Qwen3.5-27B-Instruct:Q4_K_M",
			wantName:  "Qwen3.5",
		},
		{
			name:      "upstream Qwen namespace — Qwen3.6",
			modelName: "Qwen/Qwen3.6-35B-A3B",
			wantName:  "Qwen3.6",
		},
		{
			name:      "GLM-4.7 Flash",
			modelName: "unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL",
			wantName:  "GLM-4.7 Flash",
		},
		{
			name:      "GPT OSS 20B",
			modelName: "unsloth/gpt-oss-20b-GGUF:Q4_K_M",
			wantName:  "GPT OSS",
		},
		{
			name:      "GPT OSS 120B",
			modelName: "openai/gpt-oss-120b:Q4_K_M",
			wantName:  "GPT OSS",
		},

		// Qwen sub-families — all must win against the more generic Qwen3 preset.
		{
			name:      "Qwen3-Coder wins over Qwen3",
			modelName: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M",
			wantName:  "Qwen3-Coder",
		},
		{
			name:      "Qwen3-Coder-Next wins over Qwen3-Coder",
			modelName: "unsloth/Qwen3-Coder-Next-GGUF:UD-Q4_K_XL",
			wantName:  "Qwen3-Coder-Next",
		},
		{
			name:      "Qwen3-Next Instruct wins over Qwen3",
			modelName: "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF:Q4_K_M",
			wantName:  "Qwen3-Next Instruct",
		},
		{
			name:      "Qwen3-Next Thinking",
			modelName: "unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF:Q4_K_M",
			wantName:  "Qwen3-Next Thinking",
		},
		{
			name:      "Qwen3-VL Instruct",
			modelName: "unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL",
			wantName:  "Qwen3-VL Instruct",
		},
		{
			name:      "Qwen3-VL Thinking",
			modelName: "unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:Q4_K_XL",
			wantName:  "Qwen3-VL Thinking",
		},
		{
			name:      "QwQ",
			modelName: "unsloth/QwQ-32B-GGUF:Q4_K_M",
			wantName:  "QwQ",
		},

		// DeepSeek — R1-0528 and V3-0324 must win over their more-general siblings.
		{
			name:      "DeepSeek-R1-0528 wins over DeepSeek-R1",
			modelName: "unsloth/DeepSeek-R1-0528-GGUF:UD-Q4_K_XL",
			wantName:  "DeepSeek-R1-0528",
		},
		{
			name:      "DeepSeek-R1 base",
			modelName: "unsloth/DeepSeek-R1-GGUF:UD-IQ1_S",
			wantName:  "DeepSeek-R1",
		},
		{
			name:      "DeepSeek-V3-0324",
			modelName: "unsloth/DeepSeek-V3-0324-GGUF:UD-Q2_K_XL",
			wantName:  "DeepSeek-V3-0324",
		},
		{
			name:      "DeepSeek-V3.1 Terminus",
			modelName: "unsloth/DeepSeek-V3.1-Terminus-GGUF:UD-Q4_K_XL",
			wantName:  "DeepSeek-V3.1",
		},

		// GLM — 5.1 must win over 5 for version-5.1 repos.
		{
			name:      "GLM-5.1",
			modelName: "unsloth/GLM-5.1-GGUF:Q4_K_M",
			wantName:  "GLM-5.1",
		},
		{
			name:      "GLM-5 base",
			modelName: "unsloth/GLM-5-GGUF:Q4_K_M",
			wantName:  "GLM-5",
		},

		// Gemma — 3, 3n, 4, and FunctionGemma must be distinct.
		{
			name:      "Gemma 3",
			modelName: "unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
			wantName:  "Gemma 3",
		},
		{
			name:      "Gemma 3n does not match Gemma 3",
			modelName: "unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M",
			wantName:  "Gemma 3n",
		},
		{
			name:      "FunctionGemma",
			modelName: "unsloth/FunctionGemma-270m-it-GGUF:Q4_K_M",
			wantName:  "FunctionGemma",
		},

		// MiniMax.
		{
			name:      "MiniMax M2.5",
			modelName: "unsloth/MiniMax-M2.5-GGUF:UD-Q4_K_XL",
			wantName:  "MiniMax M2.5",
		},
		{
			name:      "MiniMax M2.7",
			modelName: "MiniMaxAI/MiniMax-M2.7",
			wantName:  "MiniMax M2.7",
		},

		// Kimi — K2.5 and K2-Thinking must win over K2 base.
		{
			name:      "Kimi-K2.5",
			modelName: "unsloth/Kimi-K2.5-GGUF:UD-Q2_K_XL",
			wantName:  "Kimi-K2.5",
		},
		{
			name:      "Kimi-K2 Thinking",
			modelName: "moonshotai/Kimi-K2-Thinking",
			wantName:  "Kimi-K2 Thinking",
		},
		{
			name:      "Kimi-K2 Instruct base",
			modelName: "moonshotai/Kimi-K2-Instruct",
			wantName:  "Kimi-K2",
		},
		{
			name:      "Kimi-K2 bare GGUF repo",
			modelName: "unsloth/Kimi-K2-GGUF:Q4_K_M",
			wantName:  "Kimi-K2",
		},

		// Nemotron — single preset covers both Nemotron-3 and NVIDIA-Nemotron-3 namespacing.
		{
			name:      "Nemotron 3 Super",
			modelName: "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF:UD-Q4_K_XL",
			wantName:  "Nemotron 3",
		},
		{
			name:      "Nemotron 3 Nano",
			modelName: "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
			wantName:  "Nemotron 3",
		},

		// Mistral family.
		{
			name:      "Magistral",
			modelName: "unsloth/Magistral-Small-2509-GGUF:UD-Q4_K_XL",
			wantName:  "Magistral",
		},
		{
			name:      "Devstral",
			modelName: "bartowski/Devstral-Small-2507-GGUF:Q4_K_M",
			wantName:  "Devstral",
		},
		{
			name:      "Ministral 3 Instruct",
			modelName: "unsloth/Ministral-3-14B-Instruct-2512-GGUF:Q4_K_M",
			wantName:  "Ministral 3 Instruct",
		},
		{
			name:      "Ministral 3 Reasoning",
			modelName: "unsloth/Ministral-3-8B-Reasoning-2512-GGUF:Q4_K_M",
			wantName:  "Ministral 3 Reasoning",
		},

		// Misc.
		{
			name:      "Llama 4 Scout",
			modelName: "meta-llama/Llama-4-Scout-17B-16E-Instruct",
			wantName:  "Llama 4",
		},
		{
			name:      "Llama 4 Maverick GGUF",
			modelName: "unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF",
			wantName:  "Llama 4",
		},
		{
			name:      "Grok 2",
			modelName: "unsloth/grok-2-GGUF:UD-Q3_K_XL",
			wantName:  "Grok 2",
		},
		{
			name:      "Phi-4 Reasoning",
			modelName: "microsoft/Phi-4-reasoning-plus",
			wantName:  "Phi-4 Reasoning",
		},
		{
			name:      "Phi-4 mini reasoning",
			modelName: "unsloth/Phi-4-mini-reasoning-GGUF:Q4_K_M",
			wantName:  "Phi-4 Reasoning",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preset, pattern := Find(tt.modelName)
			if tt.wantNoMatch {
				if preset != nil {
					t.Errorf("expected no match, got preset %q (pattern %q)", preset.Name, pattern)
				}
				return
			}
			if preset == nil {
				t.Fatalf("expected preset %q, got nil", tt.wantName)
			}
			if preset.Name != tt.wantName {
				t.Errorf("got preset %q, want %q (pattern %q)", preset.Name, tt.wantName, pattern)
			}
			if pattern == "" {
				t.Error("expected non-empty matched pattern")
			}
		})
	}
}

func TestRepoName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"user/repo:Q4_K_M", "user/repo"},
		{"user/repo", "user/repo"},
		{"user/repo:Q8_0", "user/repo"},
	}
	for _, tt := range tests {
		got := repoName(tt.input)
		if got != tt.want {
			t.Errorf("repoName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestGetFloatOption(t *testing.T) {
	p := &Preset{Options: map[string]any{"temp": 0.7, "top-k": 20}}

	if v := p.GetFloatOption("temp", 0); v != 0.7 {
		t.Errorf("got %v, want 0.7", v)
	}
	if v := p.GetFloatOption("missing", 1.0); v != 1.0 {
		t.Errorf("got %v, want 1.0 (default)", v)
	}
	// int value coerced to float
	if v := p.GetFloatOption("top-k", 0); v != 20.0 {
		t.Errorf("got %v, want 20.0", v)
	}

	var nilPreset *Preset
	if v := nilPreset.GetFloatOption("temp", 0.5); v != 0.5 {
		t.Errorf("nil preset: got %v, want 0.5", v)
	}
}

func TestGetIntOption(t *testing.T) {
	p := &Preset{Options: map[string]any{"top-k": 20, "temp": 0.7}}

	if v := p.GetIntOption("top-k", 0); v != 20 {
		t.Errorf("got %v, want 20", v)
	}
	if v := p.GetIntOption("missing", 40); v != 40 {
		t.Errorf("got %v, want 40 (default)", v)
	}
	// float64 value coerced to int
	if v := p.GetIntOption("temp", 0); v != 0 {
		t.Errorf("got %v, want 0 (truncated float)", v)
	}
}

func TestMergeServerOptions(t *testing.T) {
	t.Run("both nil returns nil", func(t *testing.T) {
		if got := MergeServerOptions(nil, nil); got != nil {
			t.Errorf("expected nil, got %v", got)
		}
	})

	t.Run("preset only", func(t *testing.T) {
		p := &Preset{Options: map[string]any{"temp": 0.7, "top-k": 20}}
		got := MergeServerOptions(p, nil)
		if got == nil {
			t.Fatal("expected non-nil")
		}
		if got["temp"] != 0.7 {
			t.Errorf("temp = %v, want 0.7", got["temp"])
		}
		if got["top-k"] != 20 {
			t.Errorf("top-k = %v, want 20", got["top-k"])
		}
	})

	t.Run("persona only", func(t *testing.T) {
		got := MergeServerOptions(nil, map[string]any{"ctx-size": 4096})
		if got == nil {
			t.Fatal("expected non-nil")
		}
		if got["ctx-size"] != 4096 {
			t.Errorf("ctx-size = %v, want 4096", got["ctx-size"])
		}
	})

	t.Run("persona wins over preset on conflict", func(t *testing.T) {
		p := &Preset{Options: map[string]any{"temp": 0.7, "top-k": 20}}
		persona := map[string]any{"temp": 0.5, "ctx-size": 4096}
		got := MergeServerOptions(p, persona)
		if got["temp"] != 0.5 {
			t.Errorf("temp = %v, want 0.5 (persona wins)", got["temp"])
		}
		if got["top-k"] != 20 {
			t.Errorf("top-k = %v, want 20 (from preset)", got["top-k"])
		}
		if got["ctx-size"] != 4096 {
			t.Errorf("ctx-size = %v, want 4096 (from persona)", got["ctx-size"])
		}
	})

	t.Run("returned map is independent of inputs", func(t *testing.T) {
		p := &Preset{Options: map[string]any{"temp": 0.7}}
		persona := map[string]any{"top-k": 20}
		got := MergeServerOptions(p, persona)
		got["injected"] = true
		if _, ok := p.Options["injected"]; ok {
			t.Error("mutation affected preset")
		}
		if _, ok := persona["injected"]; ok {
			t.Error("mutation affected persona map")
		}
	})
}

func TestGetOptions(t *testing.T) {
	p := &Preset{Options: map[string]any{"temp": 0.7, "top-k": 20}}
	got := p.GetOptions()
	if got == nil {
		t.Fatal("expected non-nil options")
	}
	// Mutating the copy must not affect the preset.
	got["injected"] = true
	if _, ok := p.Options["injected"]; ok {
		t.Error("mutating returned options affected preset")
	}

	var nilPreset *Preset
	if nilPreset.GetOptions() != nil {
		t.Error("nil preset GetOptions should return nil")
	}
}
