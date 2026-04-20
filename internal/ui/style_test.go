package ui

import (
	"strings"
	"testing"
)

func TestStyleFunctions(t *testing.T) {
	testCases := []struct {
		name string
		fn   func(string) string
	}{
		{"Header", Header},
		{"Success", Success},
		{"ErrorMsg", ErrorMsg},
		{"Warning", Warning},
		{"Muted", Muted},
		{"Bold", Bold},
		{"Keyword", Keyword},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			input := "test text"
			result := tc.fn(input)

			if result == "" {
				t.Errorf("%s() returned empty string", tc.name)
			}

			if !strings.Contains(result, "test text") {
				t.Errorf("%s() result does not contain input text", tc.name)
			}

			// Note: lipgloss disables ANSI codes when not in a terminal,
			// so we can't assert result != input in test environment
		})
	}
}

func TestStyleFunctionsEmptyInput(t *testing.T) {
	testCases := []struct {
		name string
		fn   func(string) string
	}{
		{"Header", Header},
		{"Success", Success},
		{"ErrorMsg", ErrorMsg},
		{"Warning", Warning},
		{"Muted", Muted},
		{"Bold", Bold},
		{"Keyword", Keyword},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"_empty", func(t *testing.T) {
			// Should not panic on empty input
			result := tc.fn("")
			_ = result
		})
	}
}

func TestLlamaCppCredit(t *testing.T) {
	result := LlamaCppCredit("b1234")
	if result == "" {
		t.Error("Expected non-empty result")
	}
	if !strings.Contains(result, "llama.cpp") {
		t.Error("Expected result to contain 'llama.cpp'")
	}
	if !strings.Contains(result, "b1234") {
		t.Error("Expected result to contain version 'b1234'")
	}
}

func TestSwiftLMCredit(t *testing.T) {
	result := SwiftLMCredit("b517")
	if !strings.Contains(result, "SwiftLM") || !strings.Contains(result, "b517") {
		t.Errorf("SwiftLMCredit = %q, want to include SwiftLM and version", result)
	}
}

func TestBackendsCredit(t *testing.T) {
	tests := []struct {
		name       string
		llama      string
		swiftlm    string
		wantEmpty  bool
		wantSubstr []string
	}{
		{name: "both tags", llama: "b1234", swiftlm: "b517", wantSubstr: []string{"llama.cpp b1234", "SwiftLM b517", "•"}},
		{name: "llama only", llama: "b1234", wantSubstr: []string{"llama.cpp b1234"}},
		{name: "swiftlm only", swiftlm: "b517", wantSubstr: []string{"SwiftLM b517"}},
		{name: "neither", wantEmpty: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BackendsCredit(tt.llama, tt.swiftlm)
			if tt.wantEmpty {
				if got != "" {
					t.Errorf("BackendsCredit() = %q, want empty", got)
				}
				return
			}
			for _, s := range tt.wantSubstr {
				if !strings.Contains(got, s) {
					t.Errorf("BackendsCredit() = %q, want to contain %q", got, s)
				}
			}
			// Never print a naked "llama.cpp " or "SwiftLM " with no tag.
			if tt.llama == "" && strings.Contains(got, "llama.cpp") {
				t.Errorf("should omit llama.cpp when empty tag: %q", got)
			}
			if tt.swiftlm == "" && strings.Contains(got, "SwiftLM") {
				t.Errorf("should omit SwiftLM when empty tag: %q", got)
			}
		})
	}
}

func TestIconConstants(t *testing.T) {
	if IconCheck == "" {
		t.Error("Expected IconCheck to be non-empty")
	}
	if IconCross == "" {
		t.Error("Expected IconCross to be non-empty")
	}
	if IconArrow == "" {
		t.Error("Expected IconArrow to be non-empty")
	}
}

func TestFatal_ExitFuncOverride(t *testing.T) {
	var exitCode int
	originalExit := ExitFunc
	ExitFunc = func(code int) { exitCode = code }
	t.Cleanup(func() { ExitFunc = originalExit })

	Fatal("test error: %s", "details")

	if exitCode != 1 {
		t.Errorf("expected exit code 1, got %d", exitCode)
	}
}
