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
