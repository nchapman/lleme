package ui

import (
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/nchapman/lleme/internal/styles"
)

// Re-export icons from shared styles for convenience.
const (
	IconCheck = styles.IconCheck
	IconCross = styles.IconCross
	IconArrow = styles.IconArrow
)

var (
	headerStyle  = lipgloss.NewStyle().Bold(true).Foreground(styles.ColorPrimary)
	successStyle = lipgloss.NewStyle().Foreground(styles.ColorSuccess)
	errorStyle   = lipgloss.NewStyle().Foreground(styles.ColorError)
	warningStyle = lipgloss.NewStyle().Foreground(styles.ColorWarning)
	mutedStyle   = lipgloss.NewStyle().Foreground(styles.ColorMuted)
	boldStyle    = lipgloss.NewStyle().Bold(true)
	keywordStyle = lipgloss.NewStyle().Bold(true).Foreground(styles.ColorAccent)

	// ExitFunc is the function called by Fatal. Override in tests to prevent os.Exit.
	// Tests that modify this must use t.Cleanup() to restore the original value.
	ExitFunc = os.Exit
)

func Header(text string) string {
	return headerStyle.Render(text)
}

func Success(text string) string {
	return successStyle.Render(text)
}

func ErrorMsg(text string) string {
	return errorStyle.Render(text)
}

func Warning(text string) string {
	return warningStyle.Render(text)
}

func Muted(text string) string {
	return mutedStyle.Render(text)
}

func Bold(text string) string {
	return boldStyle.Render(text)
}

func Keyword(text string) string {
	return keywordStyle.Render(text)
}

// LlamaCppCredit returns the llama.cpp attribution line.
func LlamaCppCredit(version string) string {
	return Muted(fmt.Sprintf("Powered by llama.cpp %s", version))
}

// SwiftLMCredit returns the SwiftLM attribution line.
func SwiftLMCredit(version string) string {
	return Muted(fmt.Sprintf("Powered by SwiftLM %s", version))
}

// BackendsCredit returns a combined attribution line for whichever backends
// are installed. Empty version strings are omitted so we never show a
// "Powered by llama.cpp " with no tag on hosts where the backend isn't
// present. Separator is a middle dot for visual parity with the rest of
// the status lines.
func BackendsCredit(llamaVersion, swiftLMVersion string) string {
	var parts []string
	if llamaVersion != "" {
		parts = append(parts, fmt.Sprintf("llama.cpp %s", llamaVersion))
	}
	if swiftLMVersion != "" {
		parts = append(parts, fmt.Sprintf("SwiftLM %s", swiftLMVersion))
	}
	if len(parts) == 0 {
		return ""
	}
	return Muted("Powered by " + strings.Join(parts, " • "))
}

// Fatal prints an error message to stderr and exits with code 1.
func Fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s %s\n", ErrorMsg("Error:"), fmt.Sprintf(format, args...))
	ExitFunc(1)
}

// PrintError prints an error message to stderr without exiting.
func PrintError(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s %s\n", ErrorMsg("Error:"), fmt.Sprintf(format, args...))
}
