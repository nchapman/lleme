package components

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestInput_Basic(t *testing.T) {
	input := NewInputWithCompletions(nil, nil)

	// Initially focused
	if !input.IsFocused() {
		t.Error("expected input to be focused initially")
	}
	if !input.Focused() {
		t.Error("expected Focused() alias to match")
	}

	// Initial value is empty
	if input.Value() != "" {
		t.Errorf("expected empty value, got '%s'", input.Value())
	}

	// Set value
	input.SetValue("hello")
	if input.Value() != "hello" {
		t.Errorf("expected 'hello', got '%s'", input.Value())
	}

	// Reset clears value
	input.Reset()
	if input.Value() != "" {
		t.Errorf("expected empty after reset, got '%s'", input.Value())
	}
}

func TestInput_FocusBlur(t *testing.T) {
	input := NewInputWithCompletions(nil, nil)

	input.Blur()
	if input.IsFocused() {
		t.Error("expected not focused after Blur()")
	}

	input.Focus()
	if !input.IsFocused() {
		t.Error("expected focused after Focus()")
	}
}

func TestInput_ValueTrimmed(t *testing.T) {
	input := NewInputWithCompletions(nil, nil)

	input.SetValue("  hello world  ")
	if input.Value() != "hello world" {
		t.Errorf("expected trimmed value 'hello world', got '%s'", input.Value())
	}
}

func TestInput_CompletionsWithCommands(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/help", Value: "/help"},
		{Text: "/clear", Value: "/clear"},
		{Text: "/set", Value: "/set"},
	}
	input := NewInputWithCompletions(cmdItems, nil)

	// Initially completions not open
	if input.IsCompletionsOpen() {
		t.Error("expected completions closed initially")
	}

	// Type "/" - should open completions
	input.SetValue("/")
	input.updateCompletions()
	if !input.IsCompletionsOpen() {
		t.Error("expected completions to open with '/'")
	}

	// Type "/h" - should filter to /help
	input.SetValue("/h")
	input.updateCompletions()
	if !input.IsCompletionsOpen() {
		t.Error("expected completions still open with '/h'")
	}

	// Type "/help " (with space) - should close completions
	input.SetValue("/help ")
	input.updateCompletions()
	if input.IsCompletionsOpen() {
		t.Error("expected completions to close when space added")
	}

	// Type something without "/" - completions should stay closed
	input.SetValue("hello")
	input.updateCompletions()
	if input.IsCompletionsOpen() {
		t.Error("expected completions to stay closed without '/'")
	}
}

func TestInput_SetOptionCompletions(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/set", Value: "/set"},
	}
	setOptionItems := []Completion{
		{Text: "temp", Value: "temp"},
		{Text: "top-p", Value: "top-p"},
		{Text: "top-k", Value: "top-k"},
	}
	input := NewInputWithCompletions(cmdItems, setOptionItems)

	// Type "/set " - should show set option completions
	input.SetValue("/set ")
	input.updateCompletions()
	if !input.IsCompletionsOpen() {
		t.Error("expected completions to open for /set options")
	}

	// Type "/set t" - should filter to temp, top-p, top-k
	input.SetValue("/set t")
	input.updateCompletions()
	if !input.IsCompletionsOpen() {
		t.Error("expected completions to stay open with filter")
	}

	// Type "/set temp " (with value space) - should close
	input.SetValue("/set temp ")
	input.updateCompletions()
	if input.IsCompletionsOpen() {
		t.Error("expected completions to close when typing value")
	}

	// Type "/set temp 0.7" - should stay closed
	input.SetValue("/set temp 0.7")
	input.updateCompletions()
	if input.IsCompletionsOpen() {
		t.Error("expected completions to stay closed with value")
	}
}

func TestInput_IsSetOptionContext(t *testing.T) {
	input := NewInputWithCompletions(nil, nil)

	tests := []struct {
		value    string
		expected bool
	}{
		{"/set ", true},
		{"/SET ", true},
		{"/set temp", true},
		{"/set temp 0.7", true},
		{"/set", false},  // No space after /set
		{"/sett", false}, // Not exactly /set
		{"/help", false},
		{"hello", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.value, func(t *testing.T) {
			input.SetValue(tt.value)
			if got := input.isSetOptionContext(); got != tt.expected {
				t.Errorf("isSetOptionContext(%q) = %v, want %v", tt.value, got, tt.expected)
			}
		})
	}
}

func TestInput_Height(t *testing.T) {
	input := NewInputWithCompletions(nil, nil)

	// Default height is minHeight (1)
	if input.Height() != 1 {
		t.Errorf("expected initial height 1, got %d", input.Height())
	}
}

func TestInput_SetWidth(t *testing.T) {
	input := NewInputWithCompletions(nil, nil)
	input.SetWidth(80)
	// Just verify it doesn't panic - width is internal state
}

func TestInput_NoCompletionsWithoutSlash(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/help", Value: "/help"},
	}
	input := NewInputWithCompletions(cmdItems, nil)

	input.SetValue("regular text")
	input.updateCompletions()

	if input.IsCompletionsOpen() {
		t.Error("completions should not open for regular text")
	}
}

func TestInput_CompletionsViewEmpty(t *testing.T) {
	input := NewInputWithCompletions(nil, nil)

	// Without completions set up, should return empty
	view := input.CompletionsView()
	if view != "" {
		t.Errorf("expected empty view with nil completions, got '%s'", view)
	}
}

func TestInput_NoCommandCompletions(t *testing.T) {
	// Input without command items
	input := NewInputWithCompletions(nil, nil)

	input.SetValue("/")
	input.updateCompletions()

	// Should not panic, and completions should stay closed
	if input.IsCompletionsOpen() {
		t.Error("completions should not open without cmdItems")
	}
}

func TestInput_SetOptionWithoutOptions(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/set", Value: "/set"},
	}
	// No setOptionItems
	input := NewInputWithCompletions(cmdItems, nil)

	input.SetValue("/set ")
	input.updateCompletions()

	// Should close since no set options available
	if input.IsCompletionsOpen() {
		t.Error("completions should close without setOptionItems")
	}
}

func TestInput_CompletionNavigationViaUpdate(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/help", Value: "/help"},
		{Text: "/clear", Value: "/clear"},
		{Text: "/bye", Value: "/bye"},
	}
	input := NewInputWithCompletions(cmdItems, nil)

	// Open completions
	input.SetValue("/")
	input.updateCompletions()

	if !input.IsCompletionsOpen() {
		t.Fatal("expected completions to open")
	}

	// Initial selection should be first item
	sel := input.completions.Selected()
	if sel == nil || sel.Value != "/help" {
		t.Errorf("expected initial selection /help, got %v", sel)
	}

	// Down key should move selection
	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyDown})
	sel = input.completions.Selected()
	if sel == nil || sel.Value != "/clear" {
		t.Errorf("expected /clear after down, got %v", sel)
	}

	// Up key should move back
	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyUp})
	sel = input.completions.Selected()
	if sel == nil || sel.Value != "/help" {
		t.Errorf("expected /help after up, got %v", sel)
	}
}

func TestInput_CompletionEscClosesPopup(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/help", Value: "/help"},
	}
	input := NewInputWithCompletions(cmdItems, nil)

	input.SetValue("/")
	input.updateCompletions()

	if !input.IsCompletionsOpen() {
		t.Fatal("expected completions to open")
	}

	// Esc should close
	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyEscape})
	if input.IsCompletionsOpen() {
		t.Error("expected completions to close after Esc")
	}
}

func TestInput_CompletionSpaceClosesPopup(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/help", Value: "/help"},
	}
	input := NewInputWithCompletions(cmdItems, nil)

	input.SetValue("/")
	input.updateCompletions()

	if !input.IsCompletionsOpen() {
		t.Fatal("expected completions to open")
	}

	// Space should close completions
	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{' '}})
	if input.IsCompletionsOpen() {
		t.Error("expected completions to close after space")
	}
}

func TestInput_CompletionTabSelectsItem(t *testing.T) {
	cmdItems := []Completion{
		{Text: "/help", Value: "/help"},
		{Text: "/clear", Value: "/clear"},
	}
	input := NewInputWithCompletions(cmdItems, nil)

	input.SetValue("/h")
	input.updateCompletions()

	if !input.IsCompletionsOpen() {
		t.Fatal("expected completions to open")
	}

	// Tab should select and close
	var cmd tea.Cmd
	input, cmd = input.Update(tea.KeyMsg{Type: tea.KeyTab})

	if input.IsCompletionsOpen() {
		t.Error("expected completions to close after tab")
	}

	// Value should be updated to selected completion
	if input.Value() != "/help" {
		t.Errorf("expected value '/help', got '%s'", input.Value())
	}

	// Should return CompletionSelectedMsg
	if cmd == nil {
		t.Error("expected cmd to be returned")
	}
}
