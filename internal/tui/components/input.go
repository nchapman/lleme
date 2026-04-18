package components

import (
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/nchapman/lleme/internal/tui/styles"
)

// InputHeightChangedMsg is sent when the input height changes
type InputHeightChangedMsg struct {
	Height int
}

// CompletionSelectedMsg is sent when a completion is selected
type CompletionSelectedMsg struct {
	Value string
}

// Input wraps a textarea for message input
type Input struct {
	textarea       textarea.Model
	width          int
	focused        bool
	minHeight      int
	maxHeight      int
	completions    *Completions
	cmdItems       []Completion // Available command completions
	setOptionItems []Completion // Available /set option completions
}

// NewInput creates a new input component
func NewInput() Input {
	return NewInputWithCompletions(nil, nil)
}

// NewInputWithCompletions creates a new input component with command completions
func NewInputWithCompletions(cmdItems, setOptionItems []Completion) Input {
	ta := textarea.New()
	ta.Placeholder = "Type a message..."
	ta.ShowLineNumbers = false
	ta.CharLimit = 0 // No limit
	ta.SetHeight(1)
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle()
	// No prompt - the border indicates input area
	ta.Prompt = ""
	ta.Focus()

	return Input{
		textarea:       ta,
		focused:        true,
		minHeight:      1,
		maxHeight:      4,
		completions:    NewCompletions(),
		cmdItems:       cmdItems,
		setOptionItems: setOptionItems,
	}
}

// Init returns the initial command
func (i Input) Init() tea.Cmd {
	return textarea.Blink
}

// Update handles input events
func (i Input) Update(msg tea.Msg) (Input, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if i.completions != nil && i.completions.IsOpen() {
			if result, cmd, handled := i.handleCompletionsKey(msg); handled {
				return result, cmd
			}
		}

		switch msg.String() {
		case "shift+enter", "ctrl+j":
			i.textarea.InsertRune('\n')
			return i, i.checkHeightChange()
		case "enter":
			return i, nil
		}
	}

	var cmd tea.Cmd
	i.textarea, cmd = i.textarea.Update(msg)

	i.updateCompletions()

	if heightCmd := i.checkHeightChange(); heightCmd != nil {
		return i, tea.Batch(cmd, heightCmd)
	}
	return i, cmd
}

func (i Input) handleCompletionsKey(msg tea.KeyMsg) (Input, tea.Cmd, bool) {
	switch msg.String() {
	case "up":
		i.completions.MoveUp()
		return i, nil, true
	case "down":
		i.completions.MoveDown()
		return i, nil, true
	case "tab", "enter":
		if sel := i.completions.Selected(); sel != nil {
			var newValue string
			if i.isSetOptionContext() {
				newValue = "/set " + sel.Value + " "
			} else {
				newValue = sel.Value + " "
			}
			i.textarea.SetValue(newValue)
			i.textarea.SetCursor(len(newValue))
			i.completions.Close()
			return i, func() tea.Msg { return CompletionSelectedMsg{Value: sel.Value} }, true
		}
		i.completions.Close()
		return i, nil, true
	case "esc":
		i.completions.Close()
		return i, nil, true
	case " ":
		// Space closes completions and falls through to normal handling
		i.completions.Close()
		return i, nil, false
	}
	return i, nil, false
}

// checkHeightChange adjusts height based on line count and returns a command if changed
func (i *Input) checkHeightChange() tea.Cmd {
	lines := i.textarea.LineCount()
	targetHeight := max(i.minHeight, min(lines, i.maxHeight))
	currentHeight := i.textarea.Height()

	if currentHeight != targetHeight {
		i.textarea.SetHeight(targetHeight)
		return func() tea.Msg {
			return InputHeightChangedMsg{Height: targetHeight}
		}
	}
	return nil
}

// View renders the input
func (i Input) View() string {
	style := styles.InputStyle
	if i.focused {
		style = styles.InputFocusedStyle
	}
	divider := styles.HorizontalDivider(i.width)
	return lipgloss.JoinVertical(lipgloss.Left, "", divider, style.Render(i.textarea.View()))
}

// SetWidth sets the input width
func (i *Input) SetWidth(width int) {
	i.width = width
	i.textarea.SetWidth(width - 4) // Account for padding (2 on each side)
}

// Focus focuses the input
func (i *Input) Focus() tea.Cmd {
	i.focused = true
	return i.textarea.Focus()
}

// Blur removes focus from the input
func (i *Input) Blur() {
	i.focused = false
	i.textarea.Blur()
}

// Value returns the current input value
func (i Input) Value() string {
	return strings.TrimSpace(i.textarea.Value())
}

// SetValue sets the input value
func (i *Input) SetValue(v string) {
	i.textarea.SetValue(v)
}

// Reset clears the input and restores default height
func (i *Input) Reset() {
	i.textarea.Reset()
	i.textarea.SetHeight(i.minHeight)
}

// IsFocused returns whether the input is focused (implements Focusable interface)
func (i Input) IsFocused() bool {
	return i.focused
}

// Focused returns whether the input is focused (alias for IsFocused)
func (i Input) Focused() bool {
	return i.IsFocused()
}

// Height returns the current textarea height
func (i Input) Height() int {
	return i.textarea.Height()
}

// IsCompletionsOpen returns whether completions popup is open
func (i Input) IsCompletionsOpen() bool {
	return i.completions != nil && i.completions.IsOpen()
}

// CompletionsView returns the rendered completions popup
func (i Input) CompletionsView() string {
	if i.completions == nil {
		return ""
	}
	return i.completions.View()
}

const setCommandPrefix = "/set "

// updateCompletions checks input and opens/updates completions as needed
func (i *Input) updateCompletions() {
	if i.completions == nil {
		return
	}

	value := i.textarea.Value()

	// Check for /set <option> context first
	if i.isSetOptionContext() {
		if len(i.setOptionItems) == 0 {
			i.completions.Close()
			return
		}

		// Get the part after "/set "
		optionPart := value[len(setCommandPrefix):]

		// If there's another space, user is typing value - close completions
		if strings.Contains(optionPart, " ") {
			i.completions.Close()
			return
		}

		// Show/filter set options
		if !i.completions.IsOpen() {
			i.completions.Open(i.setOptionItems)
		}
		i.completions.Filter(optionPart)
		return
	}

	// Command completions: only if starts with "/" and no space
	if len(i.cmdItems) == 0 {
		return
	}

	if !strings.HasPrefix(value, "/") {
		i.completions.Close()
		return
	}

	// Don't show command completions if there's a space (user is typing args)
	if strings.Contains(value, " ") {
		i.completions.Close()
		return
	}

	// Open or filter command completions
	if !i.completions.IsOpen() {
		i.completions.Open(i.cmdItems)
	}
	i.completions.Filter(value)
}

// isSetOptionContext returns true if input is in "/set <option>" context
func (i Input) isSetOptionContext() bool {
	return strings.HasPrefix(strings.ToLower(i.textarea.Value()), setCommandPrefix)
}
