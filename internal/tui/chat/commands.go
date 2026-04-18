package chat

import (
	"fmt"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nchapman/lleme/internal/server"
	"github.com/nchapman/lleme/internal/tui/components"
)

// handleCommand processes a slash command and returns a command
func (m *Model) handleCommand(input string) tea.Cmd {
	parts := strings.Fields(input)
	if len(parts) == 0 {
		return nil
	}

	cmd := strings.ToLower(parts[0])
	args := parts[1:]

	return func() tea.Msg {
		switch cmd {
		case "/help", "/?":
			return CommandResultMsg{Message: m.helpText()}

		case "/bye", "/exit", "/quit":
			return CommandResultMsg{Message: "Goodbye!", Exit: true}

		case "/clear":
			m.initSystemPrompt()
			m.messages.ClearMessages()
			return CommandResultMsg{Message: "Conversation cleared"}

		case "/system":
			if len(args) == 0 {
				// Show current system prompt
				if len(m.chatMessages) > 0 && m.chatMessages[0].Role == "system" {
					return CommandResultMsg{Message: "System prompt:\n" + m.chatMessages[0].Content}
				}
				return CommandResultMsg{Message: "No system prompt set"}
			}
			// Set new system prompt
			newPrompt := strings.Join(args, " ")
			m.chatMessages = []server.ChatMessage{{Role: "system", Content: newPrompt}}
			m.messages.ClearMessages()
			return CommandResultMsg{Message: "System prompt updated, conversation cleared"}

		case "/set":
			if len(args) < 2 {
				return CommandResultMsg{
					Message: "Usage: /set <option> <value>\nOptions: temp, top-p, top-k, repeat-penalty, min-p, ctx-size, gpu-layers, threads",
					IsError: true,
				}
			}
			return m.handleSet(args[0], args[1])

		case "/reload":
			return m.handleReload()

		case "/show":
			return CommandResultMsg{Message: m.showSettings()}

		default:
			return CommandResultMsg{
				Message: fmt.Sprintf("Unknown command: %s (type /? for help)", cmd),
				IsError: true,
			}
		}
	}
}

// handleSet processes the /set command
func (m *Model) handleSet(option, value string) CommandResultMsg {
	option = strings.ToLower(option)

	floatVal, floatErr := strconv.ParseFloat(value, 64)
	intVal, intErr := strconv.Atoi(value)

	switch option {
	case "temp", "temperature":
		if floatErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for temp: %s", value), IsError: true}
		}
		m.options.Temp = floatVal
		return CommandResultMsg{Message: fmt.Sprintf("Set temp = %g", floatVal)}

	case "top-p":
		if floatErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for top-p: %s", value), IsError: true}
		}
		m.options.TopP = floatVal
		return CommandResultMsg{Message: fmt.Sprintf("Set top-p = %g", floatVal)}

	case "top-k":
		if intErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for top-k: %s", value), IsError: true}
		}
		m.options.TopK = intVal
		return CommandResultMsg{Message: fmt.Sprintf("Set top-k = %d", intVal)}

	case "repeat-penalty":
		if floatErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for repeat-penalty: %s", value), IsError: true}
		}
		m.options.RepeatPenalty = floatVal
		return CommandResultMsg{Message: fmt.Sprintf("Set repeat-penalty = %g", floatVal)}

	case "min-p":
		if floatErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for min-p: %s", value), IsError: true}
		}
		m.options.MinP = floatVal
		return CommandResultMsg{Message: fmt.Sprintf("Set min-p = %g", floatVal)}

	case "ctx-size":
		if intErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for ctx-size: %s", value), IsError: true}
		}
		m.options.CtxSize = intVal
		m.options.CtxSizeSet = true
		m.pendingReload = true
		return CommandResultMsg{Message: fmt.Sprintf("Set ctx-size = %d (use /reload to apply)", intVal)}

	case "gpu-layers":
		if intErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for gpu-layers: %s", value), IsError: true}
		}
		m.options.GpuLayers = intVal
		m.options.GpuLayersSet = true
		m.pendingReload = true
		return CommandResultMsg{Message: fmt.Sprintf("Set gpu-layers = %d (use /reload to apply)", intVal)}

	case "threads":
		if intErr != nil {
			return CommandResultMsg{Message: fmt.Sprintf("Invalid value for threads: %s", value), IsError: true}
		}
		m.options.Threads = intVal
		m.options.ThreadsSet = true
		m.pendingReload = true
		return CommandResultMsg{Message: fmt.Sprintf("Set threads = %d (use /reload to apply)", intVal)}

	default:
		return CommandResultMsg{
			Message: fmt.Sprintf("Unknown option: %s\nOptions: temp, top-p, top-k, repeat-penalty, min-p, ctx-size, gpu-layers, threads", option),
			IsError: true,
		}
	}
}

// handleReload reloads the model with new server options
func (m *Model) handleReload() CommandResultMsg {
	if !m.pendingReload {
		return CommandResultMsg{Message: "No pending server option changes to apply"}
	}

	// Stop the current model
	if err := m.api.StopModel(m.model); err != nil {
		return CommandResultMsg{Message: fmt.Sprintf("Failed to stop model: %v", err), IsError: true}
	}

	// Reload with persona options as base, session options override
	opts := &server.RunOptions{}
	if m.persona != nil {
		opts.Options = m.persona.GetServerOptions()
	}
	if m.options.CtxSizeSet {
		opts.CtxSize = server.IntPtr(m.options.CtxSize)
	}
	if m.options.GpuLayersSet {
		opts.GpuLayers = server.IntPtr(m.options.GpuLayers)
	}
	if m.options.ThreadsSet {
		opts.Threads = server.IntPtr(m.options.Threads)
	}
	if err := m.api.Run(m.model, opts); err != nil {
		return CommandResultMsg{Message: fmt.Sprintf("Failed to reload model: %v", err), IsError: true}
	}

	m.pendingReload = false
	return CommandResultMsg{Message: "Model reloaded"}
}

// helpText returns the help message
func (m *Model) helpText() string {
	var sb strings.Builder
	sb.WriteString("Commands:\n")
	for _, cmd := range Commands {
		// Format: name + aliases, padded to 22 chars, then description
		names := cmd.Name
		if len(cmd.Aliases) > 0 {
			names += ", " + strings.Join(cmd.Aliases, ", ")
		}
		fmt.Fprintf(&sb, "  %-20s %s\n", names, cmd.Description)
	}
	sb.WriteString("\nOptions for /set:\n")
	sb.WriteString("  temp, top-p, top-k, repeat-penalty, min-p\n")
	sb.WriteString("  ctx-size*, gpu-layers*, threads*  (* require /reload)")
	return sb.String()
}

// showSettings returns the current settings as a string
func (m *Model) showSettings() string {
	var sb strings.Builder

	sb.WriteString("Current Settings\n\n")
	fmt.Fprintf(&sb, "  Model: %s\n\n", m.model)

	// Show system prompt (truncated if long)
	if len(m.chatMessages) > 0 && m.chatMessages[0].Role == "system" {
		prompt := m.chatMessages[0].Content
		if len(prompt) > 80 {
			prompt = prompt[:77] + "..."
		}
		fmt.Fprintf(&sb, "  System: %s\n\n", prompt)
	}

	// Request-time options
	sb.WriteString("  Sampling:\n")
	sb.WriteString(m.formatOption("temp", m.options.Temp, m.resolver.GetConfigFloat("temp")))
	sb.WriteString(m.formatOption("top-p", m.options.TopP, m.resolver.GetConfigFloat("top-p")))
	sb.WriteString(m.formatOptionInt("top-k", m.options.TopK, m.resolver.GetConfigInt("top-k")))
	sb.WriteString(m.formatOption("repeat-penalty", m.options.RepeatPenalty, m.resolver.GetConfigFloat("repeat-penalty")))
	sb.WriteString(m.formatOption("min-p", m.options.MinP, m.resolver.GetConfigFloat("min-p")))
	sb.WriteString("\n")

	// Server options
	sb.WriteString("  Server:\n")
	sb.WriteString(m.formatServerOption("ctx-size", m.options.CtxSize, m.options.CtxSizeSet, m.resolver.GetConfigInt("ctx-size")))
	sb.WriteString(m.formatServerOption("gpu-layers", m.options.GpuLayers, m.options.GpuLayersSet, m.resolver.GetConfigInt("gpu-layers")))
	sb.WriteString(m.formatServerOption("threads", m.options.Threads, m.options.ThreadsSet, m.resolver.GetConfigInt("threads")))

	return sb.String()
}

// formatSetting formats a setting line showing session/config/default value.
func formatSetting(name, sessionVal, configVal string) string {
	if sessionVal != "" {
		return fmt.Sprintf("    %s = %s (session)\n", name, sessionVal)
	}
	if configVal != "" {
		return fmt.Sprintf("    %s = %s (config)\n", name, configVal)
	}
	return fmt.Sprintf("    %s = default\n", name)
}

func (m *Model) formatOption(name string, sessionVal, configVal float64) string {
	var session, config string
	if sessionVal != 0 {
		session = fmt.Sprintf("%g", sessionVal)
	}
	if configVal != 0 {
		config = fmt.Sprintf("%g", configVal)
	}
	return formatSetting(name, session, config)
}

func (m *Model) formatOptionInt(name string, sessionVal, configVal int) string {
	var session, config string
	if sessionVal != 0 {
		session = fmt.Sprintf("%d", sessionVal)
	}
	if configVal != 0 {
		config = fmt.Sprintf("%d", configVal)
	}
	return formatSetting(name, session, config)
}

func (m *Model) formatServerOption(name string, sessionVal int, isSet bool, configVal int) string {
	var session, config string
	if isSet {
		session = fmt.Sprintf("%d", sessionVal)
	}
	if configVal != 0 {
		config = fmt.Sprintf("%d", configVal)
	}
	return formatSetting(name, session, config)
}

// ClearMessages clears the messages viewport (called from command handler)
func (m *Model) ClearMessages() {
	m.messages.ClearMessages()
}

// AddSystemMessage adds a system message to the viewport
func (m *Model) AddSystemMessage(content string) {
	m.messages.AddMessage(components.Message{
		Role:    components.RoleSystem,
		Content: content,
	})
}
