package chat

// CommandDef defines a slash command for the TUI
type CommandDef struct {
	Name        string   // Primary name (e.g., "/help")
	Aliases     []string // Alternative names (e.g., ["/?"])
	Description string   // Short description for popup
}

// Commands is the list of available slash commands
var Commands = []CommandDef{
	{Name: "/help", Aliases: []string{"/?"}, Description: "Show help"},
	{Name: "/clear", Description: "Clear conversation"},
	{Name: "/system", Description: "Show/set system prompt"},
	{Name: "/set", Description: "Change a setting"},
	{Name: "/show", Description: "Show current settings"},
	{Name: "/reload", Description: "Reload model"},
	{Name: "/bye", Aliases: []string{"/exit", "/quit"}, Description: "Exit chat"},
}

// SetOptionDef defines an option for the /set command
type SetOptionDef struct {
	Name        string // Option name (e.g., "temp")
	Description string // Description with value hint
}

// SetOptions is the list of available /set options
var SetOptions = []SetOptionDef{
	{Name: "temp", Description: "Temperature (0.0-2.0)"},
	{Name: "top-p", Description: "Top-P sampling (0.0-1.0)"},
	{Name: "top-k", Description: "Top-K sampling (integer)"},
	{Name: "min-p", Description: "Min-P sampling (0.0-1.0)"},
	{Name: "repeat-penalty", Description: "Repeat penalty (0.0-2.0)"},
	{Name: "presence-penalty", Description: "Presence penalty (-2.0-2.0)"},
	{Name: "frequency-penalty", Description: "Frequency penalty (-2.0-2.0)"},
	{Name: "ctx-size", Description: "Context size (requires /reload)"},
	{Name: "gpu-layers", Description: "GPU layers (requires /reload)"},
	{Name: "threads", Description: "CPU threads (requires /reload)"},
}
