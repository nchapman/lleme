package cmd

import (
	"os"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/logs"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/nchapman/lleme/internal/version"
	"github.com/spf13/cobra"
)

var verbose bool

var rootCmd = &cobra.Command{
	Use:     "lleme",
	Short:   "Run local LLMs with llama.cpp and Hugging Face",
	Version: version.Version,
	Long: `Run local LLMs with llama.cpp and Hugging Face.

Point it at any GGUF model on Hugging Face, and it handles the rest—downloading,
caching, and running inference.`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		logs.InitLogger(nil, verbose)
		if err := config.EnsureDirectories(); err != nil {
			ui.Fatal("Failed to create directories: %v", err)
		}
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose output")
	rootCmd.CompletionOptions.DisableDefaultCmd = true

	// Add command groups
	rootCmd.AddGroup(
		&cobra.Group{ID: "model", Title: "Model Commands:"},
		&cobra.Group{ID: "persona", Title: "Personas:"},
		&cobra.Group{ID: "server", Title: "Server:"},
		&cobra.Group{ID: "discovery", Title: "Discovery:"},
		&cobra.Group{ID: "config", Title: "Configuration:"},
	)
}
