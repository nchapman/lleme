package cmd

import (
	"fmt"
	"runtime"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/nchapman/lleme/internal/version"
	"github.com/spf13/cobra"
)

var versionCmd = &cobra.Command{
	Use:     "version",
	Short:   "Show version information",
	GroupID: "config",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("lleme %s (%s/%s)\n", version.Version, runtime.GOOS, runtime.GOARCH)

		installed, _ := llama.GetInstalledVersion()
		if installed != nil {
			backend := "CPU"
			if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
				backend = "Metal"
			}
			fmt.Printf("%s (%s)\n", ui.LlamaCppCredit(installed.TagName), backend)
		} else {
			fmt.Println(ui.Muted("llama.cpp not installed"))
		}

		fmt.Println()
		fmt.Println(ui.Header("Paths"))
		fmt.Printf("  %-10s %s\n", "Models", ui.Muted(config.ModelsPath()))
		fmt.Printf("  %-10s %s\n", "Binaries", ui.Muted(config.BinPath()))
		fmt.Printf("  %-10s %s\n", "Config", ui.Muted(config.ConfigPath()))
	},
}

func init() {
	rootCmd.AddCommand(versionCmd)
}
