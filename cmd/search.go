package cmd

import (
	"fmt"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/proxy"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var searchCmd = &cobra.Command{
	Use:     "search [query]",
	Short:   "Search Hugging Face for compatible models",
	GroupID: "discovery",
	Long:    "Search Hugging Face for models compatible with any registered backend (GGUF and MLX today). If no query is provided, shows trending models.",
	Args:    cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := config.Load()
		if err != nil {
			ui.Fatal("Failed to load config: %v", err)
		}

		client := hf.NewClient(cfg)
		query := ""
		if len(args) > 0 {
			query = args[0]
		}

		apps := proxy.HFAppNames()
		results, err := client.SearchModels(query, 20, apps)
		if err != nil {
			ui.Fatal("Failed to search: %v", err)
		}

		if len(results) == 0 {
			if query != "" {
				fmt.Printf("No results found for '%s'\n", query)
			} else {
				fmt.Println("No models found")
			}
			fmt.Println()
			fmt.Println("Tips:")
			fmt.Println("  Try a different search term")
			fmt.Println("  Check spelling")
			fmt.Printf("  Browse Hugging Face: https://huggingface.co/models?apps=%s&sort=trending\n", strings.Join(apps, ","))
			return
		}

		// Build set of installed models (by user/repo).
		// Errors are ignored since install indicators are non-critical UI hints.
		installed := make(map[string]bool)
		resolver := proxy.NewModelResolver()
		if downloaded, err := resolver.ListDownloadedModels(); err == nil {
			for _, m := range downloaded {
				installed[m.User+"/"+m.Repo] = true
			}
		}

		table := ui.NewTable().
			Indent(0).
			AddColumn("MODEL", 0, ui.AlignLeft).
			AddColumn("DOWNLOADS", 10, ui.AlignRight).
			AddColumn("LIKES", 8, ui.AlignRight)

		for _, result := range results {
			indicator := "○"
			if installed[result.ID] {
				indicator = "✓"
			}
			modelName := indicator + " " + result.ID
			if result.Gated {
				modelName += " (gated)"
			}
			table.AddRow(modelName, ui.FormatNumber(result.Downloads), ui.FormatNumber(result.Likes))
		}

		fmt.Print(table.Render())
		fmt.Println("\n✓ = installed")
		if query != "" {
			fmt.Printf("%d results for \"%s\"\n", len(results), query)
		} else {
			fmt.Printf("%d trending models\n", len(results))
		}
	},
}

func init() {
	rootCmd.AddCommand(searchCmd)
}
