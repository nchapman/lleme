package cmd

import (
	"fmt"
	"sort"
	"time"

	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:     "list",
	Aliases: []string{"ls"},
	Short:   "List downloaded models",
	GroupID: "model",
	Run: func(cmd *cobra.Command, args []string) {
		models, err := hf.ListLocalModels()
		if err != nil {
			ui.Fatal("Failed to list models: %v", err)
		}

		if len(models) == 0 {
			fmt.Println(ui.Muted("No models downloaded yet"))
			fmt.Println()
			fmt.Println("Use 'lleme pull <user/repo>' to download a model")
			return
		}

		// Most-recently-used first.
		sort.Slice(models, func(i, j int) bool {
			return models[i].LastUsed.After(models[j].LastUsed)
		})

		table := ui.NewTable().
			Indent(0).
			AddColumn("MODEL", 0, ui.AlignLeft).
			AddColumn("QUANT", 0, ui.AlignLeft).
			AddColumn("BACKEND", 0, ui.AlignLeft).
			AddColumn("SIZE", 10, ui.AlignRight).
			AddColumn("LAST USED", 12, ui.AlignRight)

		var totalSize int64
		for _, m := range models {
			lastUsed := hf.GetLastUsed(m.User, m.Repo, m.Quant)
			if lastUsed.IsZero() {
				lastUsed = m.LastUsed
			}
			modelRef := fmt.Sprintf("%s/%s", m.User, m.Repo)
			table.AddRow(modelRef, m.Quant, m.Backend, ui.FormatBytes(m.Size), formatTime(lastUsed))
			totalSize += m.Size
		}

		fmt.Print(table.Render())
		fmt.Println()
		fmt.Printf("%d models, %s total\n", len(models), ui.FormatBytes(totalSize))
	},
}

func formatTime(t time.Time) string {
	now := time.Now()
	diff := now.Sub(t)

	switch {
	case diff < time.Hour:
		return "Just now"
	case diff < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(diff.Hours()))
	case diff < 30*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(diff.Hours()/24))
	default:
		return t.Format("Jan 2006")
	}
}

func init() {
	rootCmd.AddCommand(listCmd)
}
