package cmd

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/config"
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
		modelsDir := config.ModelsPath()

		var models []ModelInfo
		var totalSize int64
		seenSplitDirs := make(map[string]bool)

		err := filepath.WalkDir(modelsDir, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return fmt.Errorf("walk %s: %w", path, err)
			}

			if d.IsDir() {
				return nil
			}

			if filepath.Ext(d.Name()) != ".gguf" {
				return nil
			}
			if hf.IsMMProjFileName(d.Name()) {
				return nil
			}

			relPath, err := filepath.Rel(modelsDir, path)
			if err != nil {
				return fmt.Errorf("relative path of %s: %w", path, err)
			}

			parts := strings.Split(relPath, string(filepath.Separator))
			if len(parts) < 3 {
				return nil
			}

			user := parts[0]
			repo := parts[1]
			var quant string
			var modelSize int64

			// Check if this is a split file (in a quant subdirectory)
			// Structure: user/repo/quant/model-00001-of-NNNNN.gguf
			if len(parts) == 4 && hf.SplitFilePattern.MatchString(d.Name()) {
				quant = parts[2]
				splitDirKey := filepath.Join(user, repo, quant)

				// Only add the first split file we encounter for this quant
				if seenSplitDirs[splitDirKey] {
					return nil
				}
				seenSplitDirs[splitDirKey] = true

				// Calculate total size of all split files. Read errors here are
				// non-fatal: we just report whatever sizes we could sum.
				splitDir := filepath.Dir(path)
				entries, readErr := os.ReadDir(splitDir)
				if readErr != nil {
					return fmt.Errorf("read split dir %s: %w", splitDir, readErr)
				}
				for _, entry := range entries {
					if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".gguf") {
						continue
					}
					if info, err := entry.Info(); err == nil {
						modelSize += info.Size()
					}
				}
			} else {
				// Standard single-file model: user/repo/quant.gguf
				quant = strings.TrimSuffix(d.Name(), ".gguf")
				info, err := d.Info()
				if err != nil {
					return fmt.Errorf("stat %s: %w", path, err)
				}
				modelSize = info.Size()
			}

			lastUsed := hf.GetLastUsed(user, repo, quant)
			if lastUsed.IsZero() {
				info, _ := d.Info()
				if info != nil {
					lastUsed = info.ModTime() // Fall back to download time
				} else {
					lastUsed = time.Now()
				}
			}

			models = append(models, ModelInfo{
				User:     user,
				Repo:     repo,
				Quant:    quant,
				Size:     modelSize,
				LastUsed: lastUsed,
			})

			totalSize += modelSize

			return nil
		})

		if err != nil {
			ui.Fatal("Failed to list models: %v", err)
		}

		if len(models) == 0 {
			fmt.Println(ui.Muted("No models downloaded yet"))
			fmt.Println()
			fmt.Println("Use 'lleme pull <user/repo>' to download a model")
			return
		}

		// Sort by most recently used
		sort.Slice(models, func(i, j int) bool {
			return models[i].LastUsed.After(models[j].LastUsed)
		})

		table := ui.NewTable().
			Indent(0).
			AddColumn("MODEL", 0, ui.AlignLeft).
			AddColumn("QUANT", 0, ui.AlignLeft).
			AddColumn("SIZE", 10, ui.AlignRight).
			AddColumn("LAST USED", 12, ui.AlignRight)

		for _, m := range models {
			modelRef := fmt.Sprintf("%s/%s", m.User, m.Repo)
			table.AddRow(modelRef, m.Quant, ui.FormatBytes(m.Size), formatTime(m.LastUsed))
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
