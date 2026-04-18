package cmd

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var (
	rmForce            bool
	rmOlderThan        string
	rmLargerThan       string
	rmPartialDownloads bool
)

var removeCmd = &cobra.Command{
	Use:     "remove [pattern]",
	Aliases: []string{"rm"},
	Short:   "Remove downloaded models",
	GroupID: "model",
	Long: `Remove downloaded models by name, pattern, or filter.

Examples:
  lleme remove user/repo:quant       Remove specific model
  lleme remove user/repo             Remove all quants of a model
  lleme remove user/*                Remove all models from user
  lleme remove *                     Remove all models
  lleme remove --older-than 30d      Remove models unused for 30 days
  lleme remove --larger-than 10GB    Remove models larger than 10GB
  lleme remove --partial-downloads   Remove incomplete downloads
  lleme remove user/* --older-than 7d  Combine pattern with filter`,
	Args: cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// Handle --partial-downloads flag
		if rmPartialDownloads {
			count, err := hf.CleanupPartialFiles()
			if err != nil {
				ui.Fatal("Failed to clean up partial downloads: %v", err)
			}
			if count == 0 {
				fmt.Println("No partial downloads found")
			} else {
				fmt.Printf("Removed %d partial download(s)\n", count)
			}
			return
		}

		pattern := ""
		if len(args) > 0 {
			pattern = args[0]
		}

		// Must have pattern or filters
		if pattern == "" && rmOlderThan == "" && rmLargerThan == "" {
			ui.PrintError("Specify a model pattern or use --older-than/--larger-than")
			fmt.Println()
			fmt.Println("Examples:")
			fmt.Println("  lleme remove user/repo:quant")
			fmt.Println("  lleme remove --older-than 30d")
			fmt.Println("  lleme remove --larger-than 10GB")
			os.Exit(1)
		}

		// Parse filters
		var olderThan time.Duration
		var largerThan int64
		var err error

		if rmOlderThan != "" {
			olderThan, err = parseDuration(rmOlderThan)
			if err != nil {
				ui.Fatal("Invalid --older-than value: %v", err)
			}
		}

		if rmLargerThan != "" {
			largerThan, err = parseSize(rmLargerThan)
			if err != nil {
				ui.Fatal("Invalid --larger-than value: %v", err)
			}
		}

		// Default pattern to * when using filters
		if pattern == "" {
			pattern = "*"
		}

		// Find matching models
		models, err := findModels(pattern, olderThan, largerThan)
		if err != nil {
			ui.Fatal("%v", err)
		}

		if len(models) == 0 {
			fmt.Println("No models match the criteria")
			return
		}

		// Calculate total size
		var totalSize int64
		for _, m := range models {
			totalSize += m.Size
		}

		if !rmForce {
			fmt.Println("Models to remove:")
			fmt.Println()
			for _, m := range models {
				fmt.Printf("  %s (%s)\n", hf.FormatModelName(m.User, m.Repo, m.Quant), ui.FormatBytes(m.Size))
			}
			fmt.Println()

			var prompt string
			if len(models) == 1 {
				m := models[0]
				prompt = fmt.Sprintf("Remove %s (%s)?", hf.FormatModelName(m.User, m.Repo, m.Quant), ui.FormatBytes(m.Size))
			} else {
				prompt = fmt.Sprintf("Remove %d model(s), %s total?", len(models), ui.FormatBytes(totalSize))
			}

			if !ui.PromptYesNo(prompt, false) {
				fmt.Println(ui.Muted("Cancelled"))
				return
			}
		}

		// Remove models
		removed := 0
		var freedSize int64
		for _, m := range models {
			// Find the actual model path (handles both single and split files)
			modelPath := hf.FindModelFile(m.User, m.Repo, m.Quant)
			if modelPath == "" {
				modelPath = hf.GetModelFilePath(m.User, m.Repo, m.Quant)
			}

			// Check if this is a split file (stored in a quant subdirectory)
			splitDir := hf.GetSplitModelDir(m.User, m.Repo, m.Quant)
			if info, err := os.Stat(splitDir); err == nil && info.IsDir() {
				// Remove the entire split directory
				if err := os.RemoveAll(splitDir); err != nil {
					ui.PrintError("Failed to remove %s: %v", hf.FormatModelName(m.User, m.Repo, m.Quant), err)
					continue
				}
			} else {
				// Single file - remove it directly
				if err := os.Remove(modelPath); err != nil {
					ui.PrintError("Failed to remove %s: %v", hf.FormatModelName(m.User, m.Repo, m.Quant), err)
					continue
				}
			}
			freedSize += m.Size

			// Also remove associated files (manifest, mmproj, metadata)
			// These may not exist; errors are expected and safe to ignore
			os.Remove(hf.GetManifestFilePath(m.User, m.Repo, m.Quant))
			os.Remove(hf.GetMMProjFilePath(m.User, m.Repo, m.Quant))

			// Clean up empty directories
			modelDir := hf.GetModelPath(m.User, m.Repo)
			cleanEmptyDir(modelDir)
			userDir := filepath.Dir(modelDir)
			cleanEmptyDir(userDir)

			removed++
		}

		if removed == 1 {
			m := models[0]
			fmt.Printf("Removed %s\n", hf.FormatModelName(m.User, m.Repo, m.Quant))
		} else {
			fmt.Printf("Removed %d models, %s freed\n", removed, ui.FormatBytes(freedSize))
		}
	},
}

// findModels returns models matching the pattern and filters
func findModels(pattern string, olderThan time.Duration, largerThan int64) ([]ModelInfo, error) {
	return findModelsInDir(config.ModelsPath(), pattern, olderThan, largerThan)
}

// findModelsInDir is the testable version of findModels
func findModelsInDir(modelsDir, pattern string, olderThan time.Duration, largerThan int64) ([]ModelInfo, error) {
	var models []ModelInfo
	seenSplitDirs := make(map[string]bool)

	// Convert glob pattern to regex
	regexPattern := globToRegex(pattern)
	re, err := regexp.Compile("^" + regexPattern + "$")
	if err != nil {
		return nil, fmt.Errorf("invalid pattern: %s", pattern)
	}

	err = filepath.WalkDir(modelsDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() || filepath.Ext(d.Name()) != ".gguf" {
			return nil
		}

		relPath, err := filepath.Rel(modelsDir, path)
		if err != nil {
			return err
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

			// Calculate total size of all split files
			splitDir := filepath.Dir(path)
			entries, _ := os.ReadDir(splitDir)
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
				return err
			}
			modelSize = info.Size()
		}

		// Check pattern match
		fullName := hf.FormatModelName(user, repo, quant)
		repoName := fmt.Sprintf("%s/%s", user, repo)

		// Try matching full name, repo name, or repo/* pattern
		if !re.MatchString(fullName) && !re.MatchString(repoName) {
			return nil
		}

		// Get last used time
		lastUsed := hf.GetLastUsed(user, repo, quant)
		if lastUsed.IsZero() {
			info, _ := d.Info()
			if info != nil {
				lastUsed = info.ModTime()
			} else {
				lastUsed = time.Now()
			}
		}

		// Apply filters
		if olderThan > 0 {
			if time.Since(lastUsed) < olderThan {
				return nil
			}
		}

		if largerThan > 0 {
			if modelSize < largerThan {
				return nil
			}
		}

		models = append(models, ModelInfo{
			User:     user,
			Repo:     repo,
			Quant:    quant,
			Size:     modelSize,
			LastUsed: lastUsed,
		})

		return nil
	})

	return models, err
}

// globToRegex converts a glob pattern to a regex pattern
func globToRegex(glob string) string {
	result := regexp.QuoteMeta(glob)
	result = strings.ReplaceAll(result, `\*`, ".*")
	result = strings.ReplaceAll(result, `\?`, ".")
	return result
}

// parseDuration parses a duration string like "30d", "7d", "1w"
func parseDuration(s string) (time.Duration, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	if len(s) < 2 {
		return 0, fmt.Errorf("invalid duration: %s", s)
	}

	unit := s[len(s)-1]
	valueStr := s[:len(s)-1]
	value, err := strconv.Atoi(valueStr)
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("invalid duration: %s", s)
	}

	switch unit {
	case 'd':
		return time.Duration(value) * 24 * time.Hour, nil
	case 'w':
		return time.Duration(value) * 7 * 24 * time.Hour, nil
	case 'h':
		return time.Duration(value) * time.Hour, nil
	default:
		return 0, fmt.Errorf("invalid duration unit: %c (use d, w, or h)", unit)
	}
}

// parseSize parses a size string like "10GB", "500MB"
func parseSize(s string) (int64, error) {
	s = strings.TrimSpace(strings.ToUpper(s))
	if len(s) < 2 {
		return 0, fmt.Errorf("invalid size: %s", s)
	}

	// Find where the number ends
	i := 0
	for i < len(s) && (s[i] >= '0' && s[i] <= '9' || s[i] == '.') {
		i++
	}

	valueStr := s[:i]
	unit := s[i:]

	value, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid size: %s", s)
	}

	switch unit {
	case "B":
		return int64(value), nil
	case "KB", "K":
		return int64(value * 1024), nil
	case "MB", "M":
		return int64(value * 1024 * 1024), nil
	case "GB", "G":
		return int64(value * 1024 * 1024 * 1024), nil
	case "TB", "T":
		return int64(value * 1024 * 1024 * 1024 * 1024), nil
	default:
		return 0, fmt.Errorf("invalid size unit: %s (use B, KB, MB, GB, TB)", unit)
	}
}

// cleanEmptyDir removes a directory if it's empty
func cleanEmptyDir(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	if len(entries) == 0 {
		os.Remove(dir)
	}
}

func init() {
	removeCmd.Flags().BoolVarP(&rmForce, "force", "f", false, "Skip confirmation")
	removeCmd.Flags().StringVar(&rmOlderThan, "older-than", "", "Remove models not used in this duration (e.g., 24h, 7d, 4w)")
	removeCmd.Flags().StringVar(&rmLargerThan, "larger-than", "", "Remove models larger than this size (e.g., 500MB, 10GB)")
	removeCmd.Flags().BoolVar(&rmPartialDownloads, "partial-downloads", false, "Remove incomplete/interrupted downloads")
	rootCmd.AddCommand(removeCmd)
}
