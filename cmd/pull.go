package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var pullCmd = &cobra.Command{
	Use:     "pull <user/repo>[:quant]",
	Short:   "Download a model from Hugging Face",
	GroupID: "model",
	Long: `Download a model from Hugging Face.

Examples:
  lleme pull unsloth/Llama-3.2-1B-Instruct-GGUF           # Download default quant
  lleme pull unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0      # Download specific quant`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelRef := args[0]

		user, repo, quant, err := parseModelRef(modelRef)
		if err != nil {
			ui.Fatal("%s", err)
		}

		cfg, err := config.Load()
		if err != nil {
			ui.Fatal("Failed to load config: %v", err)
		}

		client := hf.NewClient(cfg)

		modelInfo, err := client.GetModel(user, repo)
		if err != nil {
			handleModelError(err, user, repo)
			ui.ExitFunc(1)
			return
		}

		if bool(modelInfo.Gated) && !hf.HasToken(cfg) {
			ui.PrintError("Authentication required")
			fmt.Printf("\nThe repository '%s/%s' requires authentication.\n\n", user, repo)
			fmt.Println("To access gated models, provide a Hugging Face token:")
			fmt.Println("  1. Get a token at https://huggingface.co/settings/tokens")
			fmt.Println("  2. Run: hf auth login")
			fmt.Println("     Or set: export HF_TOKEN=hf_xxxxx")
			ui.ExitFunc(1)
			return
		}

		files, err := client.ListFiles(user, repo, "main")
		if err != nil {
			ui.Fatal("Failed to list files: %v", err)
		}

		quants := hf.ExtractQuantizations(files)
		if len(quants) == 0 {
			ui.PrintError("No GGUF files found")
			fmt.Printf("\nThe repository '%s/%s' exists but contains no GGUF files.\n", user, repo)
			ui.ExitFunc(1)
			return
		}

		// Find the quantization to use
		var selectedQuant hf.Quantization
		if quant == "" {
			quant = hf.GetBestQuantization(quants)
			selectedQuant, _ = hf.FindQuantization(quants, quant)
		} else {
			var found bool
			selectedQuant, found = hf.FindQuantization(quants, quant)
			if !found {
				ui.PrintError("Quantization '%s' not found", quant)
				fmt.Println("\nAvailable quantizations:")
				client.FetchFolderQuantSizes(user, repo, "main", quants)
				for _, q := range hf.SortQuantizations(quants) {
					fmt.Printf("  • %s (%s)\n", q.Name, ui.FormatBytes(q.Size))
				}
				ui.ExitFunc(1)
				return
			}
		}

		// Check if local files are up to date with remote manifest
		upToDate, saveManifest, _, manifestJSON, err := hf.CheckForUpdates(client, user, repo, selectedQuant)
		if err != nil {
			ui.Fatal("%v", err)
		}
		if upToDate {
			if saveManifest {
				// Legacy model without manifest - save it now
				manifestPath := hf.GetManifestFilePath(user, repo, quant)
				if err := os.WriteFile(manifestPath, manifestJSON, 0644); err != nil {
					ui.Fatal("Failed to save manifest: %v", err)
				}
			}
			// Find the actual model path (handles both single and split files)
			modelPath := hf.FindModelFile(user, repo, quant)
			if modelPath == "" {
				modelPath = hf.GetModelFilePath(user, repo, quant) // Fallback for display
			}
			fmt.Printf("Model is up to date: %s\n", ui.Bold(modelPath))
			return
		}

		result, err := pullModelWithProgress(client, user, repo, selectedQuant)
		if err != nil {
			ui.Fatal("%v", err)
		}

		modelName := hf.FormatModelName(user, repo, selectedQuant.Name)
		if result.IsVision {
			fmt.Printf("Pulled %s (vision model)\n", modelName)
		} else {
			fmt.Printf("Pulled %s\n", modelName)
		}
	},
}

// pullModelWithProgress wraps hf.PullModel with progress bar display.
func pullModelWithProgress(client *hf.Client, user, repo string, quant hf.Quantization) (*hf.PullResult, error) {
	info, manifest, manifestJSON, err := hf.GetManifestInfo(client, user, repo, quant)
	if err != nil {
		return nil, err
	}

	modelName := ui.Keyword(hf.FormatModelName(user, repo, quant.Name))
	if info.IsVision {
		fmt.Printf("Pulling %s (%s + %s mmproj)\n",
			modelName,
			ui.FormatBytes(info.GGUFSize),
			ui.FormatBytes(info.MMProjSize))
	} else {
		fmt.Printf("Pulling %s (%s)\n", modelName, ui.FormatBytes(info.GGUFSize))
	}

	opts := &hf.PullOptions{
		Manifest:     manifest,
		ManifestJSON: manifestJSON,
	}

	return hf.PullModelWithProgressFactory(client, user, repo, quant, opts, newProgressBar)
}

// newProgressBar creates a new progress bar that implements hf.ProgressDisplay.
func newProgressBar() hf.ProgressDisplay {
	return ui.NewProgressBar()
}

func parseModelRef(ref string) (user, repo, quant string, err error) {
	parts := strings.Split(ref, ":")
	if len(parts) > 2 {
		return "", "", "", fmt.Errorf("invalid model reference: %s", ref)
	}

	mainRef := parts[0]
	quantPart := ""
	if len(parts) == 2 {
		quantPart = parts[1]
	}

	repoParts := strings.Split(mainRef, "/")
	if len(repoParts) != 2 {
		return "", "", "", fmt.Errorf("model reference must be in format user/repo: %s", ref)
	}

	return repoParts[0], repoParts[1], quantPart, nil
}

func handleModelError(err error, user, repo string) {
	errStr := err.Error()

	if strings.Contains(errStr, "404") {
		fmt.Printf("%s Model not found\n", ui.ErrorMsg("Error:"))
		fmt.Printf("\nCould not find '%s/%s' on Hugging Face.\n\n", user, repo)
		fmt.Println("Tips:")
		fmt.Println("  • Check the spelling of the repository name")
		fmt.Println("  • Use 'lleme search <query>' to find models")
	} else {
		fmt.Printf("%s %v\n", ui.ErrorMsg("Error:"), err)
	}
}

func init() {
	rootCmd.AddCommand(pullCmd)
}
