package cmd

import (
	"fmt"

	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/proxy"
	"github.com/nchapman/lleme/internal/selfupdate"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var updateCmd = &cobra.Command{
	Use:     "update",
	Short:   "Update lleme and/or llama.cpp",
	GroupID: "config",
	Run:     runUpdateAll,
}

var updateLlamaCmd = &cobra.Command{
	Use:   "llama.cpp",
	Short: "Update llama.cpp to the latest version",
	Run:   runUpdateLlama,
}

var updateSelfCmd = &cobra.Command{
	Use:   "self",
	Short: "Update lleme to the latest version",
	Run:   runUpdateSelf,
}

var forceUpdate bool

func init() {
	updateCmd.PersistentFlags().BoolVarP(&forceUpdate, "force", "f", false, "Skip confirmation")

	rootCmd.AddCommand(updateCmd)
	updateCmd.AddCommand(updateLlamaCmd)
	updateCmd.AddCommand(updateSelfCmd)
}

func runUpdateAll(cmd *cobra.Command, args []string) {
	fmt.Println("Checking for updates...")
	fmt.Println()

	// Check lleme version
	llemeInstalled := selfupdate.GetInstalledVersion()
	llemeLatest, llemeErr := selfupdate.GetLatestVersion()
	llemeNeedsUpdate := llemeErr == nil && llemeInstalled != llemeLatest

	// Check llama.cpp version
	llamaInstalled, llamaErr := llama.GetInstalledVersion()
	llamaRelease, llamaFetchErr := llama.GetLatestVersion()

	llamaInstalledStr := "Not installed"
	if llamaInstalled != nil {
		llamaInstalledStr = llamaInstalled.TagName
	}

	llamaLatestStr := "Unknown"
	if llamaRelease != nil {
		llamaLatestStr = llamaRelease.TagName
	}

	llamaNeedsUpdate := llamaUpdateAvailable(llamaInstalled, llamaRelease, llamaFetchErr)

	// Display status
	printComponentStatus("lleme", llemeInstalled, llemeLatest, llemeErr, llemeNeedsUpdate)
	printComponentStatus("llama.cpp", llamaInstalledStr, llamaLatestStr, llamaFetchErr, llamaNeedsUpdate)

	if llamaErr != nil {
		ui.PrintError("Failed to check llama.cpp installed version: %v", llamaErr)
	}

	if !llemeNeedsUpdate && !llamaNeedsUpdate {
		fmt.Println("Everything is up to date")
		return
	}

	// Build update message
	var updates []string
	if llemeNeedsUpdate {
		updates = append(updates, fmt.Sprintf("lleme to %s", llemeLatest))
	}
	if llamaNeedsUpdate {
		updates = append(updates, fmt.Sprintf("llama.cpp to %s", llamaLatestStr))
	}

	if !forceUpdate {
		prompt := fmt.Sprintf("Update %s?", joinWithAnd(updates))
		if !ui.PromptYesNo(prompt, false) {
			fmt.Println(ui.Muted("Cancelled"))
			return
		}
	}
	fmt.Println()

	// Update lleme if needed
	if llemeNeedsUpdate {
		updateLleme(selfupdate.DetectInstallMethod())
		fmt.Println()
	}

	// Update llama.cpp if needed
	if llamaNeedsUpdate {
		updateLlamaCpp()
	}

	restartServerIfRunning()
}

func llamaUpdateAvailable(installed *llama.VersionInfo, latest *llama.Release, fetchErr error) bool {
	if fetchErr != nil || latest == nil {
		return false
	}
	return installed == nil || installed.TagName != latest.TagName
}

func printComponentStatus(name, installed, available string, fetchErr error, needsUpdate bool) {
	fmt.Printf("  %s:\n", name)
	fmt.Printf("    %-12s %s\n", "Installed", installed)
	if fetchErr != nil {
		fmt.Printf("    %-12s %s\n", "Available", ui.Muted("Failed to check"))
	} else if needsUpdate {
		fmt.Printf("    %-12s %s\n", "Available", available)
	} else {
		fmt.Printf("    %-12s %s %s\n", "Available", available, ui.Success(ui.IconCheck))
	}
	fmt.Println()
}

func runUpdateLlama(cmd *cobra.Command, args []string) {
	fmt.Println("Checking for llama.cpp updates...")
	fmt.Println()

	installed, err := llama.GetInstalledVersion()
	if err != nil {
		ui.Fatal("Failed to check installed version: %v", err)
	}

	release, err := llama.GetLatestVersion()
	if err != nil {
		ui.Fatal("Failed to get latest release: %v", err)
	}

	currentVersion := "Not installed"
	if installed != nil {
		currentVersion = installed.TagName
	}

	fmt.Printf("  %-12s %s\n", "Installed", currentVersion)
	fmt.Printf("  %-12s %s\n", "Available", release.TagName)
	fmt.Println()

	if installed != nil && installed.TagName == release.TagName {
		fmt.Println("llama.cpp is already up to date")
		return
	}

	if !forceUpdate {
		if !ui.PromptYesNo(fmt.Sprintf("Update to %s?", release.TagName), false) {
			fmt.Println(ui.Muted("Cancelled"))
			return
		}
	}

	fmt.Println()
	updateLlamaCpp()
	restartServerIfRunning()
}

func runUpdateSelf(cmd *cobra.Command, args []string) {
	fmt.Println("Checking for lleme updates...")
	fmt.Println()

	installed := selfupdate.GetInstalledVersion()
	latest, err := selfupdate.GetLatestVersion()
	if err != nil {
		ui.Fatal("Failed to get latest release: %v", err)
	}

	fmt.Printf("  %-12s %s\n", "Installed", installed)
	fmt.Printf("  %-12s %s\n", "Available", latest)
	fmt.Println()

	if installed == latest {
		fmt.Println("lleme is already up to date")
		return
	}

	method := selfupdate.DetectInstallMethod()
	if method == selfupdate.InstallUnknown {
		fmt.Println(selfupdate.ManualUpdateInstructions())
		return
	}

	if !forceUpdate {
		if !ui.PromptYesNo(fmt.Sprintf("Update to %s?", latest), false) {
			fmt.Println(ui.Muted("Cancelled"))
			return
		}
	}

	fmt.Println()
	updateLleme(method)
	restartServerIfRunning()
}

func updateLleme(method selfupdate.InstallMethod) {
	if method == selfupdate.InstallUnknown {
		fmt.Println(selfupdate.ManualUpdateInstructions())
		return
	}

	fmt.Println("Updating lleme...")
	if err := selfupdate.Update(method); err != nil {
		ui.Fatal("Failed to update lleme: %v", err)
	}
	fmt.Println("lleme updated successfully")
}

func updateLlamaCpp() {
	version, err := llama.InstallLatest(func(msg string) { fmt.Println(msg) })
	if err != nil {
		ui.Fatal("Failed to install llama.cpp: %v", err)
	}
	fmt.Printf("Updated to llama.cpp %s\n", version.TagName)
}

func joinWithAnd(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	case 2:
		return items[0] + " and " + items[1]
	default:
		return items[0] + ", " + joinWithAnd(items[1:])
	}
}

func restartServerIfRunning() {
	if !proxy.IsProxyRunning() {
		return
	}

	fmt.Println()
	fmt.Println("Restarting server to apply updates...")
	stopped, err := stopServer()
	if err != nil {
		ui.PrintError("Failed to stop server: %v", err)
		return
	}
	if stopped {
		fmt.Println("Stopped server")
	}
	// startServerDetached executes the binary from disk, which is now the updated version
	startServerDetached()
}
