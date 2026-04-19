package cmd

import (
	"fmt"

	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/proxy"
	"github.com/nchapman/lleme/internal/selfupdate"
	"github.com/nchapman/lleme/internal/swiftlm"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var updateCmd = &cobra.Command{
	Use:     "update",
	Short:   "Update lleme and/or backend runtimes",
	GroupID: "config",
	Run:     runUpdateAll,
}

var updateLlamaCmd = &cobra.Command{
	Use:   "llama.cpp",
	Short: "Update llama.cpp to the latest version",
	Run:   runUpdateLlama,
}

var updateSwiftLMCmd = &cobra.Command{
	Use:   "swiftlm",
	Short: "Update SwiftLM to the latest version (macOS Apple Silicon only)",
	Run:   runUpdateSwiftLM,
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
	updateCmd.AddCommand(updateSwiftLMCmd)
	updateCmd.AddCommand(updateSelfCmd)
}

func runUpdateAll(cmd *cobra.Command, args []string) {
	fmt.Println("Checking for updates...")
	fmt.Println()

	checks := collectUpdateChecks()
	renderUpdateChecks(checks)

	var updates []string
	for _, c := range checks {
		if c.needsUpdate {
			updates = append(updates, fmt.Sprintf("%s to %s", c.name, c.latestStr))
		}
	}
	if len(updates) == 0 {
		fmt.Println("Everything is up to date")
		return
	}

	if !forceUpdate {
		if !ui.PromptYesNo(fmt.Sprintf("Update %s?", joinWithAnd(updates)), false) {
			fmt.Println(ui.Muted("Cancelled"))
			return
		}
	}
	fmt.Println()

	for _, c := range checks {
		if !c.needsUpdate {
			continue
		}
		c.apply()
		fmt.Println()
	}

	restartServerIfRunning()
}

// updateCheck bundles everything runUpdateAll needs about one component
// (lleme / llama.cpp / SwiftLM) in a single shape, so the main flow stays
// declarative and doesn't branch per component.
type updateCheck struct {
	name         string
	installedStr string
	latestStr    string
	fetchErr     error
	readErr      error // surfaced after status rendering
	needsUpdate  bool
	skipStatus   bool   // SwiftLM on unsupported platforms
	readErrLabel string // what to say when readErr != nil
	apply        func()
}

func collectUpdateChecks() []updateCheck {
	return []updateCheck{
		checkLlemeUpdate(),
		checkLlamaCppUpdate(),
		checkSwiftLMUpdate(),
	}
}

func renderUpdateChecks(checks []updateCheck) {
	for _, c := range checks {
		if c.skipStatus {
			continue
		}
		printComponentStatus(c.name, c.installedStr, c.latestStr, c.fetchErr, c.needsUpdate)
	}
	for _, c := range checks {
		if c.readErr != nil && c.readErrLabel != "" {
			ui.PrintError("%s: %v", c.readErrLabel, c.readErr)
		}
	}
}

func checkLlemeUpdate() updateCheck {
	installed := selfupdate.GetInstalledVersion()
	latest, err := selfupdate.GetLatestVersion()
	needs := err == nil && installed != latest
	return updateCheck{
		name:         "lleme",
		installedStr: installed,
		latestStr:    latest,
		fetchErr:     err,
		needsUpdate:  needs,
		apply:        func() { updateLleme(selfupdate.DetectInstallMethod()) },
	}
}

func checkLlamaCppUpdate() updateCheck {
	installed, readErr := llama.GetInstalledVersion()
	release, fetchErr := llama.GetLatestVersion()
	installedStr := "Not installed"
	if installed != nil {
		installedStr = installed.TagName
	}
	latestStr := "Unknown"
	if release != nil {
		latestStr = release.TagName
	}
	return updateCheck{
		name:         "llama.cpp",
		installedStr: installedStr,
		latestStr:    latestStr,
		fetchErr:     fetchErr,
		readErr:      readErr,
		readErrLabel: "Failed to check llama.cpp installed version",
		needsUpdate:  llamaUpdateAvailable(installed, release, fetchErr),
		apply:        updateLlamaCpp,
	}
}

func checkSwiftLMUpdate() updateCheck {
	if !swiftlm.IsSupported() {
		return updateCheck{name: "SwiftLM", skipStatus: true}
	}
	installed, readErr := swiftlm.GetInstalledVersion()
	release, fetchErr := swiftlm.GetLatestVersion()
	installedStr := "Not installed"
	if installed != nil {
		installedStr = installed.TagName
	}
	latestStr := "Unknown"
	if release != nil {
		latestStr = release.TagName
	}
	needs := fetchErr == nil && release != nil && (installed == nil || installed.TagName != release.TagName)
	return updateCheck{
		name:         "SwiftLM",
		installedStr: installedStr,
		latestStr:    latestStr,
		fetchErr:     fetchErr,
		readErr:      readErr,
		readErrLabel: "Failed to check SwiftLM installed version",
		needsUpdate:  needs,
		apply:        updateSwiftLM,
	}
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

func runUpdateSwiftLM(cmd *cobra.Command, args []string) {
	if !swiftlm.IsSupported() {
		ui.Fatal("SwiftLM (MLX) requires macOS on Apple Silicon")
	}
	fmt.Println("Checking for SwiftLM updates...")
	fmt.Println()

	installed, err := swiftlm.GetInstalledVersion()
	if err != nil {
		ui.Fatal("Failed to check installed version: %v", err)
	}

	release, err := swiftlm.GetLatestVersion()
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
		fmt.Println("SwiftLM is already up to date")
		return
	}

	if !forceUpdate {
		if !ui.PromptYesNo(fmt.Sprintf("Update to %s?", release.TagName), false) {
			fmt.Println(ui.Muted("Cancelled"))
			return
		}
	}

	fmt.Println()
	updateSwiftLM()
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

func updateSwiftLM() {
	version, err := swiftlm.InstallLatest(func(msg string) { fmt.Println(msg) })
	if err != nil {
		ui.Fatal("Failed to install SwiftLM: %v", err)
	}
	fmt.Printf("Updated to SwiftLM %s\n", version.TagName)
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
