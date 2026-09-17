package selfupdate

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/version"
)

const llemeRepo = "nchapman/lleme"

// apiBase is a var (not a const) so tests can point version checks at an
// httptest server.
var apiBase = "https://api.github.com/repos/" + llemeRepo

type Release struct {
	TagName string `json:"tag_name"`
	Name    string `json:"name"`
}

type InstallMethod string

const (
	InstallHomebrew InstallMethod = "homebrew"
	InstallGo       InstallMethod = "go"
	InstallUnknown  InstallMethod = "unknown"
)

func GetInstalledVersion() string {
	return version.Version
}

func GetLatestVersion() (string, error) {
	url := apiBase + "/releases/latest"

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Accept", "application/vnd.github.v3+json")
	req.Header.Set("User-Agent", version.UserAgent())

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to fetch release: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var release Release
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	return strings.TrimPrefix(release.TagName, "v"), nil
}

func DetectInstallMethod() InstallMethod {
	execPath, err := os.Executable()
	if err != nil {
		return InstallUnknown
	}
	execPath, _ = filepath.EvalSymlinks(execPath)

	// Check if running from Homebrew's Cellar
	if out, err := exec.Command("brew", "--prefix").Output(); err == nil {
		brewPrefix := strings.TrimSpace(string(out))
		cellarPath := filepath.Join(brewPrefix, "Cellar", "lleme")
		if strings.HasPrefix(execPath, cellarPath) {
			return InstallHomebrew
		}
	}

	// Check if binary path contains go/bin (installed via go install)
	if strings.Contains(execPath, filepath.Join("go", "bin")) {
		return InstallGo
	}

	return InstallUnknown
}

func Update(method InstallMethod) error {
	switch method {
	case InstallHomebrew:
		// Update Homebrew formulas first to ensure we get the latest version
		updateCmd := exec.Command("brew", "update")
		updateCmd.Stdout = os.Stdout
		updateCmd.Stderr = os.Stderr
		if err := updateCmd.Run(); err != nil {
			return fmt.Errorf("brew update failed: %w", err)
		}

		upgradeCmd := exec.Command("brew", "upgrade", "lleme")
		upgradeCmd.Stdout = os.Stdout
		upgradeCmd.Stderr = os.Stderr
		if err := upgradeCmd.Run(); err != nil {
			return fmt.Errorf("brew upgrade failed: %w", err)
		}
		return nil

	case InstallGo:
		cmd := exec.Command("go", "install", "github.com/nchapman/lleme@latest")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("go install failed: %w", err)
		}
		return nil

	default:
		return fmt.Errorf("unknown install method")
	}
}

func ManualUpdateInstructions() string {
	return `Could not detect how lleme was installed.

To update manually, use one of the following:

  Homebrew:    brew install nchapman/tap/lleme
  Go:          go install github.com/nchapman/lleme@latest

Or download the latest release from:
  https://github.com/nchapman/lleme/releases`
}
