package llama

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/fileutil"
	"github.com/nchapman/lleme/internal/version"
)

const vulkanCheckTimeout = 2 * time.Second

// HasVulkanSupport checks if the Vulkan loader library is available.
// This mirrors what the dynamic linker does when loading the llama.cpp Vulkan binary.
// If libvulkan.so is present, the Vulkan build can run and llama.cpp will
// enumerate GPU devices at runtime via vk::enumeratePhysicalDevices().
func HasVulkanSupport() bool {
	if runtime.GOOS != "linux" {
		return false
	}

	// Check LD_LIBRARY_PATH first (matches dynamic linker behavior)
	if ldPath := os.Getenv("LD_LIBRARY_PATH"); ldPath != "" {
		for _, dir := range strings.Split(ldPath, ":") {
			if _, err := os.Stat(filepath.Join(dir, "libvulkan.so.1")); err == nil {
				return true
			}
		}
	}

	// Check standard library paths
	libPaths := []string{
		"/usr/lib/x86_64-linux-gnu/libvulkan.so.1", // Debian/Ubuntu
		"/usr/lib64/libvulkan.so.1",                // RHEL/Fedora
		"/usr/lib/libvulkan.so.1",                  // Arch/other
	}
	for _, path := range libPaths {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}

	// Fallback: use ldconfig to check if libvulkan is in the cache
	ctx, cancel := context.WithTimeout(context.Background(), vulkanCheckTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ldconfig", "-p")
	if output, err := cmd.Output(); err == nil {
		if strings.Contains(string(output), "libvulkan.so") {
			return true
		}
	}

	return false
}

const (
	llamaRepo = "ggml-org/llama.cpp"
	apiBase   = "https://api.github.com/repos/" + llamaRepo
)

type Release struct {
	TagName string  `json:"tag_name"`
	Name    string  `json:"name"`
	Assets  []Asset `json:"assets"`
}

type Asset struct {
	Name               string `json:"name"`
	Size               int64  `json:"size"`
	BrowserDownloadUrl string `json:"browser_download_url"`
}

type VersionInfo struct {
	TagName     string `json:"tag_name"`
	BinaryPath  string `json:"binary_path"`
	InstalledAt string `json:"installed_at"`
}

type VersionFile struct {
	Llama *VersionInfo `json:"llama,omitempty"`
}

func getPlatform() string {
	osName := runtime.GOOS
	arch := runtime.GOARCH

	switch osName {
	case "darwin":
		if arch == "arm64" {
			return "macos-arm64"
		}
		return "macos-x64"
	case "linux":
		if arch == "amd64" {
			// Use Vulkan build if libvulkan.so is available
			// llama.cpp handles GPU enumeration at runtime via vk::enumeratePhysicalDevices()
			if HasVulkanSupport() {
				return "ubuntu-vulkan-x64"
			}
			return "ubuntu-x64"
		}
		// llama.cpp does not ship Linux ARM64 binaries
		return ""
	default:
		return ""
	}
}

func getBinaryPattern(release *Release) string {
	platform := getPlatform()
	if platform == "" {
		return ""
	}

	return "llama-" + release.TagName + "-bin-" + platform + ".tar.gz"
}

func GetLatestVersion() (*Release, error) {
	url := apiBase + "/releases/latest"

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", "application/vnd.github.v3+json")
	req.Header.Set("User-Agent", version.UserAgent())

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var release Release
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return nil, err
	}

	return &release, nil
}

func FindAssetForPlatform(release *Release) (string, string, error) {
	binaryPattern := getBinaryPattern(release)
	if binaryPattern == "" {
		return "", "", fmt.Errorf("unsupported platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	for _, asset := range release.Assets {
		if asset.Name == binaryPattern {
			return asset.BrowserDownloadUrl, asset.Name, nil
		}
	}

	return "", "", fmt.Errorf("could not find binary for platform %s", binaryPattern)
}

func DownloadBinary(downloadURL, destPath string, progress func(int64, int64)) error {
	req, err := http.NewRequest("GET", downloadURL, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	req.Header.Set("User-Agent", version.UserAgent())

	// Use transport timeouts for connection setup, but no overall timeout for large downloads
	transport := &http.Transport{
		ResponseHeaderTimeout: 30 * time.Second,
	}
	client := &http.Client{Transport: transport}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", downloadURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", downloadURL, resp.StatusCode)
	}

	tmpPath := destPath + ".partial"
	out, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmpPath, err)
	}
	defer out.Close()

	if _, err := fileutil.StreamBody(resp.Body, out, 0, resp.ContentLength, progress); err != nil {
		return fmt.Errorf("write binary to %s: %w", tmpPath, err)
	}
	out.Close()

	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("rename %s to %s: %w", tmpPath, destPath, err)
	}
	return nil
}

func extractTarGz(archivePath, destDir, tagName string) error {
	cmd := exec.Command("tar", "-xzf", archivePath, "-C", destDir)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("extract %s: %w", archivePath, err)
	}

	llamaDirName := "llama-" + tagName
	info, err := os.Stat(filepath.Join(destDir, llamaDirName))
	if err != nil || !info.IsDir() {
		return fmt.Errorf("expected directory %s not found in archive", llamaDirName)
	}

	currentLink := filepath.Join(destDir, "llama-current")
	if err := os.RemoveAll(currentLink); err != nil {
		return fmt.Errorf("failed to remove existing llama-current: %w", err)
	}
	if err := os.Symlink(llamaDirName, currentLink); err != nil {
		return fmt.Errorf("failed to create llama-current symlink: %w", err)
	}

	return nil
}

func removeOldVersions(binDir, currentTag string) {
	currentDir := "llama-" + currentTag
	entries, err := os.ReadDir(binDir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() && strings.HasPrefix(name, "llama-b") && name != currentDir {
			os.RemoveAll(filepath.Join(binDir, name))
		}
	}
}

// StatusFunc is a callback for reporting installation progress messages.
type StatusFunc func(message string)

func InstallLatest(status StatusFunc) (*VersionInfo, error) {
	release, err := GetLatestVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to get latest release: %w", err)
	}

	downloadURL, binaryName, err := FindAssetForPlatform(release)
	if err != nil {
		return nil, err
	}

	binDir := config.BinPath()
	if err := os.MkdirAll(binDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create bin directory: %w", err)
	}

	archivePath := filepath.Join(binDir, binaryName)

	if status != nil {
		msg := fmt.Sprintf("Downloading llama.cpp %s", release.TagName)
		if HasVulkanSupport() {
			msg += " (Vulkan)"
		}
		status(msg)
	}

	if err := DownloadBinary(downloadURL, archivePath, nil); err != nil {
		return nil, fmt.Errorf("failed to download binary: %w", err)
	}

	if status != nil {
		status("Extracting...")
	}

	if err := extractTarGz(archivePath, binDir, release.TagName); err != nil {
		return nil, fmt.Errorf("failed to extract archive: %w", err)
	}

	// Clean up tarball and old versions after successful extraction
	os.Remove(archivePath)
	removeOldVersions(binDir, release.TagName)

	cliPath := filepath.Join(binDir, "llama-current", "llama-cli")
	versionInfo := &VersionInfo{
		TagName:     release.TagName,
		BinaryPath:  cliPath,
		InstalledAt: time.Now().Format(time.RFC3339),
	}

	if err := SaveVersionInfo(versionInfo); err != nil {
		return nil, fmt.Errorf("failed to save version info: %w", err)
	}

	return versionInfo, nil
}

func GetInstalledVersion() (*VersionInfo, error) {
	versionPath := filepath.Join(config.BinPath(), "version.json")

	data, err := os.ReadFile(versionPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var file VersionFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}

	return file.Llama, nil
}

func SaveVersionInfo(version *VersionInfo) error {
	versionPath := filepath.Join(config.BinPath(), "version.json")

	if err := os.MkdirAll(filepath.Dir(versionPath), 0755); err != nil {
		return fmt.Errorf("failed to create version directory: %w", err)
	}

	file := VersionFile{Llama: version}
	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal version: %w", err)
	}

	return os.WriteFile(versionPath, data, 0644)
}

func ServerPath() string {
	return filepath.Join(config.BinPath(), "llama-current", "llama-server")
}

func IsInstalled() bool {
	cliPath := filepath.Join(config.BinPath(), "llama-current", "llama-cli")
	if _, err := os.Stat(cliPath); err != nil {
		return false
	}
	return true
}
