package llama

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
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
		"/usr/lib/x86_64-linux-gnu/libvulkan.so.1",  // Debian/Ubuntu x86_64
		"/usr/lib/aarch64-linux-gnu/libvulkan.so.1", // Debian/Ubuntu ARM64
		"/usr/lib64/libvulkan.so.1",                 // RHEL/Fedora
		"/usr/lib/libvulkan.so.1",                   // Arch/other
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

const llamaRepo = "ggml-org/llama.cpp"

// apiBase is the GitHub API root for llama.cpp. Declared as var (not const) so
// tests can redirect it to an httptest server.
var apiBase = "https://api.github.com/repos/" + llamaRepo

// maxDownloadBytes bounds the size we'll accept for a llama.cpp tarball to
// guard against a malicious or corrupted response filling the disk. Current
// releases are ~300 MB; 1 GiB is a generous ceiling. Declared as var (not
// const) so tests can shrink it to exercise the overflow guard.
var maxDownloadBytes int64 = 1 << 30

// tagNameRe matches llama.cpp release tag names (e.g. "b8169"). Enforced on
// GitHub API responses so attacker-influenced values can't traverse out of
// binDir when embedded in archive names, extracted directory names, or the
// llama-current symlink target.
var tagNameRe = regexp.MustCompile(`^b\d+$`)

// allowedDownloadHosts are the hosts from which llama.cpp release assets may
// be fetched. github.com issues 302 redirects to *.githubusercontent.com.
var allowedDownloadHosts = map[string]bool{
	"github.com":                           true,
	"objects.githubusercontent.com":        true,
	"release-assets.githubusercontent.com": true,
	"releases.githubusercontent.com":       true,
}

// allowedDownloadSchemes restricts the URL schemes accepted by
// validateDownloadURL. In production this is https only; tests override it to
// include http when pointing at an httptest server.
var allowedDownloadSchemes = map[string]bool{"https": true}

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
		// Use Vulkan build if libvulkan.so is available; llama.cpp enumerates
		// GPU devices at runtime via vk::enumeratePhysicalDevices().
		vulkan := HasVulkanSupport()
		switch arch {
		case "amd64":
			if vulkan {
				return "ubuntu-vulkan-x64"
			}
			return "ubuntu-x64"
		case "arm64":
			if vulkan {
				return "ubuntu-vulkan-arm64"
			}
			return "ubuntu-arm64"
		}
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

	if !tagNameRe.MatchString(release.TagName) {
		return nil, fmt.Errorf("unexpected tag_name %q in release response (expected b<number>)", release.TagName)
	}

	return &release, nil
}

// validateDownloadURL rejects download URLs that don't use HTTPS and point to
// a known GitHub release-asset host. Prevents a compromised API response from
// redirecting the download to attacker-controlled infrastructure.
func validateDownloadURL(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid download URL %q: %w", rawURL, err)
	}
	if !allowedDownloadSchemes[u.Scheme] {
		return fmt.Errorf("download URL scheme %q is not allowed", u.Scheme)
	}
	if !allowedDownloadHosts[u.Hostname()] {
		return fmt.Errorf("download URL host %q is not on the allowlist", u.Hostname())
	}
	return nil
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
	return DownloadBinaryContext(context.Background(), downloadURL, destPath, progress)
}

// DownloadBinaryContext is DownloadBinary with cancellation support. The
// context is honored for connection setup and body streaming, so a canceled
// context aborts an in-flight download promptly.
func DownloadBinaryContext(ctx context.Context, downloadURL, destPath string, progress func(int64, int64)) error {
	if err := validateDownloadURL(downloadURL); err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "GET", downloadURL, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	req.Header.Set("User-Agent", version.UserAgent())

	// Use transport timeouts for connection setup, but no overall timeout for large downloads
	transport := &http.Transport{
		ResponseHeaderTimeout: 30 * time.Second,
	}
	client := &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if err := validateDownloadURL(req.URL.String()); err != nil {
				return fmt.Errorf("redirect blocked: %w", err)
			}
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", downloadURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", downloadURL, resp.StatusCode)
	}

	if resp.ContentLength > maxDownloadBytes {
		return fmt.Errorf("download %s: content-length %d exceeds max %d", downloadURL, resp.ContentLength, maxDownloadBytes)
	}

	tmpPath := destPath + ".partial"
	out, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmpPath, err)
	}
	defer out.Close()

	// Cap the body stream at maxDownloadBytes+1 so we can detect a truncated
	// or header-lying response and reject it instead of filling the disk.
	limited := io.LimitReader(resp.Body, maxDownloadBytes+1)
	written, err := fileutil.StreamBody(limited, out, 0, resp.ContentLength, progress)
	if err != nil {
		return fmt.Errorf("write binary to %s: %w", tmpPath, err)
	}
	if written > maxDownloadBytes {
		return fmt.Errorf("download %s exceeded max size of %d bytes", downloadURL, maxDownloadBytes)
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

	return swapCurrentSymlink(destDir, llamaDirName)
}

// swapCurrentSymlink atomically points llama-current at target by staging a
// temp symlink and renaming it over the destination. Avoids the ENOENT window
// that a RemoveAll+Symlink pair would expose to concurrent exec.Command calls.
func swapCurrentSymlink(destDir, target string) error {
	currentLink := filepath.Join(destDir, "llama-current")
	tmpLink := filepath.Join(destDir, ".llama-current.tmp")

	// A stale tmp from a prior failed run would make Symlink fail with EEXIST.
	_ = os.Remove(tmpLink)

	if err := os.Symlink(target, tmpLink); err != nil {
		return fmt.Errorf("failed to stage llama-current symlink: %w", err)
	}
	if err := os.Rename(tmpLink, currentLink); err != nil {
		_ = os.Remove(tmpLink)
		return fmt.Errorf("failed to activate llama-current symlink: %w", err)
	}
	return nil
}

func removeOldVersions(binDir, currentTag string) {
	pruneOldVersions(binDir, currentTag, 0)
}

// pruneOldVersions deletes llama-b* version directories except (a) the one
// named by currentTag, (b) whatever the llama-current symlink actually resolves
// to, and (c) the keepPrior most-recent of the rest, ordered by mtime.
// keepPrior=0 means only the current version survives. Auto-update uses
// keepPrior=2: a backend forked just before the symlink swap may still be
// lazy-dlopen'ing GPU backend plugins out of its version directory (Linux
// Vulkan/CUDA in particular), so we keep two recent predecessors as insurance.
func pruneOldVersions(binDir, currentTag string, keepPrior int) {
	spare := map[string]bool{
		"llama-" + currentTag: true,
	}
	// Resolve the symlink's actual target and always spare it, defending against
	// a mismatch between currentTag and what llama-current actually points to
	// (e.g. manual rollback or stale state from a prior partial install).
	if target, err := os.Readlink(filepath.Join(binDir, "llama-current")); err == nil {
		spare[filepath.Base(target)] = true
	}

	entries, err := os.ReadDir(binDir)
	if err != nil {
		return
	}

	type candidate struct {
		name  string
		mtime time.Time
	}
	var prior []candidate
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() || !strings.HasPrefix(name, "llama-b") || spare[name] {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		prior = append(prior, candidate{name: name, mtime: info.ModTime()})
	}

	// Sort newest-first so the first `keepPrior` entries are retained.
	sort.Slice(prior, func(i, j int) bool {
		return prior[i].mtime.After(prior[j].mtime)
	})

	for i, c := range prior {
		if i < keepPrior {
			continue
		}
		os.RemoveAll(filepath.Join(binDir, c.name))
	}
}

// StatusFunc is a callback for reporting installation progress messages.
type StatusFunc func(message string)

// InstallLatest downloads and installs the latest llama.cpp release, replacing
// the llama-current symlink and deleting all previous version directories.
func InstallLatest(status StatusFunc) (*VersionInfo, error) {
	release, err := GetLatestVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to get latest release: %w", err)
	}
	return installRelease(context.Background(), release, status, 0)
}

// InstallReleaseForAutoUpdate installs a pre-fetched release and retains the
// two most recent prior version directories. Used by the background auto-update
// path: a backend that started just before the symlink swap may still be
// executing the old binary and lazily loading shared libs from its version
// directory, so we keep recent predecessors on disk. Older versions are pruned
// and cleanup eventually catches up on subsequent installs. The context bounds
// the download so proxy shutdown can abort a long-running fetch.
func InstallReleaseForAutoUpdate(ctx context.Context, release *Release, status StatusFunc) (*VersionInfo, error) {
	return installRelease(ctx, release, status, 2)
}

func installRelease(ctx context.Context, release *Release, status StatusFunc, keepPrior int) (*VersionInfo, error) {
	downloadURL, binaryName, err := FindAssetForPlatform(release)
	if err != nil {
		return nil, err
	}

	binDir := config.BinPath()
	if err := os.MkdirAll(binDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create bin directory: %w", err)
	}

	archivePath := filepath.Join(binDir, binaryName)
	// Clean up the archive whether extraction succeeds or fails.
	defer os.Remove(archivePath)

	if status != nil {
		msg := fmt.Sprintf("Downloading llama.cpp %s", release.TagName)
		if HasVulkanSupport() {
			msg += " (Vulkan)"
		}
		status(msg)
	}

	if err := DownloadBinaryContext(ctx, downloadURL, archivePath, nil); err != nil {
		return nil, fmt.Errorf("failed to download binary: %w", err)
	}

	if status != nil {
		status("Extracting...")
	}

	if err := extractTarGz(archivePath, binDir, release.TagName); err != nil {
		return nil, fmt.Errorf("failed to extract archive: %w", err)
	}

	pruneOldVersions(binDir, release.TagName, keepPrior)

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

// NewerVersionAvailable returns the latest release when it differs from what
// is installed, and nil when we're already up-to-date. Returns nil, nil, nil
// if llama.cpp is not yet installed — the background auto-update path does not
// bootstrap fresh installs (the synchronous path handles that with UI).
func NewerVersionAvailable() (*Release, *VersionInfo, error) {
	installed, err := GetInstalledVersion()
	if err != nil {
		return nil, nil, fmt.Errorf("read installed version: %w", err)
	}
	if installed == nil {
		return nil, nil, nil
	}
	latest, err := GetLatestVersion()
	if err != nil {
		return nil, installed, fmt.Errorf("fetch latest release: %w", err)
	}
	if latest.TagName == installed.TagName {
		return nil, installed, nil
	}
	return latest, installed, nil
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
