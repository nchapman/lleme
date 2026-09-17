package llama

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/binaryrelease"
	"github.com/nchapman/lleme/internal/config"
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

// semverTagRe matches llama.cpp stable release tags (e.g. "v0.4.1"). Stable
// releases carry no binary assets; their nightly-tag.txt asset names the
// b<number> prerelease that does.
var semverTagRe = regexp.MustCompile(`^v\d+\.\d+\.\d+$`)

// maxNightlyTagBytes caps how much of nightly-tag.txt we'll read. The file
// holds a single b<number> tag (~8 bytes); anything larger is a corrupted or
// hostile response and is rejected outright.
const maxNightlyTagBytes int64 = 1024

// allowedDownloadHosts are the hosts from which llama.cpp release assets and
// GitHub API responses may be fetched. github.com issues 302 redirects to
// *.githubusercontent.com for asset downloads. Declared as var so tests can
// admit an httptest server.
var allowedDownloadHosts = binaryrelease.DefaultGitHubHosts()

// allowedDownloadSchemes restricts the URL schemes accepted for downloads.
// In production this is https only; tests override it to include http when
// pointing at an httptest server.
var allowedDownloadSchemes = binaryrelease.DefaultHTTPSOnly()

// releaseConfig bundles the download policy knobs the binaryrelease
// primitives enforce (host/scheme allow-list, size cap, user agent). Built
// from the package-level vars so tests can retarget it at an httptest server.
func releaseConfig() binaryrelease.Config {
	return binaryrelease.Config{
		AllowedHosts:   allowedDownloadHosts,
		AllowedSchemes: allowedDownloadSchemes,
		MaxBytes:       maxDownloadBytes,
		UserAgent:      version.UserAgent(),
	}
}

// Release is the GitHub release shape for llama.cpp, shared with the
// binaryrelease installer primitives.
type Release = binaryrelease.Release

// VersionInfo records what's installed in bin/version.json.
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
	release, err := fetchRelease("/releases/latest")
	if err != nil {
		return nil, err
	}

	switch {
	case tagNameRe.MatchString(release.TagName):
		return release, nil
	case semverTagRe.MatchString(release.TagName):
		return resolveStableRelease(release)
	default:
		return nil, fmt.Errorf("unexpected tag_name %q in release response (expected b<number> or vX.Y.Z)", release.TagName)
	}
}

// fetchRelease retrieves a release from the GitHub API at the given path
// (e.g. "/releases/latest" or "/releases/tags/b8169").
func fetchRelease(path string) (*Release, error) {
	return binaryrelease.FetchLatestRelease(context.Background(), releaseConfig(), apiBase+path)
}

// resolveStableRelease maps a stable vX.Y.Z release to the b<number> release
// that carries its binaries. llama.cpp stable releases ship only a
// nightly-tag.txt asset whose contents name the matching b<number> tag.
func resolveStableRelease(stable *Release) (*Release, error) {
	tag, err := fetchNightlyTag(stable)
	if err != nil {
		return nil, fmt.Errorf("resolve stable release %s: %w", stable.TagName, err)
	}
	release, err := fetchRelease("/releases/tags/" + tag)
	if err != nil {
		return nil, fmt.Errorf("fetch release %s: %w", tag, err)
	}
	if !tagNameRe.MatchString(release.TagName) {
		return nil, fmt.Errorf("unexpected tag_name %q for release %s (expected b<number>)", release.TagName, tag)
	}
	return release, nil
}

func fetchNightlyTag(stable *Release) (string, error) {
	var assetURL string
	for _, asset := range stable.Assets {
		if asset.Name == "nightly-tag.txt" {
			assetURL = asset.BrowserDownloadURL
			break
		}
	}
	if assetURL == "" {
		return "", fmt.Errorf("release has no nightly-tag.txt asset")
	}

	body, err := binaryrelease.FetchBytes(context.Background(), releaseConfig(), assetURL, maxNightlyTagBytes)
	if err != nil {
		return "", fmt.Errorf("fetch nightly-tag.txt: %w", err)
	}

	tag := strings.TrimSpace(string(body))
	if !tagNameRe.MatchString(tag) {
		return "", fmt.Errorf("nightly-tag.txt contents %q are not a b<number> tag", tag)
	}
	return tag, nil
}

func FindAssetForPlatform(release *Release) (string, string, error) {
	binaryPattern := getBinaryPattern(release)
	if binaryPattern == "" {
		return "", "", fmt.Errorf("unsupported platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	for _, asset := range release.Assets {
		if asset.Name == binaryPattern {
			return asset.BrowserDownloadURL, asset.Name, nil
		}
	}

	return "", "", fmt.Errorf("could not find binary for platform %s", binaryPattern)
}

// DownloadBinaryContext downloads a release asset to destPath, enforcing the
// GitHub host/scheme allow-list and the package size cap. The context is
// honored for connection setup and body streaming, so a canceled context
// aborts an in-flight download promptly.
func DownloadBinaryContext(ctx context.Context, downloadURL, destPath string, progress func(int64, int64)) error {
	return binaryrelease.Download(ctx, releaseConfig(), downloadURL, destPath, progress)
}

// extractTarGz unpacks the release archive with the validating extractor
// (no path traversal, no escaping or absolute symlinks, no setuid bits),
// confirms the expected llama-<tag> directory, and atomically repoints
// llama-current at it.
func extractTarGz(archivePath, destDir, tagName string) error {
	if err := binaryrelease.ExtractTarGz(archivePath, destDir); err != nil {
		return fmt.Errorf("extract %s: %w", archivePath, err)
	}

	llamaDirName := "llama-" + tagName
	info, err := os.Stat(filepath.Join(destDir, llamaDirName))
	if err != nil || !info.IsDir() {
		return fmt.Errorf("expected directory %s not found in archive", llamaDirName)
	}

	return binaryrelease.SwapCurrentSymlink(destDir, "llama-current", llamaDirName)
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
