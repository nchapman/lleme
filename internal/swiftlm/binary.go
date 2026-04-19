// Package swiftlm manages the SwiftLM server binary — a self-contained
// macOS/arm64 executable from SharpAI/SwiftLM that serves MLX models over
// an OpenAI-compatible HTTP API. The lifecycle mirrors internal/llama:
// resolve latest release → download tarball → extract → flip a "current"
// symlink. Shared security-sensitive pieces (URL allow-list, download
// caps, atomic symlink swap) live in internal/binaryrelease.
package swiftlm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
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

// swiftlmRepo is the GitHub repo we fetch releases from.
const swiftlmRepo = "SharpAI/SwiftLM"

// apiBase is overridable so tests can redirect to an httptest server.
var apiBase = "https://api.github.com/repos/" + swiftlmRepo

// maxDownloadBytes bounds the artifact size. SwiftLM is a single binary plus
// its metallib shader library; the current release hovers around 50 MB. A
// 1 GiB ceiling is generous. Declared var for test overrides.
var maxDownloadBytes int64 = 1 << 30

// tagNameRe matches SwiftLM release tag names (e.g. "b517"). Enforced on the
// GitHub response so attacker-influenced values can't traverse out of binDir
// via archive names, extracted directory names, or the swiftlm-current link.
var tagNameRe = regexp.MustCompile(`^b\d+$`)

// currentLinkName is the stable symlink callers invoke. It always resolves
// to whichever swiftlm-<tag> directory is active.
const currentLinkName = "swiftlm-current"

// VersionInfo records an installed SwiftLM version in bin/swiftlm-version.json.
// Kept in a sibling file (not merged into llama's version.json) so the two
// installers stay independent.
type VersionInfo struct {
	TagName     string `json:"tag_name"`
	BinaryPath  string `json:"binary_path"`
	InstalledAt string `json:"installed_at"`
}

// IsSupported reports whether the current platform can run SwiftLM. The
// upstream project ships only macOS/arm64 binaries.
func IsSupported() bool {
	return runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
}

// UnsupportedPlatformError is returned when install is attempted on a
// platform the upstream project does not publish binaries for. Callers
// use this to surface a clear user-facing message.
type UnsupportedPlatformError struct{}

func (UnsupportedPlatformError) Error() string {
	return "SwiftLM (MLX) requires macOS on Apple Silicon"
}

func releaseConfig() binaryrelease.Config {
	return binaryrelease.Config{
		AllowedHosts:   binaryrelease.DefaultGitHubHosts(),
		AllowedSchemes: binaryrelease.DefaultHTTPSOnly(),
		MaxBytes:       maxDownloadBytes,
		UserAgent:      version.UserAgent(),
	}
}

// GetLatestVersion resolves the latest SwiftLM release and validates the
// tag matches b<number>.
func GetLatestVersion() (*binaryrelease.Release, error) {
	release, err := binaryrelease.FetchLatestRelease(context.Background(), releaseConfig(), apiBase+"/releases/latest")
	if err != nil {
		return nil, err
	}
	if !tagNameRe.MatchString(release.TagName) {
		return nil, fmt.Errorf("unexpected tag_name %q (expected b<number>)", release.TagName)
	}
	return release, nil
}

// expectedAssetName returns the tarball name SwiftLM publishes for the
// current platform. Only macOS/arm64 is supported today.
func expectedAssetName(tagName string) string {
	return fmt.Sprintf("SwiftLM-%s-macos-arm64.tar.gz", tagName)
}

// FindAssetForPlatform picks the release asset matching the current platform.
func FindAssetForPlatform(release *binaryrelease.Release) (string, string, error) {
	if !IsSupported() {
		return "", "", UnsupportedPlatformError{}
	}
	want := expectedAssetName(release.TagName)
	for _, a := range release.Assets {
		if a.Name == want {
			return a.BrowserDownloadURL, a.Name, nil
		}
	}
	return "", "", fmt.Errorf("could not find SwiftLM binary %s in release assets", want)
}

// StatusFunc is a progress-message callback used by Install.
type StatusFunc func(message string)

// InstallLatest downloads and installs the latest SwiftLM release, swapping
// the swiftlm-current symlink and pruning prior versions.
func InstallLatest(status StatusFunc) (*VersionInfo, error) {
	if !IsSupported() {
		return nil, UnsupportedPlatformError{}
	}
	release, err := GetLatestVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to get latest release: %w", err)
	}
	return installRelease(context.Background(), release, status)
}

func installRelease(ctx context.Context, release *binaryrelease.Release, status StatusFunc) (*VersionInfo, error) {
	downloadURL, assetName, err := FindAssetForPlatform(release)
	if err != nil {
		return nil, err
	}

	binDir := config.BinPath()
	if err := os.MkdirAll(binDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create bin directory: %w", err)
	}

	// assetName came from FindAssetForPlatform → expectedAssetName, which
	// hard-codes the prefix and suffix and substitutes a tag already matched
	// against tagNameRe, so it cannot traverse outside binDir.
	archivePath := filepath.Join(binDir, assetName)
	defer os.Remove(archivePath)

	if status != nil {
		status(fmt.Sprintf("Downloading SwiftLM %s", release.TagName))
	}
	if err := binaryrelease.Download(ctx, releaseConfig(), downloadURL, archivePath, nil); err != nil {
		return nil, fmt.Errorf("failed to download binary: %w", err)
	}

	if status != nil {
		status("Extracting...")
	}
	if err := extractTarGz(archivePath, binDir, release.TagName); err != nil {
		return nil, fmt.Errorf("failed to extract archive: %w", err)
	}

	pruneOldVersions(binDir, release.TagName)

	binPath := filepath.Join(binDir, currentLinkName, "SwiftLM")
	info := &VersionInfo{
		TagName:     release.TagName,
		BinaryPath:  binPath,
		InstalledAt: time.Now().Format(time.RFC3339),
	}
	if err := SaveVersionInfo(info); err != nil {
		return nil, fmt.Errorf("failed to save version info: %w", err)
	}
	return info, nil
}

// versionedDirName is the directory layout inside the SwiftLM tarball,
// named to match the archive convention.
func versionedDirName(tagName string) string {
	return "SwiftLM-" + tagName + "-macos-arm64"
}

// extractTarGz unpacks the SwiftLM tarball into a fresh versioned directory
// under destDir. Upstream ships the archive flat (files at the top level, no
// wrapper directory), so we create binDir/SwiftLM-<tag>-macos-arm64/ and
// extract into that — giving us a stable "versioned" layout the symlink and
// pruner can target.
func extractTarGz(archivePath, destDir, tagName string) error {
	versionDir := versionedDirName(tagName)
	versionPath := filepath.Join(destDir, versionDir)

	// Extract into a staging directory, then rename into place. os.Rename
	// is atomic when source and destination live on the same volume (both
	// under destDir here), which avoids a concurrent installer racing
	// against a half-extracted versionPath.
	stagePath := versionPath + ".staging"
	if err := os.RemoveAll(stagePath); err != nil {
		return fmt.Errorf("clear stage %s: %w", stagePath, err)
	}
	if err := os.MkdirAll(stagePath, 0755); err != nil {
		return fmt.Errorf("mkdir stage %s: %w", stagePath, err)
	}
	defer os.RemoveAll(stagePath) // no-op after successful rename
	if err := binaryrelease.ExtractTarGz(archivePath, stagePath); err != nil {
		return fmt.Errorf("extract %s: %w", archivePath, err)
	}
	// Sanity-check the binary landed where we expect.
	binPath := filepath.Join(stagePath, "SwiftLM")
	if info, err := os.Stat(binPath); err != nil || info.IsDir() {
		return fmt.Errorf("SwiftLM binary not found in archive at %s", binPath)
	}
	// Drop any prior install at the target so Rename succeeds on linux
	// where rename over a non-empty dir fails with ENOTEMPTY.
	if err := os.RemoveAll(versionPath); err != nil {
		return fmt.Errorf("clear %s: %w", versionPath, err)
	}
	if err := os.Rename(stagePath, versionPath); err != nil {
		return fmt.Errorf("activate %s: %w", versionPath, err)
	}
	return binaryrelease.SwapCurrentSymlink(destDir, currentLinkName, versionDir)
}

// pruneOldVersions removes swiftlm-b* directories other than the current one
// and whatever the symlink actually resolves to. SwiftLM doesn't dlopen GPU
// backend plugins the way llama.cpp does, so we don't need to keep prior
// versions alive for in-flight processes.
func pruneOldVersions(binDir, currentTag string) {
	spare := map[string]bool{versionedDirName(currentTag): true}
	if target, err := os.Readlink(filepath.Join(binDir, currentLinkName)); err == nil {
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
	const prefix = "SwiftLM-b"
	const suffix = "-macos-arm64"
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() || spare[name] {
			continue
		}
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, suffix) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		prior = append(prior, candidate{name: name, mtime: info.ModTime()})
	}

	sort.Slice(prior, func(i, j int) bool { return prior[i].mtime.After(prior[j].mtime) })
	for _, c := range prior {
		os.RemoveAll(filepath.Join(binDir, c.name))
	}
}

// versionFilePath is the sibling file that records what's installed.
func versionFilePath() string {
	return filepath.Join(config.BinPath(), "swiftlm-version.json")
}

// SaveVersionInfo writes the installed-version metadata.
func SaveVersionInfo(info *VersionInfo) error {
	if err := os.MkdirAll(config.BinPath(), 0755); err != nil {
		return fmt.Errorf("create bin dir: %w", err)
	}
	data, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal version: %w", err)
	}
	// Write + rename so a concurrent update or a crash mid-write can't leave
	// a truncated file that GetInstalledVersion would fail to unmarshal.
	path := versionFilePath()
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("activate %s: %w", path, err)
	}
	return nil
}

// GetInstalledVersion reads the installed-version metadata, returning
// (nil, nil) if SwiftLM has never been installed.
func GetInstalledVersion() (*VersionInfo, error) {
	data, err := os.ReadFile(versionFilePath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var info VersionInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return nil, err
	}
	return &info, nil
}

// ServerPath returns the absolute path to the SwiftLM binary under the
// current symlink. Caller is responsible for checking it actually exists.
func ServerPath() string {
	return filepath.Join(config.BinPath(), currentLinkName, "SwiftLM")
}

// IsInstalled reports whether a SwiftLM binary is present under the current
// symlink. Returns false on unsupported platforms without hitting the disk.
func IsInstalled() bool {
	if !IsSupported() {
		return false
	}
	_, err := os.Stat(ServerPath())
	return err == nil
}
