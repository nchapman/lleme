package swiftlm

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nchapman/lleme/internal/binaryrelease"
)

func TestExpectedAssetName(t *testing.T) {
	if got := expectedAssetName("b517"); got != "SwiftLM-b517-macos-arm64.tar.gz" {
		t.Errorf("expectedAssetName = %q", got)
	}
}

func TestFindAssetForPlatformMatches(t *testing.T) {
	if !IsSupported() {
		t.Skip("platform not supported")
	}
	rel := &binaryrelease.Release{
		TagName: "b1",
		Assets: []binaryrelease.Asset{
			{Name: "SwiftLM-b1-macos-arm64.tar.gz", BrowserDownloadURL: "https://github.com/SharpAI/SwiftLM/releases/download/b1/x"},
			{Name: "other-asset.zip", BrowserDownloadURL: "https://example.invalid/y"},
		},
	}
	url, name, err := FindAssetForPlatform(rel)
	if err != nil {
		t.Fatal(err)
	}
	if name != "SwiftLM-b1-macos-arm64.tar.gz" {
		t.Errorf("name = %q", name)
	}
	if !strings.HasPrefix(url, "https://github.com/") {
		t.Errorf("url = %q", url)
	}
}

func TestFindAssetForPlatformMissing(t *testing.T) {
	if !IsSupported() {
		t.Skip("platform not supported")
	}
	rel := &binaryrelease.Release{
		TagName: "b2",
		Assets:  []binaryrelease.Asset{{Name: "wrong.tar.gz"}},
	}
	_, _, err := FindAssetForPlatform(rel)
	if err == nil || !strings.Contains(err.Error(), "could not find") {
		t.Errorf("err = %v", err)
	}
}

func TestFindAssetForPlatformUnsupported(t *testing.T) {
	if IsSupported() {
		t.Skip("this test exercises the unsupported-platform branch")
	}
	_, _, err := FindAssetForPlatform(&binaryrelease.Release{TagName: "b1"})
	if _, ok := err.(UnsupportedPlatformError); !ok {
		t.Errorf("err = %v, want UnsupportedPlatformError", err)
	}
}

func TestTagNameValidation(t *testing.T) {
	good := []string{"b0", "b517", "b99999"}
	bad := []string{"", "v1.2.3", "b", "b517-rc1", "../escape", "b517/..", "latest"}
	for _, s := range good {
		if !tagNameRe.MatchString(s) {
			t.Errorf("tagNameRe should match %q", s)
		}
	}
	for _, s := range bad {
		if tagNameRe.MatchString(s) {
			t.Errorf("tagNameRe should NOT match %q", s)
		}
	}
}

func TestVersionInfoRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	if got, err := GetInstalledVersion(); err != nil || got != nil {
		t.Fatalf("fresh install: got %+v err %v, want nil, nil", got, err)
	}

	want := &VersionInfo{TagName: "b1", BinaryPath: "/fake/SwiftLM", InstalledAt: "2026-04-19T00:00:00Z"}
	if err := SaveVersionInfo(want); err != nil {
		t.Fatalf("SaveVersionInfo: %v", err)
	}
	got, err := GetInstalledVersion()
	if err != nil {
		t.Fatalf("GetInstalledVersion: %v", err)
	}
	if got == nil || *got != *want {
		t.Errorf("round-trip mismatch: got %+v, want %+v", got, want)
	}

	// File actually written where we expect.
	if _, err := os.Stat(filepath.Join(tmpDir, ".lleme", "bin", "swiftlm-version.json")); err != nil {
		t.Errorf("version file not at expected path: %v", err)
	}
}

// SaveVersionInfo must not leave a stray .tmp sibling behind on success.
// Crash-window behavior (tmp survives a mid-write kill) is tested
// implicitly — if a .tmp ever ends up where the real file should be, the
// next GetInstalledVersion would reject it as unparseable.
func TestSaveVersionInfoIsAtomic(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)
	os.Setenv("HOME", tmpDir)

	info := &VersionInfo{TagName: "b999", BinaryPath: "/fake", InstalledAt: "now"}
	if err := SaveVersionInfo(info); err != nil {
		t.Fatal(err)
	}
	binDir := filepath.Join(tmpDir, ".lleme", "bin")
	if _, err := os.Stat(filepath.Join(binDir, "swiftlm-version.json.tmp")); !os.IsNotExist(err) {
		t.Errorf("leftover .tmp sibling after SaveVersionInfo: err=%v", err)
	}
	if _, err := os.Stat(filepath.Join(binDir, "swiftlm-version.json")); err != nil {
		t.Errorf("final version file missing: %v", err)
	}
}

func TestPruneOldVersionsKeepsCurrentAndSymlinkTarget(t *testing.T) {
	tmpDir := t.TempDir()
	dirs := []string{"SwiftLM-b1-macos-arm64", "SwiftLM-b2-macos-arm64", "SwiftLM-b3-macos-arm64"}
	for _, d := range dirs {
		if err := os.MkdirAll(filepath.Join(tmpDir, d), 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := binaryrelease.SwapCurrentSymlink(tmpDir, currentLinkName, "SwiftLM-b3-macos-arm64"); err != nil {
		t.Fatal(err)
	}

	// Prune with currentTag=b3, keepPrior=0: b1 and b2 should go.
	pruneOldVersions(tmpDir, "b3", 0)

	if _, err := os.Stat(filepath.Join(tmpDir, "SwiftLM-b3-macos-arm64")); err != nil {
		t.Errorf("current version should survive: %v", err)
	}
	for _, d := range []string{"SwiftLM-b1-macos-arm64", "SwiftLM-b2-macos-arm64"} {
		if _, err := os.Stat(filepath.Join(tmpDir, d)); !os.IsNotExist(err) {
			t.Errorf("%s should have been pruned", d)
		}
	}
}

// keepPrior=2 retains the two most-recent prior versions (newest first by
// mtime) in addition to the current one. Mirrors the llama-side guarantee
// used by auto-update to insure backends forked just before a symlink swap
// keep the old binary on disk.
func TestPruneOldVersionsKeepsPrior(t *testing.T) {
	tmpDir := t.TempDir()
	// Create four versions and stagger mtimes so the sort order is
	// deterministic: b4 newest, b1 oldest.
	dirs := []string{"SwiftLM-b1-macos-arm64", "SwiftLM-b2-macos-arm64", "SwiftLM-b3-macos-arm64", "SwiftLM-b4-macos-arm64"}
	baseTime := time.Now()
	for i, d := range dirs {
		p := filepath.Join(tmpDir, d)
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatal(err)
		}
		mtime := baseTime.Add(time.Duration(i) * time.Second)
		if err := os.Chtimes(p, mtime, mtime); err != nil {
			t.Fatal(err)
		}
	}
	if err := binaryrelease.SwapCurrentSymlink(tmpDir, currentLinkName, "SwiftLM-b4-macos-arm64"); err != nil {
		t.Fatal(err)
	}

	// currentTag=b4, keepPrior=2: b4 + b3 + b2 survive, b1 is pruned.
	pruneOldVersions(tmpDir, "b4", 2)

	for _, survivor := range []string{"SwiftLM-b2-macos-arm64", "SwiftLM-b3-macos-arm64", "SwiftLM-b4-macos-arm64"} {
		if _, err := os.Stat(filepath.Join(tmpDir, survivor)); err != nil {
			t.Errorf("%s should survive keepPrior=2, got err %v", survivor, err)
		}
	}
	if _, err := os.Stat(filepath.Join(tmpDir, "SwiftLM-b1-macos-arm64")); !os.IsNotExist(err) {
		t.Errorf("oldest version should have been pruned, stat err=%v", err)
	}
}

// Symlink target must survive pruning regardless of keepPrior or whether
// the caller's currentTag matches it. Defends against partial install state
// leaving swiftlm-current pointing at a directory that pruneOldVersions
// would otherwise delete.
func TestPruneRespectsSymlinkTarget(t *testing.T) {
	tmpDir := t.TempDir()
	now := time.Now()
	// b1 is oldest by mtime — would normally be pruned with keepPrior=1 and
	// currentTag=b4. We'll point swiftlm-current at it to simulate stale
	// state; the symlink target must survive anyway.
	versions := []struct {
		name  string
		mtime time.Time
	}{
		{"SwiftLM-b4-macos-arm64", now},
		{"SwiftLM-b3-macos-arm64", now.Add(-1 * time.Hour)},
		{"SwiftLM-b2-macos-arm64", now.Add(-2 * time.Hour)},
		{"SwiftLM-b1-macos-arm64", now.Add(-10 * time.Hour)},
	}
	for _, v := range versions {
		dir := filepath.Join(tmpDir, v.name)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(dir, v.mtime, v.mtime); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink("SwiftLM-b1-macos-arm64", filepath.Join(tmpDir, currentLinkName)); err != nil {
		t.Fatal(err)
	}

	// currentTag=b4 with keepPrior=1: normally would save b4 + b3, drop b2/b1.
	// b1 must survive because it's the symlink target.
	pruneOldVersions(tmpDir, "b4", 1)

	if _, err := os.Stat(filepath.Join(tmpDir, "SwiftLM-b1-macos-arm64")); err != nil {
		t.Errorf("symlink target must be retained even on mismatched currentTag: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tmpDir, "SwiftLM-b4-macos-arm64")); err != nil {
		t.Errorf("currentTag directory should be retained: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tmpDir, "SwiftLM-b3-macos-arm64")); err != nil {
		t.Errorf("most-recent prior should be retained with keepPrior=1: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tmpDir, "SwiftLM-b2-macos-arm64")); !os.IsNotExist(err) {
		t.Errorf("b2 should be pruned with keepPrior=1: %v", err)
	}
}

// extractTarGz must swap the swiftlm-current symlink to the newly-extracted
// version even when a prior symlink points elsewhere. The upstream archive
// is flat (no wrapper directory); the test mimics that shape.
func TestExtractTarGzSwapsSymlink(t *testing.T) {
	if _, err := exec.LookPath("tar"); err != nil {
		t.Skip("tar not available")
	}
	tmpDir := t.TempDir()
	binDir := filepath.Join(tmpDir, "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Build a flat archive containing just a stub `SwiftLM` binary.
	srcDir := filepath.Join(tmpDir, "src")
	if err := os.MkdirAll(srcDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(srcDir, "SwiftLM"), []byte("stub"), 0755); err != nil {
		t.Fatal(err)
	}
	archive := filepath.Join(tmpDir, "swiftlm.tar.gz")
	if err := exec.Command("tar", "-czf", archive, "-C", srcDir, "SwiftLM").Run(); err != nil {
		t.Fatalf("tar: %v", err)
	}

	// Pre-create an older version dir + symlink to simulate upgrade.
	oldDir := filepath.Join(binDir, "SwiftLM-b500-macos-arm64")
	if err := os.MkdirAll(oldDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("SwiftLM-b500-macos-arm64", filepath.Join(binDir, currentLinkName)); err != nil {
		t.Fatal(err)
	}

	if err := extractTarGz(archive, binDir, "b600"); err != nil {
		t.Fatalf("extractTarGz: %v", err)
	}

	target, err := os.Readlink(filepath.Join(binDir, currentLinkName))
	if err != nil {
		t.Fatalf("readlink: %v", err)
	}
	if target != "SwiftLM-b600-macos-arm64" {
		t.Errorf("symlink target = %q, want SwiftLM-b600-macos-arm64", target)
	}
	// Sanity: the new version dir exists and contains the stub binary.
	if _, err := os.Stat(filepath.Join(binDir, "SwiftLM-b600-macos-arm64", "SwiftLM")); err != nil {
		t.Errorf("extracted binary missing: %v", err)
	}
}

// extractTarGz must reject an archive that doesn't contain the SwiftLM
// binary where expected — a mangled upstream asset shouldn't silently swap
// the current symlink to an unusable directory.
func TestExtractTarGzRejectsMissingBinary(t *testing.T) {
	if _, err := exec.LookPath("tar"); err != nil {
		t.Skip("tar not available")
	}
	tmpDir := t.TempDir()
	binDir := filepath.Join(tmpDir, "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}
	// Archive contains the wrong binary name.
	srcDir := filepath.Join(tmpDir, "src")
	if err := os.MkdirAll(srcDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(srcDir, "WrongName"), []byte("stub"), 0755); err != nil {
		t.Fatal(err)
	}
	archive := filepath.Join(tmpDir, "swiftlm.tar.gz")
	if err := exec.Command("tar", "-czf", archive, "-C", srcDir, "WrongName").Run(); err != nil {
		t.Fatalf("tar: %v", err)
	}

	if err := extractTarGz(archive, binDir, "b600"); err == nil {
		t.Error("extractTarGz should reject archive without SwiftLM binary")
	}
}

// NewerVersionAvailable returns (nil, nil, nil) when SwiftLM has never been
// installed — the auto-update path bootstraps nothing, only refreshes.
func TestNewerVersionAvailableNotInstalled(t *testing.T) {
	if !IsSupported() {
		t.Skip("NewerVersionAvailable bootstraps from an install; unsupported platforms short-circuit to nil anyway")
	}
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	latest, installed, err := NewerVersionAvailable()
	if err != nil {
		t.Fatalf("not-installed case should not error: %v", err)
	}
	if latest != nil || installed != nil {
		t.Errorf("expected (nil, nil), got latest=%+v installed=%+v", latest, installed)
	}
}
