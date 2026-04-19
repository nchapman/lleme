package swiftlm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

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

	// Prune with currentTag=b3: b1 and b2 should go.
	pruneOldVersions(tmpDir, "b3")

	if _, err := os.Stat(filepath.Join(tmpDir, "SwiftLM-b3-macos-arm64")); err != nil {
		t.Errorf("current version should survive: %v", err)
	}
	for _, d := range []string{"SwiftLM-b1-macos-arm64", "SwiftLM-b2-macos-arm64"} {
		if _, err := os.Stat(filepath.Join(tmpDir, d)); !os.IsNotExist(err) {
			t.Errorf("%s should have been pruned", d)
		}
	}
}
