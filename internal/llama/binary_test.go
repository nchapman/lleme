package llama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestGetPlatform(t *testing.T) {
	result := getPlatform()

	if result == "" {
		t.Skipf("Skipping test: unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" && result != "macos-arm64" {
			t.Errorf("Expected platform macos-arm64, got %s", result)
		}
		if runtime.GOARCH == "amd64" && result != "macos-x64" {
			t.Errorf("Expected platform macos-x64, got %s", result)
		}
	case "linux":
		if runtime.GOARCH == "amd64" {
			if result != "ubuntu-x64" && result != "ubuntu-vulkan-x64" {
				t.Errorf("Expected platform ubuntu-x64 or ubuntu-vulkan-x64, got %s", result)
			}
		}
		if runtime.GOARCH == "arm64" {
			if result != "ubuntu-arm64" && result != "ubuntu-vulkan-arm64" {
				t.Errorf("Expected platform ubuntu-arm64 or ubuntu-vulkan-arm64, got %s", result)
			}
		}
	}
}

func TestGetBinaryPattern(t *testing.T) {
	tests := []struct {
		name     string
		tagName  string
		expected string
	}{
		{
			name:     "b7751 release",
			tagName:  "b7751",
			expected: "llama-b7751-bin-" + getPlatform() + ".tar.gz",
		},
		{
			name:     "b7752 release",
			tagName:  "b7752",
			expected: "llama-b7752-bin-" + getPlatform() + ".tar.gz",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			release := &Release{TagName: tt.tagName}
			pattern := getBinaryPattern(release)

			if getPlatform() == "" {
				t.Skip("Skipping test: unsupported platform")
			}

			if pattern != tt.expected {
				t.Errorf("Expected pattern %s, got %s", tt.expected, pattern)
			}
		})
	}
}

func TestServerPath(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)

	os.Setenv("HOME", tmpDir)

	expectedPath := filepath.Join(tmpDir, ".lleme", "bin", "llama-current", "llama-server")
	actualPath := ServerPath()

	if actualPath != expectedPath {
		t.Errorf("Expected ServerPath %s, got %s", expectedPath, actualPath)
	}
}

func TestIsInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	defer os.Setenv("HOME", oldHome)

	os.Setenv("HOME", tmpDir)

	binDir := filepath.Join(tmpDir, ".lleme", "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatalf("Failed to create bin dir: %v", err)
	}

	t.Run("returns false when binary does not exist", func(t *testing.T) {
		if IsInstalled() {
			t.Error("Expected IsInstalled to return false when binary doesn't exist")
		}
	})

	t.Run("returns true when llama-current symlink exists with binary", func(t *testing.T) {
		versionDir := filepath.Join(binDir, "llama-b7751")
		if err := os.MkdirAll(versionDir, 0755); err != nil {
			t.Fatalf("Failed to create version dir: %v", err)
		}

		cliBinary := filepath.Join(versionDir, "llama-cli")
		if err := os.WriteFile(cliBinary, []byte("#!/bin/sh\necho test"), 0755); err != nil {
			t.Fatalf("Failed to create binary: %v", err)
		}

		currentSymlink := filepath.Join(binDir, "llama-current")
		if err := os.Symlink("llama-b7751", currentSymlink); err != nil {
			t.Fatalf("Failed to create llama-current symlink: %v", err)
		}

		if !IsInstalled() {
			t.Error("Expected IsInstalled to return true when llama-current symlink exists")
		}
	})
}

func TestVersionInfo(t *testing.T) {
	t.Run("save and load version info", func(t *testing.T) {
		tmpDir := t.TempDir()
		oldHome := os.Getenv("HOME")
		defer os.Setenv("HOME", oldHome)

		os.Setenv("HOME", tmpDir)

		version := &VersionInfo{
			TagName:     "b7751",
			BinaryPath:  "/test/path/llama-cli",
			InstalledAt: "2024-01-15T10:00:00Z",
		}

		err := SaveVersionInfo(version)
		if err != nil {
			t.Fatalf("Failed to save version info: %v", err)
		}

		loaded, err := GetInstalledVersion()
		if err != nil {
			t.Fatalf("Failed to load version info: %v", err)
		}

		if loaded == nil {
			t.Fatal("Expected loaded version to be non-nil")
		}

		if loaded.TagName != version.TagName {
			t.Errorf("Expected TagName %s, got %s", version.TagName, loaded.TagName)
		}
		if loaded.BinaryPath != version.BinaryPath {
			t.Errorf("Expected BinaryPath %s, got %s", version.BinaryPath, loaded.BinaryPath)
		}
		if loaded.InstalledAt != version.InstalledAt {
			t.Errorf("Expected InstalledAt %s, got %s", version.InstalledAt, loaded.InstalledAt)
		}
	})

	t.Run("returns nil when version file does not exist", func(t *testing.T) {
		tmpDir := t.TempDir()
		oldHome := os.Getenv("HOME")
		defer os.Setenv("HOME", oldHome)

		os.Setenv("HOME", tmpDir)

		version, err := GetInstalledVersion()
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if version != nil {
			t.Error("Expected nil version when file doesn't exist")
		}
	})

	t.Run("returns error for invalid JSON", func(t *testing.T) {
		tmpDir := t.TempDir()
		oldHome := os.Getenv("HOME")
		defer os.Setenv("HOME", oldHome)

		os.Setenv("HOME", tmpDir)

		binDir := filepath.Join(tmpDir, ".lleme", "bin")
		if err := os.MkdirAll(binDir, 0755); err != nil {
			t.Fatalf("Failed to create bin dir: %v", err)
		}

		versionPath := filepath.Join(binDir, "version.json")
		if err := os.WriteFile(versionPath, []byte("invalid json"), 0644); err != nil {
			t.Fatalf("Failed to write invalid version file: %v", err)
		}

		_, err := GetInstalledVersion()
		if err == nil {
			t.Error("Expected error for invalid JSON, got nil")
		}
	})
}

func TestFindAssetForPlatform(t *testing.T) {
	if getPlatform() == "" {
		t.Skip("Skipping test: unsupported platform")
	}

	t.Run("finds matching asset", func(t *testing.T) {
		platform := getPlatform()
		expectedPattern := "llama-b7751-bin-" + platform + ".tar.gz"

		release := &Release{
			TagName: "b7751",
			Assets: []Asset{
				{Name: "llama-b7751-bin-" + platform + ".tar.gz", BrowserDownloadUrl: "http://example.com/" + expectedPattern},
				{Name: "llama-b7751-bin-linux-x64.tar.gz", BrowserDownloadUrl: "http://example.com/linux"},
			},
		}

		url, name, err := FindAssetForPlatform(release)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}

		if name != expectedPattern {
			t.Errorf("Expected asset name %s, got %s", expectedPattern, name)
		}

		if !contains(url, expectedPattern) {
			t.Errorf("Expected URL to contain %s, got %s", expectedPattern, url)
		}
	})

	t.Run("returns error for unsupported platform", func(t *testing.T) {
		release := &Release{
			TagName: "b7751",
			Assets: []Asset{
				{Name: "llama-b7751-bin-windows-x64.zip"},
			},
		}

		_, _, err := FindAssetForPlatform(release)
		if err == nil {
			t.Error("Expected error for unsupported platform, got nil")
		}
	})

	t.Run("returns error when asset not found", func(t *testing.T) {
		release := &Release{
			TagName: "b7751",
			Assets: []Asset{
				{Name: "source.tar.gz"},
			},
		}

		_, _, err := FindAssetForPlatform(release)
		if err == nil {
			t.Error("Expected error when asset not found, got nil")
		}
	})
}

func TestLlamaCurrentSymlinkStructure(t *testing.T) {
	tmpDir := t.TempDir()
	binDir := filepath.Join(tmpDir, "bin")
	versionDir := filepath.Join(binDir, "llama-b7751")

	if err := os.MkdirAll(versionDir, 0755); err != nil {
		t.Fatalf("Failed to create version dir: %v", err)
	}

	t.Run("creates llama-current symlink pointing to version directory", func(t *testing.T) {
		cliBinary := filepath.Join(versionDir, "llama-cli")
		serverBinary := filepath.Join(versionDir, "llama-server")

		if err := os.WriteFile(cliBinary, []byte("#!/bin/sh\necho cli"), 0755); err != nil {
			t.Fatalf("Failed to create CLI binary: %v", err)
		}
		if err := os.WriteFile(serverBinary, []byte("#!/bin/sh\necho server"), 0755); err != nil {
			t.Fatalf("Failed to create server binary: %v", err)
		}

		currentSymlink := filepath.Join(binDir, "llama-current")
		if err := os.Symlink("llama-b7751", currentSymlink); err != nil {
			t.Fatalf("Failed to create llama-current symlink: %v", err)
		}

		if _, err := os.Lstat(currentSymlink); err != nil {
			t.Errorf("Expected llama-current symlink to exist: %v", err)
		}

		// Verify symlink target
		target, err := os.Readlink(currentSymlink)
		if err != nil {
			t.Fatalf("Failed to read symlink: %v", err)
		}
		if target != "llama-b7751" {
			t.Errorf("Expected symlink target llama-b7751, got %s", target)
		}

		// Verify binaries are accessible through llama-current symlink
		if _, err := os.Stat(filepath.Join(binDir, "llama-current", "llama-cli")); err != nil {
			t.Errorf("Expected llama-cli to be accessible through llama-current symlink: %v", err)
		}
		if _, err := os.Stat(filepath.Join(binDir, "llama-current", "llama-server")); err != nil {
			t.Errorf("Expected llama-server to be accessible through llama-current symlink: %v", err)
		}
	})
}

func TestVersionFileJSON(t *testing.T) {
	version := &VersionInfo{
		TagName:     "b7751",
		BinaryPath:  "/path/to/llama-cli",
		InstalledAt: "2024-01-15T10:00:00Z",
	}
	file := VersionFile{Llama: version}

	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal version file: %v", err)
	}

	// Verify JSON structure has "llama" key
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("Failed to unmarshal as map: %v", err)
	}
	if _, ok := raw["llama"]; !ok {
		t.Error("Expected 'llama' key in JSON output")
	}

	var decoded VersionFile
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal version file: %v", err)
	}

	if decoded.Llama == nil {
		t.Fatal("Expected Llama to be non-nil")
	}
	if decoded.Llama.TagName != version.TagName {
		t.Errorf("Expected TagName %s, got %s", version.TagName, decoded.Llama.TagName)
	}
	if decoded.Llama.BinaryPath != version.BinaryPath {
		t.Errorf("Expected BinaryPath %s, got %s", version.BinaryPath, decoded.Llama.BinaryPath)
	}
	if decoded.Llama.InstalledAt != version.InstalledAt {
		t.Errorf("Expected InstalledAt %s, got %s", version.InstalledAt, decoded.Llama.InstalledAt)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && indexOfSubstring(s, substr) >= 0))
}

func indexOfSubstring(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func TestHasVulkanSupport(t *testing.T) {
	// HasVulkanSupport only runs on Linux
	if runtime.GOOS != "linux" {
		t.Run("returns false on non-Linux", func(t *testing.T) {
			if HasVulkanSupport() {
				t.Errorf("HasVulkanSupport() on %s = true, want false", runtime.GOOS)
			}
		})
		return
	}

	// On Linux, just verify it returns a boolean without error
	t.Run("returns boolean on Linux", func(t *testing.T) {
		// Just call it to ensure no panic
		_ = HasVulkanSupport()
	})
}

func TestExtractTarGz(t *testing.T) {
	t.Run("symlinks to correct version by tag name", func(t *testing.T) {
		tmpDir := t.TempDir()
		binDir := filepath.Join(tmpDir, "bin")
		if err := os.MkdirAll(binDir, 0755); err != nil {
			t.Fatalf("Failed to create bin dir: %v", err)
		}

		// Create a tarball containing llama-b8169 directory
		srcDir := filepath.Join(tmpDir, "src")
		llamaDir := filepath.Join(srcDir, "llama-b8169")
		if err := os.MkdirAll(llamaDir, 0755); err != nil {
			t.Fatalf("Failed to create llama dir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(llamaDir, "llama-cli"), []byte("cli"), 0755); err != nil {
			t.Fatalf("Failed to create binary: %v", err)
		}

		archivePath := filepath.Join(tmpDir, "test.tar.gz")
		cmd := exec.Command("tar", "-czf", archivePath, "-C", srcDir, "llama-b8169")
		if err := cmd.Run(); err != nil {
			t.Fatalf("Failed to create archive: %v", err)
		}

		// Pre-create an older version directory to simulate upgrade scenario
		oldDir := filepath.Join(binDir, "llama-b7786")
		if err := os.MkdirAll(oldDir, 0755); err != nil {
			t.Fatalf("Failed to create old dir: %v", err)
		}
		// Create old symlink pointing to old version
		oldLink := filepath.Join(binDir, "llama-current")
		if err := os.Symlink("llama-b7786", oldLink); err != nil {
			t.Fatalf("Failed to create old symlink: %v", err)
		}

		if err := extractTarGz(archivePath, binDir, "b8169"); err != nil {
			t.Fatalf("extractTarGz failed: %v", err)
		}

		// Verify symlink points to new version, not old
		target, err := os.Readlink(filepath.Join(binDir, "llama-current"))
		if err != nil {
			t.Fatalf("Failed to read symlink: %v", err)
		}
		if target != "llama-b8169" {
			t.Errorf("Expected symlink target llama-b8169, got %s", target)
		}
	})

	t.Run("returns error when expected directory missing", func(t *testing.T) {
		tmpDir := t.TempDir()
		binDir := filepath.Join(tmpDir, "bin")
		if err := os.MkdirAll(binDir, 0755); err != nil {
			t.Fatalf("Failed to create bin dir: %v", err)
		}

		// Create a tarball with a different directory name
		srcDir := filepath.Join(tmpDir, "src")
		llamaDir := filepath.Join(srcDir, "llama-b7786")
		if err := os.MkdirAll(llamaDir, 0755); err != nil {
			t.Fatalf("Failed to create llama dir: %v", err)
		}

		archivePath := filepath.Join(tmpDir, "test.tar.gz")
		cmd := exec.Command("tar", "-czf", archivePath, "-C", srcDir, "llama-b7786")
		if err := cmd.Run(); err != nil {
			t.Fatalf("Failed to create archive: %v", err)
		}

		err := extractTarGz(archivePath, binDir, "b9999")
		if err == nil {
			t.Error("Expected error when expected directory is missing")
		}
	})
}

func TestRemoveOldVersions(t *testing.T) {
	t.Run("removes old llama-b* directories", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create current and old version directories
		os.MkdirAll(filepath.Join(tmpDir, "llama-b8169"), 0755)
		os.MkdirAll(filepath.Join(tmpDir, "llama-b7786"), 0755)
		os.MkdirAll(filepath.Join(tmpDir, "llama-b7000"), 0755)
		// Non-matching entries should be left alone
		os.MkdirAll(filepath.Join(tmpDir, "llama-current"), 0755)
		os.WriteFile(filepath.Join(tmpDir, "version.json"), []byte("{}"), 0644)

		removeOldVersions(tmpDir, "b8169")

		// Current version should remain
		if _, err := os.Stat(filepath.Join(tmpDir, "llama-b8169")); err != nil {
			t.Error("Current version directory should not be removed")
		}

		// Old versions should be gone
		if _, err := os.Stat(filepath.Join(tmpDir, "llama-b7786")); !os.IsNotExist(err) {
			t.Error("Old version llama-b7786 should be removed")
		}
		if _, err := os.Stat(filepath.Join(tmpDir, "llama-b7000")); !os.IsNotExist(err) {
			t.Error("Old version llama-b7000 should be removed")
		}

		// Non-matching entries should remain
		if _, err := os.Stat(filepath.Join(tmpDir, "llama-current")); err != nil {
			t.Error("llama-current should not be removed")
		}
		if _, err := os.Stat(filepath.Join(tmpDir, "version.json")); err != nil {
			t.Error("version.json should not be removed")
		}
	})
}

func TestPruneOldVersionsKeepPrior(t *testing.T) {
	t.Run("retains the N most recent prior directories by mtime", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create version dirs and set distinct mtimes so newest-first ordering is deterministic.
		type entry struct {
			name  string
			mtime time.Time
		}
		now := time.Now()
		versions := []entry{
			{"llama-b8169", now},
			{"llama-b8168", now.Add(-1 * time.Hour)},
			{"llama-b8167", now.Add(-2 * time.Hour)},
			{"llama-b8166", now.Add(-3 * time.Hour)},
			{"llama-b8165", now.Add(-4 * time.Hour)},
		}
		for _, v := range versions {
			dir := filepath.Join(tmpDir, v.name)
			if err := os.MkdirAll(dir, 0755); err != nil {
				t.Fatalf("Failed to create %s: %v", v.name, err)
			}
			if err := os.Chtimes(dir, v.mtime, v.mtime); err != nil {
				t.Fatalf("Failed to set mtime on %s: %v", v.name, err)
			}
		}

		// keepPrior=2 should retain b8168 and b8167 alongside the current (b8169).
		pruneOldVersions(tmpDir, "b8169", 2)

		for _, keep := range []string{"llama-b8169", "llama-b8168", "llama-b8167"} {
			if _, err := os.Stat(filepath.Join(tmpDir, keep)); err != nil {
				t.Errorf("%s should be retained", keep)
			}
		}
		for _, prune := range []string{"llama-b8166", "llama-b8165"} {
			if _, err := os.Stat(filepath.Join(tmpDir, prune)); !os.IsNotExist(err) {
				t.Errorf("%s should be pruned", prune)
			}
		}
	})
}

func TestNewerVersionAvailable(t *testing.T) {
	t.Run("returns no-op when not installed", func(t *testing.T) {
		t.Setenv("LLEME_HOME", t.TempDir())

		latest, installed, err := NewerVersionAvailable()
		if err != nil {
			t.Fatalf("Expected no error when not installed, got %v", err)
		}
		if latest != nil {
			t.Error("Expected no latest version when not installed")
		}
		if installed != nil {
			t.Error("Expected installed to be nil when not installed")
		}
	})

	t.Run("returns nil latest when already up-to-date", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("LLEME_HOME", home)
		writeInstalledVersion(t, "b8169")

		srv := mockGitHubLatest(t, "b8169")
		defer srv.Close()
		withAPIBase(t, srv.URL)

		latest, installed, err := NewerVersionAvailable()
		if err != nil {
			t.Fatalf("Expected no error when up-to-date, got %v", err)
		}
		if latest != nil {
			t.Errorf("Expected no latest when up-to-date, got %+v", latest)
		}
		if installed == nil || installed.TagName != "b8169" {
			t.Errorf("Expected installed=b8169, got %+v", installed)
		}
	})

	t.Run("returns latest when newer is available", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("LLEME_HOME", home)
		writeInstalledVersion(t, "b8169")

		srv := mockGitHubLatest(t, "b8170")
		defer srv.Close()
		withAPIBase(t, srv.URL)

		latest, installed, err := NewerVersionAvailable()
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		if latest == nil || latest.TagName != "b8170" {
			t.Errorf("Expected latest=b8170, got %+v", latest)
		}
		if installed == nil || installed.TagName != "b8169" {
			t.Errorf("Expected installed=b8169, got %+v", installed)
		}
	})

	t.Run("propagates fetch error but still reports installed", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("LLEME_HOME", home)
		writeInstalledVersion(t, "b8169")

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer srv.Close()
		withAPIBase(t, srv.URL)

		latest, installed, err := NewerVersionAvailable()
		if err == nil {
			t.Fatal("Expected error when fetch fails")
		}
		if latest != nil {
			t.Errorf("Expected no latest on error, got %+v", latest)
		}
		if installed == nil || installed.TagName != "b8169" {
			t.Errorf("Expected installed reported even on fetch error, got %+v", installed)
		}
	})
}

func TestGetLatestVersionRejectsBadTag(t *testing.T) {
	tests := []struct {
		name    string
		tag     string
		wantErr bool
	}{
		{"valid tag", "b8169", false},
		{"valid high-numbered tag", "b999999", false},
		{"path traversal in tag", "../../../etc/passwd", true},
		{"shell metachar", "b8169; rm -rf /", true},
		{"empty tag", "", true},
		{"non-b prefix", "v1.0.0", true},
		{"embedded slash", "b8169/extra", true},
		{"leading slash", "/b8169", true},
		{"wrong case", "B8169", true},
		{"non-numeric suffix", "babc", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := mockGitHubLatest(t, tt.tag)
			defer srv.Close()
			withAPIBase(t, srv.URL)

			_, err := GetLatestVersion()
			if tt.wantErr && err == nil {
				t.Errorf("Expected error for tag %q, got nil", tt.tag)
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Expected no error for tag %q, got %v", tt.tag, err)
			}
		})
	}
}

func TestValidateDownloadURL(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{"github.com https", "https://github.com/ggml-org/llama.cpp/releases/download/b8169/asset.tar.gz", false},
		{"objects.githubusercontent.com", "https://objects.githubusercontent.com/foo", false},
		{"release-assets.githubusercontent.com", "https://release-assets.githubusercontent.com/foo", false},
		{"http scheme rejected in prod", "http://github.com/foo", true},
		{"file scheme rejected", "file:///etc/passwd", true},
		{"attacker host rejected", "https://attacker.example.com/evil.tar.gz", true},
		{"host-spoof with github in path rejected", "https://attacker.example.com/github.com/foo", true},
		{"empty URL rejected", "", true},
		{"malformed URL rejected", "://not-a-url", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateDownloadURL(tt.url)
			if tt.wantErr && err == nil {
				t.Errorf("Expected error for %q, got nil", tt.url)
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Expected no error for %q, got %v", tt.url, err)
			}
		})
	}
}

func TestDownloadBinaryRejectsBadURL(t *testing.T) {
	dest := filepath.Join(t.TempDir(), "out.tar.gz")
	err := DownloadBinary("https://attacker.example.com/evil.tar.gz", dest, nil)
	if err == nil {
		t.Fatal("Expected error for disallowed host")
	}
	if !strings.Contains(err.Error(), "allowlist") {
		t.Errorf("Expected allowlist error, got %v", err)
	}
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Error("Expected no file to be created for rejected URL")
	}
}

func TestDownloadBinaryEnforcesSizeCap(t *testing.T) {
	t.Run("content-length over cap is rejected before streaming starts", func(t *testing.T) {
		withMaxDownloadBytes(t, 1024)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", maxDownloadBytes+1))
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()
		withDownloadHost(t, srv)

		dest := filepath.Join(t.TempDir(), "out.tar.gz")
		err := DownloadBinary(srv.URL+"/asset.tar.gz", dest, nil)
		if err == nil || !strings.Contains(err.Error(), "exceeds max") {
			t.Errorf("Expected content-length cap error, got %v", err)
		}
	})

	t.Run("streaming guard fires when content-length is absent", func(t *testing.T) {
		withMaxDownloadBytes(t, 512)
		// With no declared Content-Length, Go uses chunked transfer and we can
		// stream past the client's cap. The LimitReader + post-write check
		// must catch this.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			chunk := []byte(strings.Repeat("A", 512))
			for range 16 {
				if _, err := w.Write(chunk); err != nil {
					return
				}
				flusher.Flush()
			}
		}))
		defer srv.Close()
		withDownloadHost(t, srv)

		dest := filepath.Join(t.TempDir(), "out.tar.gz")
		err := DownloadBinary(srv.URL+"/asset.tar.gz", dest, nil)
		if err == nil || !strings.Contains(err.Error(), "exceeded max size") {
			t.Errorf("Expected streaming overflow error, got %v", err)
		}
		// The partial file must not have been promoted to the final path.
		if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
			t.Error("Expected no file at dest after overflow rejection")
		}
	})

	t.Run("legitimate payload under cap succeeds", func(t *testing.T) {
		withMaxDownloadBytes(t, 1<<20) // 1 MiB
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			payload := []byte(strings.Repeat("A", 1024))
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(payload)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(payload)
		}))
		defer srv.Close()
		withDownloadHost(t, srv)

		dest := filepath.Join(t.TempDir(), "out.tar.gz")
		if err := DownloadBinary(srv.URL+"/asset.tar.gz", dest, nil); err != nil {
			t.Errorf("Expected success for under-cap payload, got %v", err)
		}
		info, err := os.Stat(dest)
		if err != nil {
			t.Fatalf("Expected dest file to exist, got %v", err)
		}
		if info.Size() != 1024 {
			t.Errorf("Expected 1024 bytes, got %d", info.Size())
		}
	})
}

func TestDownloadBinaryContextCancellation(t *testing.T) {
	// Server streams slowly so the client cancel arrives mid-body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000000")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		buf := make([]byte, 1024)
		for {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			if _, err := w.Write(buf); err != nil {
				return
			}
			flusher.Flush()
			time.Sleep(20 * time.Millisecond)
		}
	}))
	defer srv.Close()
	withDownloadHost(t, srv)

	ctx, cancel := context.WithCancel(context.Background())
	dest := filepath.Join(t.TempDir(), "out.tar.gz")

	errCh := make(chan error, 1)
	go func() {
		errCh <- DownloadBinaryContext(ctx, srv.URL+"/asset.tar.gz", dest, nil)
	}()

	// Give the request time to reach the server, then cancel.
	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		if err == nil {
			t.Error("Expected error on canceled context")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Download did not abort after context cancel")
	}
}

func TestDownloadBinaryBlocksDisallowedRedirect(t *testing.T) {
	// Allowed entry issues a 302 to an off-allowlist host. The CheckRedirect
	// callback must reject before the HTTP client connects to the attacker.
	const attackerURL = "https://attacker.example.com/evil.tar.gz"
	entry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, attackerURL, http.StatusFound)
	}))
	defer entry.Close()
	withDownloadHost(t, entry)

	dest := filepath.Join(t.TempDir(), "out.tar.gz")
	err := DownloadBinary(entry.URL+"/asset.tar.gz", dest, nil)
	if err == nil {
		t.Fatal("Expected redirect to attacker host to be blocked")
	}
	if !strings.Contains(err.Error(), "redirect blocked") {
		t.Errorf("Expected redirect blocked error, got %v", err)
	}
}

func TestSwapCurrentSymlinkAtomic(t *testing.T) {
	t.Run("points llama-current at the target", func(t *testing.T) {
		tmpDir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(tmpDir, "llama-b8169"), 0755); err != nil {
			t.Fatal(err)
		}

		if err := swapCurrentSymlink(tmpDir, "llama-b8169"); err != nil {
			t.Fatalf("swapCurrentSymlink failed: %v", err)
		}

		target, err := os.Readlink(filepath.Join(tmpDir, "llama-current"))
		if err != nil {
			t.Fatalf("readlink: %v", err)
		}
		if target != "llama-b8169" {
			t.Errorf("Expected symlink target llama-b8169, got %q", target)
		}
	})

	t.Run("replaces an existing symlink without an ENOENT window", func(t *testing.T) {
		tmpDir := t.TempDir()
		for _, name := range []string{"llama-b8168", "llama-b8169"} {
			if err := os.MkdirAll(filepath.Join(tmpDir, name), 0755); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.Symlink("llama-b8168", filepath.Join(tmpDir, "llama-current")); err != nil {
			t.Fatal(err)
		}

		// Spin a reader goroutine that stats the symlink in a tight loop;
		// assert it never observes ENOENT across the swap.
		done := make(chan struct{})
		var wg sync.WaitGroup
		var enoentSeen int64
		for range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					select {
					case <-done:
						return
					default:
					}
					_, err := os.Readlink(filepath.Join(tmpDir, "llama-current"))
					if err != nil && errors.Is(err, os.ErrNotExist) {
						atomic.AddInt64(&enoentSeen, 1)
					}
				}
			}()
		}

		// Perform many swaps to widen the observation window.
		for range 200 {
			if err := swapCurrentSymlink(tmpDir, "llama-b8169"); err != nil {
				t.Fatalf("swap to b8169 failed: %v", err)
			}
			if err := swapCurrentSymlink(tmpDir, "llama-b8168"); err != nil {
				t.Fatalf("swap to b8168 failed: %v", err)
			}
		}
		close(done)
		wg.Wait()

		if atomic.LoadInt64(&enoentSeen) != 0 {
			t.Errorf("llama-current disappeared %d times during swap — swap is not atomic", enoentSeen)
		}
	})

	t.Run("clears stale tmp from a prior failed run", func(t *testing.T) {
		tmpDir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(tmpDir, "llama-b8169"), 0755); err != nil {
			t.Fatal(err)
		}
		// Plant a stale tmp symlink as if a prior run crashed mid-stage.
		if err := os.Symlink("llama-b0000", filepath.Join(tmpDir, ".llama-current.tmp")); err != nil {
			t.Fatal(err)
		}

		if err := swapCurrentSymlink(tmpDir, "llama-b8169"); err != nil {
			t.Fatalf("Expected recovery from stale tmp, got %v", err)
		}
		target, err := os.Readlink(filepath.Join(tmpDir, "llama-current"))
		if err != nil {
			t.Fatalf("readlink: %v", err)
		}
		if target != "llama-b8169" {
			t.Errorf("Expected target llama-b8169 after recovery, got %q", target)
		}
	})
}

func TestPruneRespectsSymlinkTarget(t *testing.T) {
	t.Run("never deletes the directory llama-current points at", func(t *testing.T) {
		tmpDir := t.TempDir()
		now := time.Now()
		// b8165 is oldest by mtime — would normally be pruned with keepPrior=1.
		// We'll point llama-current at it and pass a DIFFERENT currentTag to
		// pruneOldVersions to simulate stale state. The symlink target must
		// still survive.
		versions := []struct {
			name  string
			mtime time.Time
		}{
			{"llama-b8169", now},
			{"llama-b8168", now.Add(-1 * time.Hour)},
			{"llama-b8167", now.Add(-2 * time.Hour)},
			{"llama-b8165", now.Add(-10 * time.Hour)},
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
		if err := os.Symlink("llama-b8165", filepath.Join(tmpDir, "llama-current")); err != nil {
			t.Fatal(err)
		}

		// currentTag says b8169, but the symlink actually resolves to b8165.
		// keepPrior=1 would normally save only b8168; b8165 should survive
		// anyway because it's the symlink target.
		pruneOldVersions(tmpDir, "b8169", 1)

		if _, err := os.Stat(filepath.Join(tmpDir, "llama-b8165")); err != nil {
			t.Error("Symlink target should be retained even with mismatched currentTag")
		}
		if _, err := os.Stat(filepath.Join(tmpDir, "llama-b8169")); err != nil {
			t.Error("currentTag directory should be retained")
		}
		if _, err := os.Stat(filepath.Join(tmpDir, "llama-b8168")); err != nil {
			t.Error("Most-recent prior (b8168) should be retained with keepPrior=1")
		}
		if _, err := os.Stat(filepath.Join(tmpDir, "llama-b8167")); !os.IsNotExist(err) {
			t.Error("b8167 should be pruned when keepPrior=1")
		}
	})
}

// --- helpers ---

func withAPIBase(t *testing.T, url string) {
	t.Helper()
	old := apiBase
	apiBase = url
	t.Cleanup(func() { apiBase = old })
}

func withDownloadHost(t *testing.T, srv *httptest.Server) {
	t.Helper()
	u, err := urlHostnameFromHTTPTest(srv)
	if err != nil {
		t.Fatal(err)
	}
	allowedDownloadHosts[u] = true
	allowedDownloadSchemes["http"] = true
	t.Cleanup(func() {
		delete(allowedDownloadHosts, u)
		delete(allowedDownloadSchemes, "http")
	})
}

func urlHostnameFromHTTPTest(srv *httptest.Server) (string, error) {
	u, err := url.Parse(srv.URL)
	if err != nil {
		return "", fmt.Errorf("unexpected httptest URL %q: %w", srv.URL, err)
	}
	return u.Hostname(), nil
}

func withMaxDownloadBytes(t *testing.T, limit int64) {
	t.Helper()
	old := maxDownloadBytes
	maxDownloadBytes = limit
	t.Cleanup(func() { maxDownloadBytes = old })
}

func mockGitHubLatest(t *testing.T, tagName string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/releases/latest") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		body := fmt.Sprintf(`{"tag_name": %q, "name": "release", "assets": []}`, tagName)
		_, _ = io.WriteString(w, body)
	}))
}

func writeInstalledVersion(t *testing.T, tag string) {
	t.Helper()
	binDir := filepath.Join(os.Getenv("LLEME_HOME"), "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}
	info := VersionFile{Llama: &VersionInfo{TagName: tag, BinaryPath: "/dev/null", InstalledAt: time.Now().Format(time.RFC3339)}}
	data, err := json.Marshal(info)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "version.json"), data, 0644); err != nil {
		t.Fatal(err)
	}
}

func TestGetPlatformLinuxVariants(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Skipping Linux-specific test on non-Linux platform")
	}

	var plain, vulkan string
	switch runtime.GOARCH {
	case "amd64":
		plain, vulkan = "ubuntu-x64", "ubuntu-vulkan-x64"
	case "arm64":
		plain, vulkan = "ubuntu-arm64", "ubuntu-vulkan-arm64"
	default:
		t.Skipf("Skipping Linux test on unsupported arch %s", runtime.GOARCH)
	}

	result := getPlatform()
	validPlatforms := []string{plain, vulkan}
	if !slices.Contains(validPlatforms, result) {
		t.Errorf("getPlatform() on Linux %s = %q, want one of %v", runtime.GOARCH, result, validPlatforms)
	}

	// Platform selection is based on libvulkan.so availability, not GPU detection
	hasVulkan := HasVulkanSupport()
	if hasVulkan && result != vulkan {
		t.Errorf("Vulkan support detected but platform is %q, expected %s", result, vulkan)
	}
	if !hasVulkan && result != plain {
		t.Errorf("No Vulkan support but platform is %q, expected %s", result, plain)
	}
}
