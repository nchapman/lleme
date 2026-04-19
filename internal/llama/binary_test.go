package llama

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
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
