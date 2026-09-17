package llama

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/nchapman/lleme/internal/binaryrelease"
)

// buildFakeReleaseTarball packs a llama-<tag> directory (llama-cli +
// llama-server executables plus a relative symlink, mirroring real release
// archives) into the asset name getPlatform() expects for this OS/arch.
func buildFakeReleaseTarball(t *testing.T, tag string) (filename string, body []byte) {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)

	dir := "llama-" + tag + "/"
	for _, f := range []struct {
		name string
		mode int64
	}{
		{dir + "llama-cli", 0755},
		{dir + "llama-server", 0755},
	} {
		if err := tw.WriteHeader(&tar.Header{Name: f.name, Mode: f.mode, Size: 8, Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte("binary!\n")); err != nil {
			t.Fatal(err)
		}
	}
	// Real archives ship relative versioned-library symlinks.
	if err := tw.WriteHeader(&tar.Header{Name: dir + "libllama.dylib", Linkname: "libllama.0.dylib", Typeflag: tar.TypeSymlink}); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}

	release := &Release{TagName: tag}
	return getBinaryPattern(release), buf.Bytes()
}

func withFakeReleaseServer(t *testing.T, tag string) {
	t.Helper()
	assetName, tarball := buildFakeReleaseTarball(t, tag)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch p := r.URL.Path; p {
		case "/releases/latest":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(Release{
				TagName: tag,
				Name:    tag,
				Assets: []binaryrelease.Asset{{
					Name:               assetName,
					Size:               int64(len(tarball)),
					BrowserDownloadURL: "http://" + r.Host + "/asset/" + assetName,
				}},
			})
		case "/asset/" + assetName:
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(tarball)))
			_, _ = w.Write(tarball)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)

	withAPIBase(t, srv.URL)
	withDownloadHost(t, srv)
}

func TestInstallLatestEndToEnd(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix-specific install layout")
	}
	t.Setenv("LLEME_HOME", t.TempDir())
	withFakeReleaseServer(t, "b99999")

	info, err := InstallLatest(nil)
	if err != nil {
		t.Fatalf("InstallLatest: %v", err)
	}
	if info.TagName != "b99999" {
		t.Errorf("TagName = %q, want b99999", info.TagName)
	}

	// The symlinked layout must be active and version.json persisted.
	cli := filepath.Join(os.Getenv("LLEME_HOME"), "bin", "llama-current", "llama-cli")
	if info.BinaryPath != cli {
		t.Errorf("BinaryPath = %q, want %q", info.BinaryPath, cli)
	}
	if !IsInstalled() {
		t.Error("IsInstalled() = false after install")
	}
	linkTarget, err := os.Readlink(filepath.Join(os.Getenv("LLEME_HOME"), "bin", "llama-current"))
	if err != nil {
		t.Fatalf("llama-current symlink missing: %v", err)
	}
	if linkTarget != "llama-b99999" {
		t.Errorf("llama-current -> %q, want llama-b99999", linkTarget)
	}

	installed, err := GetInstalledVersion()
	if err != nil || installed == nil {
		t.Fatalf("GetInstalledVersion: %v, %v", installed, err)
	}
	if installed.TagName != "b99999" {
		t.Errorf("version.json tag = %q, want b99999", installed.TagName)
	}

	// Archive must have been cleaned up.
	asset, _ := buildFakeReleaseTarball(t, "b99999")
	if _, err := os.Stat(filepath.Join(os.Getenv("LLEME_HOME"), "bin", asset)); !os.IsNotExist(err) {
		t.Error("release archive left behind after install")
	}
}

func TestInstallReleaseForAutoUpdateKeepsPriorVersions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix-specific install layout")
	}
	t.Setenv("LLEME_HOME", t.TempDir())
	withFakeReleaseServer(t, "b88888")

	release, err := GetLatestVersion()
	if err != nil {
		t.Fatalf("GetLatestVersion: %v", err)
	}

	// Simulate a prior version directory still in use by a live backend.
	binDir := filepath.Join(os.Getenv("LLEME_HOME"), "bin")
	oldDir := filepath.Join(binDir, "llama-b77777")
	if err := os.MkdirAll(oldDir, 0755); err != nil {
		t.Fatal(err)
	}

	if _, err := InstallReleaseForAutoUpdate(t.Context(), release, nil); err != nil {
		t.Fatalf("InstallReleaseForAutoUpdate: %v", err)
	}

	if _, err := os.Stat(oldDir); err != nil {
		t.Errorf("prior version directory pruned by auto-update install: %v", err)
	}
}
