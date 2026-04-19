package binaryrelease

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testConfig(t *testing.T, srv *httptest.Server) Config {
	t.Helper()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse srv url: %v", err)
	}
	return Config{
		AllowedHosts:   map[string]bool{u.Hostname(): true},
		AllowedSchemes: map[string]bool{"http": true},
		MaxBytes:       1 << 20,
		UserAgent:      "lleme-test",
	}
}

func TestValidateURL(t *testing.T) {
	cfg := Config{
		AllowedHosts:   map[string]bool{"github.com": true},
		AllowedSchemes: map[string]bool{"https": true},
	}
	tests := []struct {
		name    string
		url     string
		wantErr string
	}{
		{"good", "https://github.com/foo/bar/releases/download/v1/a.tar.gz", ""},
		{"bad scheme", "http://github.com/x", "scheme \"http\""},
		{"bad host", "https://evil.example.com/x", "host \"evil.example.com\""},
		{"bad url", "://not a url", "invalid URL"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateURL(cfg, tt.url)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("got %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("err = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestFetchLatestRelease(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ua := r.Header.Get("User-Agent"); ua != "lleme-test" {
			t.Errorf("User-Agent = %q, want lleme-test", ua)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"tag_name":"b517","name":"Release","assets":[{"name":"a.tar.gz","size":1024,"browser_download_url":"https://example.invalid/a.tar.gz"}]}`)
	}))
	defer srv.Close()

	cfg := testConfig(t, srv)
	rel, err := FetchLatestRelease(cfg, srv.URL+"/releases/latest")
	if err != nil {
		t.Fatal(err)
	}
	if rel.TagName != "b517" || len(rel.Assets) != 1 || rel.Assets[0].Name != "a.tar.gz" {
		t.Errorf("unexpected release: %+v", rel)
	}
}

func TestFetchLatestReleaseRejectsDisallowedHost(t *testing.T) {
	cfg := Config{
		AllowedHosts:   map[string]bool{"api.github.com": true},
		AllowedSchemes: DefaultHTTPSOnly(),
	}
	_, err := FetchLatestRelease(cfg, "https://evil.example.com/repos/x/y/releases/latest")
	if err == nil || !strings.Contains(err.Error(), "allowlist") {
		t.Errorf("err = %v, want allowlist rejection", err)
	}
}

func TestDownload(t *testing.T) {
	payload := strings.Repeat("x", 300)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(payload)))
		fmt.Fprint(w, payload)
	}))
	defer srv.Close()

	cfg := testConfig(t, srv)
	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := Download(context.Background(), cfg, srv.URL+"/artifact", dest, nil); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read dest: %v", err)
	}
	if string(data) != payload {
		t.Errorf("got %d bytes, want %d", len(data), len(payload))
	}
}

func TestDownloadRejectsOversizedContentLength(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "99999")
		fmt.Fprint(w, "small")
	}))
	defer srv.Close()

	cfg := testConfig(t, srv)
	cfg.MaxBytes = 100
	err := Download(context.Background(), cfg, srv.URL+"/artifact", filepath.Join(t.TempDir(), "x"), nil)
	if err == nil || !strings.Contains(err.Error(), "exceeds max") {
		t.Errorf("err = %v, want content-length exceeded", err)
	}
}

func TestDownloadRejectsOversizedBody(t *testing.T) {
	// Chunked response: no Content-Length header, so the first guard can't
	// catch it and the streamed-bytes guard must.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
		_, _ = w.Write([]byte(strings.Repeat("y", 500)))
	}))
	defer srv.Close()

	cfg := testConfig(t, srv)
	cfg.MaxBytes = 50
	dir := t.TempDir()
	dest := filepath.Join(dir, "x")
	err := Download(context.Background(), cfg, srv.URL+"/artifact", dest, nil)
	if err == nil || !strings.Contains(err.Error(), "max size") {
		t.Errorf("err = %v, want body exceeded", err)
	}
	if _, err := os.Stat(dest + ".partial"); !os.IsNotExist(err) {
		t.Errorf(".partial file should be cleaned up after size rejection (err: %v)", err)
	}
}

func TestDownloadRejectsDisallowedHost(t *testing.T) {
	cfg := Config{
		AllowedHosts:   map[string]bool{"github.com": true},
		AllowedSchemes: map[string]bool{"https": true},
	}
	err := Download(context.Background(), cfg, "https://evil.example.com/a", "/tmp/x", nil)
	if err == nil || !strings.Contains(err.Error(), "allowlist") {
		t.Errorf("err = %v, want allowlist rejection", err)
	}
}

func TestSwapCurrentSymlink(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "v1"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "v2"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := SwapCurrentSymlink(dir, "current", "v1"); err != nil {
		t.Fatal(err)
	}
	target, err := os.Readlink(filepath.Join(dir, "current"))
	if err != nil || target != "v1" {
		t.Errorf("first swap: target=%q err=%v", target, err)
	}

	// Second swap replaces the link atomically.
	if err := SwapCurrentSymlink(dir, "current", "v2"); err != nil {
		t.Fatal(err)
	}
	target, err = os.Readlink(filepath.Join(dir, "current"))
	if err != nil || target != "v2" {
		t.Errorf("second swap: target=%q err=%v", target, err)
	}

	// A leftover .current.tmp should not block a swap.
	tmp := filepath.Join(dir, ".current.tmp")
	if err := os.Symlink("stale", tmp); err != nil {
		t.Fatal(err)
	}
	if err := SwapCurrentSymlink(dir, "current", "v1"); err != nil {
		t.Errorf("swap with stale tmp: %v", err)
	}
}
