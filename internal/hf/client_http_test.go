package hf

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nchapman/lleme/internal/config"
)

// withHFServer points the client's base URLs at a test server for the
// duration of the test.
func withHFServer(t *testing.T, srv *httptest.Server) {
	t.Helper()
	oldBase, oldAPI := baseURL, apiBase
	baseURL = srv.URL
	apiBase = srv.URL + "/api"
	t.Cleanup(func() {
		baseURL, apiBase = oldBase, oldAPI
	})
}

func TestGetModelHTTP(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/testuser/model-repo" {
			http.NotFound(w, r)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer tok-123" {
			t.Errorf("Authorization = %q, want Bearer tok-123", got)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"modelId":"testuser/model-repo","author":"testuser","gated":false,"downloads":42,"likes":7,"tags":["gguf"],"cardData":{"license":"apache-2.0"}}`)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	cfg := &config.Config{}
	cfg.HuggingFace.Token = "tok-123"
	client := NewClient(cfg)

	info, err := client.GetModel("testuser", "model-repo")
	if err != nil {
		t.Fatal(err)
	}
	if info.ModelId != "testuser/model-repo" || info.Author != "testuser" {
		t.Errorf("unexpected model info: %+v", info)
	}
	if info.Downloads != 42 || info.Likes != 7 {
		t.Errorf("stats not decoded: %+v", info)
	}
	if len(info.Tags) != 1 || info.CardData.License != "apache-2.0" {
		t.Errorf("tags/license not decoded: %+v", info)
	}
}

func TestGetModelHTTPNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	if _, err := NewClient(&config.Config{}).GetModel("x", "y"); err == nil || !strings.Contains(err.Error(), "404") {
		t.Errorf("err = %v, want HTTP 404 error", err)
	}
}

func TestListFilesHTTP(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/testuser/repo/tree/main" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `[
			{"path":"model.Q4_K_M.gguf","type":"file","size":100,"lfs":{"oid":"abc","size":100}},
			{"path":"Q6_K","type":"directory","size":0}
		]`)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	files, err := NewClient(&config.Config{}).ListFiles("testuser", "repo", "main")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Fatalf("got %d files, want 2: %+v", len(files), files)
	}
	if files[0].Path != "model.Q4_K_M.gguf" || files[0].LFS.Size != 100 {
		t.Errorf("file[0] not decoded: %+v", files[0])
	}
	if files[1].Type != "directory" {
		t.Errorf("file[1].Type = %q, want directory", files[1].Type)
	}
}

func TestSearchModelsHTTP(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models-json" {
			http.NotFound(w, r)
			return
		}
		if apps := r.URL.Query().Get("apps"); apps != "llama.cpp" {
			t.Errorf("apps = %q, want llama.cpp", apps)
		}
		if q := r.URL.Query().Get("search"); q != "gemma" {
			t.Errorf("search = %q, want gemma", q)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"models":[{"id":"u/one","downloads":5,"likes":1},{"id":"u/two","downloads":6,"likes":2},{"id":"u/three","downloads":7,"likes":3}]}`)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	results, err := NewClient(&config.Config{}).SearchModels("gemma", 2, []string{"llama.cpp"})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want limit 2 applied", len(results))
	}
	if results[0].ID != "u/one" || results[0].Downloads != 5 {
		t.Errorf("results[0] not decoded: %+v", results[0])
	}
}

func TestGetManifestHTTP(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/testuser/repo/manifests/Q4_K_M" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ggufFile":{"rfilename":"model.Q4_K_M.gguf","size":123},"mmprojFile":{"rfilename":"mmproj.gguf","size":456}}`)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	manifest, raw, err := NewClient(&config.Config{}).GetManifest("testuser", "repo", "Q4_K_M")
	if err != nil {
		t.Fatal(err)
	}
	if manifest.GGUFFile == nil || manifest.GGUFFile.RFilename != "model.Q4_K_M.gguf" {
		t.Errorf("ggufFile not decoded: %+v", manifest.GGUFFile)
	}
	if manifest.MMProjFile == nil || manifest.MMProjFile.Size != 456 {
		t.Errorf("mmprojFile not decoded: %+v", manifest.MMProjFile)
	}
	if len(raw) == 0 {
		t.Error("raw JSON bytes not returned")
	}
}

func TestDownloadModelFresh(t *testing.T) {
	content := strings.Repeat("lleme-test-payload-", 40) // 760 bytes
	var gotRange string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/testuser/repo/resolve/main/model.Q4_K_M.gguf" {
			http.NotFound(w, r)
			return
		}
		gotRange = r.Header.Get("Range")
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(content)))
		fmt.Fprint(w, content)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	dest := filepath.Join(t.TempDir(), "model.Q4_K_M.gguf")
	progress, err := NewDownloaderWithProgress(NewClient(&config.Config{}), nil).DownloadModel(
		"testuser", "repo", "main", "model.Q4_K_M.gguf", dest, int64(len(content)))
	if err != nil {
		t.Fatal(err)
	}
	if gotRange != "" {
		t.Errorf("fresh download sent Range %q, want none", gotRange)
	}
	data, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != content {
		t.Errorf("downloaded %d bytes, content mismatch", len(data))
	}
	if progress.Downloaded != progress.Total {
		t.Errorf("progress = %d/%d, want complete", progress.Downloaded, progress.Total)
	}
	if _, err := os.Stat(dest + ".partial"); !os.IsNotExist(err) {
		t.Error(".partial file left behind after successful download")
	}
}

func TestDownloadModelResumeWithRange(t *testing.T) {
	content := strings.Repeat("0123456789", 100) // 1000 bytes
	existing := 300
	var gotRange string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRange = r.Header.Get("Range")
		if gotRange != fmt.Sprintf("bytes=%d-", existing) {
			http.Error(w, "expected resume range", http.StatusBadRequest)
			return
		}
		rest := content[existing:]
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(rest)))
		w.WriteHeader(http.StatusPartialContent)
		fmt.Fprint(w, rest)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(dest+".partial", []byte(content[:existing]), 0644); err != nil {
		t.Fatal(err)
	}

	if _, err := NewDownloaderWithProgress(NewClient(&config.Config{}), nil).DownloadModel(
		"testuser", "repo", "main", "model.gguf", dest, int64(len(content))); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != content {
		t.Errorf("resumed download produced %d bytes with wrong content", len(data))
	}
}

// Regression: a server that ignores the Range header answers 200 with the
// full body. The downloader must restart from zero instead of appending the
// full body onto the stale .partial prefix.
func TestDownloadModelServerIgnoresRange(t *testing.T) {
	content := strings.Repeat("abcdef", 100) // 600 bytes
	existing := 200
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") == "" {
			http.Error(w, "expected Range on resume", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(content)))
		fmt.Fprint(w, content) // 200, not 206: Range ignored
	}))
	defer srv.Close()
	withHFServer(t, srv)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(dest+".partial", []byte(content[:existing]), 0644); err != nil {
		t.Fatal(err)
	}

	if _, err := NewDownloaderWithProgress(NewClient(&config.Config{}), nil).DownloadModel(
		"testuser", "repo", "main", "model.gguf", dest, int64(len(content))); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != content {
		t.Errorf("got %d bytes, want exactly the %d-byte full body (no double-counted prefix)", len(data), len(content))
	}
}

func TestDownloadModelDeclaredSizeCap(t *testing.T) {
	big := strings.Repeat("x", 3<<20) // 3 MiB
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(big)))
		fmt.Fprint(w, big)
	}))
	defer srv.Close()
	withHFServer(t, srv)

	// Manifest declared 5 bytes; cap is 2×5 + 1 MiB slack, so a 3 MiB
	// Content-Length must be refused before any bytes are written.
	dest := filepath.Join(t.TempDir(), "model.gguf")
	_, err := NewDownloaderWithProgress(NewClient(&config.Config{}), nil).DownloadModel(
		"testuser", "repo", "main", "model.gguf", dest, 5)
	if err == nil || !strings.Contains(err.Error(), "exceeds cap") {
		t.Fatalf("err = %v, want size cap rejection", err)
	}
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Error("dest must not exist after a refused download")
	}
}
