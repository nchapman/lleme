package selfupdate

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nchapman/lleme/internal/version"
)

func withAPIBase(t *testing.T, url string) {
	t.Helper()
	old := apiBase
	apiBase = url
	t.Cleanup(func() { apiBase = old })
}

func TestGetLatestVersion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/repos/nchapman/lleme/releases/latest" {
			http.NotFound(w, r)
			return
		}
		if ua := r.Header.Get("User-Agent"); ua != version.UserAgent() {
			t.Errorf("User-Agent = %q, want %q", ua, version.UserAgent())
		}
		fmt.Fprint(w, `{"tag_name":"v0.12.1","name":"v0.12.1"}`)
	}))
	defer srv.Close()
	withAPIBase(t, srv.URL+"/repos/nchapman/lleme")

	v, err := GetLatestVersion()
	if err != nil {
		t.Fatal(err)
	}
	if v != "0.12.1" {
		t.Errorf("GetLatestVersion() = %q, want 0.12.1 (v prefix stripped)", v)
	}
}

func TestGetLatestVersionHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusForbidden)
	}))
	defer srv.Close()
	withAPIBase(t, srv.URL)

	if _, err := GetLatestVersion(); err == nil {
		t.Error("expected error for HTTP 403")
	}
}

func TestGetLatestVersionInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "not json")
	}))
	defer srv.Close()
	withAPIBase(t, srv.URL)

	if _, err := GetLatestVersion(); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestUpdateUnknownMethod(t *testing.T) {
	if err := Update(InstallMethod("bogus")); err == nil {
		t.Error("Update with unknown method must error")
	}
}

func TestManualUpdateInstructions(t *testing.T) {
	s := ManualUpdateInstructions()
	for _, want := range []string{"Homebrew", "go install", "releases"} {
		if !strings.Contains(s, want) {
			t.Errorf("instructions missing %q: %s", want, s)
		}
	}
}
