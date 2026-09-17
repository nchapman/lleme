package hf

import (
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nchapman/lleme/internal/config"
)

func TestGatedStatusUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{
			name:  "bool false",
			input: `{"gated": false}`,
			want:  false,
		},
		{
			name:  "bool true",
			input: `{"gated": true}`,
			want:  true,
		},
		{
			name:  "string manual",
			input: `{"gated": "manual"}`,
			want:  true,
		},
		{
			name:  "string auto",
			input: `{"gated": "auto"}`,
			want:  true,
		},
		{
			name:  "null value treated as not gated",
			input: `{"gated": null}`,
			want:  false,
		},
		{
			name:  "numeric value treated as gated",
			input: `{"gated": 1}`,
			want:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var result struct {
				Gated GatedStatus `json:"gated"`
			}
			if err := json.Unmarshal([]byte(tt.input), &result); err != nil {
				t.Fatalf("Unmarshal() error = %v", err)
			}
			if bool(result.Gated) != tt.want {
				t.Errorf("GatedStatus = %v, want %v", result.Gated, tt.want)
			}
		})
	}
}

func TestHFHostAllowed(t *testing.T) {
	tests := []struct {
		name string
		host string
		want bool
	}{
		{name: "apex huggingface.co", host: "huggingface.co", want: true},
		{name: "apex hf.co", host: "hf.co", want: true},
		{name: "legacy LFS CDN", host: "cdn-lfs.huggingface.co", want: true},
		{name: "legacy LFS CDN short domain", host: "cdn-lfs.hf.co", want: true},
		{name: "xet cas bridge", host: "cas-bridge.xethub.hf.co", want: true},
		{name: "xet cas server", host: "cas-server.xethub.hf.co", want: true},
		{name: "regional xet CDN us", host: "us.aws.cdn.hf.co", want: true},
		{name: "regional xet CDN eu", host: "eu.aws.cdn.hf.co", want: true},
		{name: "xet transfer host", host: "transfer.xethub.hf.co", want: true},
		{name: "uppercase host", host: "US.AWS.CDN.HF.CO", want: true},
		{name: "unrelated host", host: "evil.com", want: false},
		{name: "suffix spoof", host: "hf.co.evil.com", want: false},
		{name: "dot-boundary spoof", host: "evilhf.co", want: false},
		{name: "empty host", host: "", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hfHostAllowed(tt.host); got != tt.want {
				t.Errorf("hfHostAllowed(%q) = %v, want %v", tt.host, got, tt.want)
			}
		})
	}
}

func TestDownloadClientRedirectPolicy(t *testing.T) {
	c := NewClient(&config.Config{})

	redirectTarget := func(raw string) *http.Request {
		u, err := url.Parse(raw)
		if err != nil {
			t.Fatalf("parse %q: %v", raw, err)
		}
		return &http.Request{URL: u}
	}

	t.Run("regional xet CDN allowed", func(t *testing.T) {
		if err := c.downloadClient.CheckRedirect(redirectTarget("https://us.aws.cdn.hf.co/xet-bridge-us/abc"), nil); err != nil {
			t.Errorf("CheckRedirect() = %v, want nil", err)
		}
	})

	t.Run("off-domain host blocked", func(t *testing.T) {
		err := c.downloadClient.CheckRedirect(redirectTarget("https://evil.com/file.gguf"), nil)
		if err == nil || !strings.Contains(err.Error(), "redirect blocked") {
			t.Errorf("CheckRedirect() = %v, want redirect blocked error", err)
		}
	})

	t.Run("plain http blocked", func(t *testing.T) {
		err := c.downloadClient.CheckRedirect(redirectTarget("http://cdn-lfs.hf.co/file.gguf"), nil)
		if err == nil || !strings.Contains(err.Error(), "not https") {
			t.Errorf("CheckRedirect() = %v, want scheme rejection", err)
		}
	})

	t.Run("redirect loop capped", func(t *testing.T) {
		via := make([]*http.Request, 10)
		err := c.downloadClient.CheckRedirect(redirectTarget("https://cdn-lfs.hf.co/file.gguf"), via)
		if err == nil || !strings.Contains(err.Error(), "too many redirects") {
			t.Errorf("CheckRedirect() = %v, want too many redirects", err)
		}
	})
}

func TestHasToken(t *testing.T) {
	// Save original env and restore after test
	origToken := os.Getenv("HF_TOKEN")
	defer os.Setenv("HF_TOKEN", origToken)

	t.Run("from env var", func(t *testing.T) {
		os.Setenv("HF_TOKEN", "test-token")
		defer os.Unsetenv("HF_TOKEN")

		cfg := &config.Config{}
		if !HasToken(cfg) {
			t.Error("HasToken() = false, want true when HF_TOKEN is set")
		}
	})

	t.Run("from config", func(t *testing.T) {
		os.Unsetenv("HF_TOKEN")

		cfg := &config.Config{
			HuggingFace: config.HuggingFace{
				Token: "config-token",
			},
		}
		if !HasToken(cfg) {
			t.Error("HasToken() = false, want true when config token is set")
		}
	})

	t.Run("from cache file", func(t *testing.T) {
		os.Unsetenv("HF_TOKEN")

		// Create temp home dir with token file
		tmpDir := t.TempDir()
		oldHome := os.Getenv("HOME")
		defer os.Setenv("HOME", oldHome)
		os.Setenv("HOME", tmpDir)

		tokenDir := filepath.Join(tmpDir, ".cache", "huggingface")
		if err := os.MkdirAll(tokenDir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(tokenDir, "token"), []byte("file-token"), 0600); err != nil {
			t.Fatal(err)
		}

		cfg := &config.Config{}
		if !HasToken(cfg) {
			t.Error("HasToken() = false, want true when cache file token exists")
		}
	})

	t.Run("from cache file with trailing newline", func(t *testing.T) {
		os.Unsetenv("HF_TOKEN")

		// Create temp home dir with token file containing trailing newline
		tmpDir := t.TempDir()
		oldHome := os.Getenv("HOME")
		defer os.Setenv("HOME", oldHome)
		os.Setenv("HOME", tmpDir)

		tokenDir := filepath.Join(tmpDir, ".cache", "huggingface")
		if err := os.MkdirAll(tokenDir, 0755); err != nil {
			t.Fatal(err)
		}
		// Token with trailing newline (common when created by editors or echo)
		if err := os.WriteFile(filepath.Join(tokenDir, "token"), []byte("file-token\n"), 0600); err != nil {
			t.Fatal(err)
		}

		cfg := &config.Config{}
		if !HasToken(cfg) {
			t.Error("HasToken() = false, want true when cache file token exists with newline")
		}
	})

	t.Run("no token", func(t *testing.T) {
		os.Unsetenv("HF_TOKEN")

		// Use temp home dir with no token file
		tmpDir := t.TempDir()
		oldHome := os.Getenv("HOME")
		defer os.Setenv("HOME", oldHome)
		os.Setenv("HOME", tmpDir)

		cfg := &config.Config{}
		if HasToken(cfg) {
			t.Error("HasToken() = true, want false when no token is available")
		}
	})
}
