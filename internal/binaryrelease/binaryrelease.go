// Package binaryrelease contains platform-agnostic primitives for installing
// third-party release artifacts (tarballs hosted on GitHub releases, fetched
// over HTTPS, pinned by tag, activated via a "current" symlink).
//
// It owns the security-relevant parts: URL host/scheme allow-list, download
// size caps, and the atomic symlink swap. Per-project concerns (which repo
// to fetch, which asset matches the current platform, which tag formats to
// accept, where the version file lives) stay in the caller.
package binaryrelease

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"github.com/nchapman/lleme/internal/fileutil"
)

// Config bundles the policy knobs a caller passes into Download and
// FetchLatestRelease. Keeping policy per-call (rather than in package state)
// means callers can hold their own test hooks.
//
// Zero-value semantics are strict by design: a nil AllowedHosts or
// AllowedSchemes map rejects every URL, which fails closed. MaxBytes is the
// exception — 0 means unlimited. Callers should set it explicitly; the
// package does not enforce a default ceiling, since some use cases
// legitimately need it off.
type Config struct {
	AllowedHosts   map[string]bool // Hosts that may be contacted. nil blocks all.
	AllowedSchemes map[string]bool // Schemes allowed (typically "https"). nil blocks all.
	MaxBytes       int64           // Cap on downloaded bytes. 0 = unlimited.
	UserAgent      string          // User-Agent header; omitted if empty.
}

// DefaultGitHubHosts are the hosts GitHub releases redirect through. github.com
// issues 302 redirects to *.githubusercontent.com for asset downloads.
func DefaultGitHubHosts() map[string]bool {
	return map[string]bool{
		"github.com":                           true,
		"api.github.com":                       true,
		"objects.githubusercontent.com":        true,
		"release-assets.githubusercontent.com": true,
		"releases.githubusercontent.com":       true,
	}
}

// DefaultHTTPSOnly restricts downloads to https. Callers can extend this in
// tests to permit http pointing at an httptest server.
func DefaultHTTPSOnly() map[string]bool {
	return map[string]bool{"https": true}
}

// Release is the minimal shape of a GitHub "latest release" response we use.
type Release struct {
	TagName string  `json:"tag_name"`
	Name    string  `json:"name"`
	Assets  []Asset `json:"assets"`
}

type Asset struct {
	Name               string `json:"name"`
	Size               int64  `json:"size"`
	BrowserDownloadURL string `json:"browser_download_url"`
}

// ValidateURL rejects URLs whose scheme or host is not on the allow-list.
// Exported so callers can apply the same policy to redirect targets.
func ValidateURL(cfg Config, rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL %q: %w", rawURL, err)
	}
	if !cfg.AllowedSchemes[u.Scheme] {
		return fmt.Errorf("URL scheme %q is not allowed", u.Scheme)
	}
	if !cfg.AllowedHosts[u.Hostname()] {
		return fmt.Errorf("URL host %q is not on the allowlist", u.Hostname())
	}
	return nil
}

// FetchLatestRelease GETs the given GitHub releases/latest URL and decodes
// the response into a Release. The apiURL is passed in (not constructed from
// a repo string) so callers can override it in tests.
func FetchLatestRelease(cfg Config, apiURL string) (*Release, error) {
	if err := ValidateURL(cfg, apiURL); err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodGet, apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	if cfg.UserAgent != "" {
		req.Header.Set("User-Agent", cfg.UserAgent)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var release Release
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return nil, err
	}
	return &release, nil
}

// Download streams the given URL to destPath, atomically renaming from a
// .partial once complete. Enforces the configured host/scheme/size limits
// and validates redirects against the same policy.
func Download(ctx context.Context, cfg Config, downloadURL, destPath string, progress func(int64, int64)) error {
	if err := ValidateURL(cfg, downloadURL); err != nil {
		return err
	}

	resp, err := fetch(ctx, cfg, downloadURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if cfg.MaxBytes > 0 && resp.ContentLength > cfg.MaxBytes {
		return fmt.Errorf("download %s: content-length %d exceeds max %d", downloadURL, resp.ContentLength, cfg.MaxBytes)
	}

	tmpPath := destPath + ".partial"
	written, err := streamToFile(resp.Body, cfg.MaxBytes, resp.ContentLength, tmpPath, progress)
	if err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("write to %s: %w", tmpPath, err)
	}
	if cfg.MaxBytes > 0 && written > cfg.MaxBytes {
		os.Remove(tmpPath)
		return fmt.Errorf("download %s exceeded max size of %d bytes", downloadURL, cfg.MaxBytes)
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename %s to %s: %w", tmpPath, destPath, err)
	}
	return nil
}

func fetch(ctx context.Context, cfg Config, downloadURL string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, downloadURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if cfg.UserAgent != "" {
		req.Header.Set("User-Agent", cfg.UserAgent)
	}

	client := &http.Client{
		Transport: &http.Transport{ResponseHeaderTimeout: 30 * time.Second},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if err := ValidateURL(cfg, req.URL.String()); err != nil {
				return fmt.Errorf("redirect blocked: %w", err)
			}
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download %s: %w", downloadURL, err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("download %s: HTTP %d", downloadURL, resp.StatusCode)
	}
	return resp, nil
}

func streamToFile(src io.Reader, maxBytes, contentLength int64, tmpPath string, progress func(int64, int64)) (int64, error) {
	out, err := os.Create(tmpPath)
	if err != nil {
		return 0, fmt.Errorf("create %s: %w", tmpPath, err)
	}
	defer out.Close()

	// Cap the body stream so a lying Content-Length can't fill the disk.
	if maxBytes > 0 {
		src = io.LimitReader(src, maxBytes+1)
	}
	return fileutil.StreamBody(src, out, 0, contentLength, progress)
}

// SwapCurrentSymlink atomically points <dir>/<linkName> at target by staging
// a tmp symlink and renaming it into place. Avoids the ENOENT window a
// remove+symlink pair would expose to concurrent exec.Command calls.
func SwapCurrentSymlink(dir, linkName, target string) error {
	currentLink := filepath.Join(dir, linkName)
	tmpLink := filepath.Join(dir, "."+linkName+".tmp")

	// A stale tmp from a prior failed run would make Symlink fail with EEXIST.
	_ = os.Remove(tmpLink)

	if err := os.Symlink(target, tmpLink); err != nil {
		return fmt.Errorf("stage %s symlink: %w", linkName, err)
	}
	if err := os.Rename(tmpLink, currentLink); err != nil {
		_ = os.Remove(tmpLink)
		return fmt.Errorf("activate %s symlink: %w", linkName, err)
	}
	return nil
}
