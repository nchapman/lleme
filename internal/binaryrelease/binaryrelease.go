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
	"archive/tar"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/fileutil"
)

// maxAPIBodyBytes caps the size of a GitHub API response we'll buffer. Real
// release JSON is tens of KiB; 1 MiB is generous headroom with a hard ceiling.
const maxAPIBodyBytes = 1 << 20

// tarEntryMode masks permission bits from archive entries. We allow the
// standard user/group/other triplets but strip setuid/setgid/sticky so a
// compromised archive can't install a setuid binary.
const tarEntryMode os.FileMode = 0755

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
// a repo string) so callers can override it in tests. Response body is
// bounded by maxAPIBodyBytes and redirects are revalidated against cfg via
// the shared client, so the allow-list holds end-to-end.
func FetchLatestRelease(ctx context.Context, cfg Config, apiURL string) (*Release, error) {
	if err := ValidateURL(cfg, apiURL); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	if cfg.UserAgent != "" {
		req.Header.Set("User-Agent", cfg.UserAgent)
	}

	resp, err := newClient(cfg).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body := io.LimitReader(resp.Body, maxAPIBodyBytes)
	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(errBody))
	}

	var release Release
	if err := json.NewDecoder(body).Decode(&release); err != nil {
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

// newClient builds the HTTP client used for every outbound request. The
// CheckRedirect hook re-runs ValidateURL on each hop so the allow-list holds
// across redirects; transport-level ResponseHeaderTimeout bounds connection
// setup; the client Timeout bounds short metadata requests (callers pass
// context for long downloads).
func newClient(cfg Config) *http.Client {
	return &http.Client{
		Timeout:   30 * time.Second,
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
}

func fetch(ctx context.Context, cfg Config, downloadURL string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, downloadURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if cfg.UserAgent != "" {
		req.Header.Set("User-Agent", cfg.UserAgent)
	}

	// Downloads stream large bodies, so suppress the client-level deadline
	// that newClient sets for metadata requests. CheckRedirect still applies.
	client := newClient(cfg)
	client.Timeout = 0

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

// streamToFile copies src to tmpPath and returns the number of bytes it
// wrote this call (independent of any StreamBody running-tally semantics).
// The size cap is enforced via LimitReader(maxBytes+1); callers check the
// returned count against their own max to detect a lying Content-Length.
func streamToFile(src io.Reader, maxBytes, contentLength int64, tmpPath string, progress func(int64, int64)) (int64, error) {
	out, err := os.Create(tmpPath)
	if err != nil {
		return 0, fmt.Errorf("create %s: %w", tmpPath, err)
	}
	defer out.Close()

	// Cap the body stream so a lying Content-Length can't fill the disk.
	// The maxBytes+1 trick lets the caller detect the cap was hit (not a
	// natural EOF). maxBytes < MaxInt64 is asserted by the caller — treat
	// a nonsensical value as "no cap" rather than overflow to negative.
	if maxBytes > 0 && maxBytes < (1<<62) {
		src = io.LimitReader(src, maxBytes+1)
	}
	start := int64(0)
	end, err := fileutil.StreamBody(src, out, start, contentLength, progress)
	return end - start, err
}

// ExtractTarGz unpacks a .tar.gz archive into destDir, rejecting any entry
// whose path would escape destDir (absolute paths, .. traversal, symlinks
// pointing outside). Only regular files, directories, and same-tree
// symlinks are extracted; hard links, devices, and setuid bits are dropped
// so a malicious archive can't mutate files outside the extract tree.
func ExtractTarGz(archivePath, destDir string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open archive: %w", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip reader: %w", err)
	}
	defer gz.Close()

	absDest, err := filepath.Abs(destDir)
	if err != nil {
		return fmt.Errorf("resolve destination: %w", err)
	}

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read archive: %w", err)
		}
		if err := extractEntry(tr, hdr, absDest); err != nil {
			return err
		}
	}
}

// SafeJoin returns absDest/rel, guaranteed to live under absDest. Rejects
// absolute entry names, .. traversal, and any path that lexically escapes
// absDest. Used by ExtractTarGz and by callers (e.g. hf.PullMLXModel) that
// need to join an attacker-influenced relative path onto a local root.
func SafeJoin(absDest, name string) (string, error) {
	clean := filepath.Clean(name)
	if filepath.IsAbs(clean) {
		return "", fmt.Errorf("path %q is absolute", name)
	}
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes destination", name)
	}
	joined := filepath.Join(absDest, clean)
	if joined != absDest && !strings.HasPrefix(joined, absDest+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes destination", name)
	}
	return joined, nil
}

func extractEntry(tr *tar.Reader, hdr *tar.Header, absDest string) error {
	target, err := SafeJoin(absDest, hdr.Name)
	if err != nil {
		return err
	}
	mode := os.FileMode(hdr.Mode) & tarEntryMode

	switch hdr.Typeflag {
	case tar.TypeDir:
		return os.MkdirAll(target, mode|0755)
	case tar.TypeReg:
		return writeTarFile(tr, hdr, target, mode)
	case tar.TypeSymlink:
		return writeTarSymlink(hdr, target, absDest)
	default:
		// Silently skip hard links, devices, fifos, etc. They aren't needed
		// for the artifacts we install and extracting them would broaden the
		// attack surface.
		return nil
	}
}

func writeTarFile(tr *tar.Reader, hdr *tar.Header, target string, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		return fmt.Errorf("mkdir parent of %s: %w", hdr.Name, err)
	}
	out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode|0644)
	if err != nil {
		return fmt.Errorf("create %s: %w", hdr.Name, err)
	}
	// Cap the per-entry copy so an inflated gzip bomb can't fill disk.
	// hdr.Size<=0 means "unknown/empty"; clamp to a generous 8 GiB ceiling.
	// Use LimitReader(limit+1)+io.Copy so a source that exactly matches the
	// cap — and has leftover bytes that would bleed into the next tar entry
	// — is rejected. io.CopyN can't distinguish "source ended at limit" from
	// "source has more than limit," which would corrupt subsequent files.
	limit := hdr.Size
	if limit <= 0 || limit > 1<<33 {
		limit = 1 << 33
	}
	written, err := io.Copy(out, io.LimitReader(tr, limit+1))
	if err != nil {
		out.Close()
		return fmt.Errorf("write %s: %w", hdr.Name, err)
	}
	if written > limit {
		out.Close()
		return fmt.Errorf("archive entry %s exceeded size limit of %d bytes", hdr.Name, limit)
	}
	return out.Close()
}

func writeTarSymlink(hdr *tar.Header, target, absDest string) error {
	if filepath.IsAbs(hdr.Linkname) {
		return fmt.Errorf("archive symlink %q has absolute target", hdr.Name)
	}
	resolved := filepath.Clean(filepath.Join(filepath.Dir(target), hdr.Linkname))
	if resolved != absDest && !strings.HasPrefix(resolved, absDest+string(filepath.Separator)) {
		return fmt.Errorf("archive symlink %q escapes destination", hdr.Name)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		return fmt.Errorf("mkdir parent of %s: %w", hdr.Name, err)
	}
	// Remove any pre-existing entry so Symlink doesn't fail with EEXIST
	// on a re-extraction.
	_ = os.Remove(target)
	return os.Symlink(hdr.Linkname, target)
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
