package hf

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/version"
)

const (
	baseURL    = "https://huggingface.co"
	apiBase    = "https://huggingface.co/api"
	maxRetries = 3
	retryDelay = 1 * time.Second
)

type Client struct {
	httpClient     *http.Client
	downloadClient *http.Client
	token          string
}

type ModelInfo struct {
	ModelId      string      `json:"modelId"`
	Author       string      `json:"author"`
	CreatedAt    time.Time   `json:"createdAt"`
	LastModified time.Time   `json:"lastModified"`
	Private      bool        `json:"private"`
	Gated        GatedStatus `json:"gated"`
	Downloads    int64       `json:"downloads"`
	Likes        int64       `json:"likes"`
	Siblings     []Sibling   `json:"siblings"`
	Tags         []string    `json:"tags"`
	CardData     CardData    `json:"cardData"`
}

type Sibling struct {
	RFileName string `json:"rfilename"`
	Size      int64  `json:"size"`
}

type CardData struct {
	License   string   `json:"license"`
	BaseModel any      `json:"base_model"` // Can be string or []string
	Tags      []string `json:"tags"`
}

// FileTreeLFS contains LFS metadata for a file in the tree API response.
type FileTreeLFS struct {
	OID         string `json:"oid"` // SHA256 hash
	Size        int64  `json:"size"`
	PointerSize int    `json:"pointerSize"`
}

// FileTree represents a file or directory in a HuggingFace repository.
type FileTree struct {
	Path string      `json:"path"`
	Type string      `json:"type"`
	Size int64       `json:"size"`
	LFS  FileTreeLFS `json:"lfs,omitempty"` // Present for LFS files
}

type SearchResult struct {
	ID           string      `json:"id"` // Full model ID: "author/repo"
	Author       string      `json:"author"`
	LastModified time.Time   `json:"lastModified"`
	Private      bool        `json:"private"`
	Gated        GatedStatus `json:"gated"`
	Downloads    int64       `json:"downloads"`
	Likes        int64       `json:"likes"`
}

// GatedStatus handles the HuggingFace "gated" field which can be bool or string.
type GatedStatus bool

func (g *GatedStatus) UnmarshalJSON(data []byte) error {
	// Try bool first
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		*g = GatedStatus(b)
		return nil
	}
	// Must be a string like "manual" or "auto" - treat as gated
	*g = true
	return nil
}

// searchResponse wraps the models-json API response.
type searchResponse struct {
	Models        []SearchResult `json:"models"`
	NumTotalItems int            `json:"numTotalItems"`
}

// ManifestLFS contains Git LFS metadata including the sha256 hash for verification.
type ManifestLFS struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// ManifestFile represents a file entry in the HuggingFace manifest API response.
type ManifestFile struct {
	RFilename string       `json:"rfilename"`
	Size      int64        `json:"size"`
	LFS       *ManifestLFS `json:"lfs"`
}

// Manifest represents the HuggingFace manifest API response.
// This API returns the recommended GGUF file and optional mmproj file for vision models.
type Manifest struct {
	GGUFFile   *ManifestFile   `json:"ggufFile"`
	MMProjFile *ManifestFile   `json:"mmprojFile"`
	SplitFiles []*ManifestFile `json:"splitFiles,omitempty"` // Additional split files (local augmentation)
}

func NewClient(cfg *config.Config) *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		downloadClient: &http.Client{
			Transport: &http.Transport{
				ResponseHeaderTimeout: 30 * time.Second,
			},
		},
		token: getToken(cfg),
	}
}

func getToken(cfg *config.Config) string {
	if token := os.Getenv("HF_TOKEN"); token != "" {
		return token
	}

	tokenPath := filepath.Join(config.UserHomeDir(), ".cache", "huggingface", "token")
	if data, err := os.ReadFile(tokenPath); err == nil {
		return strings.TrimSpace(string(data))
	}

	return cfg.HuggingFace.Token
}

// HasToken returns true if a HuggingFace token is available from any source.
func HasToken(cfg *config.Config) bool {
	return getToken(cfg) != ""
}

func (c *Client) doRequest(req *http.Request) (*http.Response, error) {
	// Only set User-Agent if not already set (allows callers to override)
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", version.UserAgent())
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	var lastErr error
	for i := 0; i < maxRetries; i++ {
		resp, err := c.httpClient.Do(req)
		if err == nil {
			if resp.StatusCode == http.StatusTooManyRequests {
				resp.Body.Close()
				time.Sleep(retryDelay * time.Duration(i+1))
				continue
			}
			return resp, nil
		}
		lastErr = err
		time.Sleep(retryDelay)
	}

	return nil, fmt.Errorf("request failed after %d retries: %w", maxRetries, lastErr)
}

func (c *Client) GetModel(user, repo string) (*ModelInfo, error) {
	url := fmt.Sprintf("%s/models/%s/%s", apiBase, user, repo)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var model ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&model); err != nil {
		return nil, err
	}

	return &model, nil
}

func (c *Client) ListFiles(user, repo, branch string) ([]FileTree, error) {
	return c.ListFilesInPath(user, repo, branch, "")
}

// ListFilesInPath lists files in a specific path within a repository.
func (c *Client) ListFilesInPath(user, repo, branch, path string) ([]FileTree, error) {
	var urlStr string
	if path == "" {
		urlStr = fmt.Sprintf("%s/models/%s/%s/tree/%s", apiBase, user, repo, branch)
	} else {
		urlStr = fmt.Sprintf("%s/models/%s/%s/tree/%s/%s", apiBase, user, repo, branch, path)
	}
	req, err := http.NewRequest("GET", urlStr, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var files []FileTree
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, err
	}

	return files, nil
}

// GetFileSize returns the size of a file in a repository using a HEAD request.
func (c *Client) GetFileSize(user, repo, branch, filename string) (int64, error) {
	url := fmt.Sprintf("%s/%s/%s/resolve/%s/%s", baseURL, user, repo, branch, filename)
	req, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		return 0, err
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return 0, err
	}
	resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	return resp.ContentLength, nil
}

func (c *Client) SearchModels(query string, limit int) ([]SearchResult, error) {
	// Use models-json endpoint with apps=llama.cpp filter for llama.cpp compatible models
	searchURL := fmt.Sprintf("%s/models-json?apps=llama.cpp&sort=trending", baseURL)
	if query != "" {
		searchURL += "&search=" + url.QueryEscape(query)
	}
	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var response searchResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	// Limit results
	results := response.Models
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

func (c *Client) DownloadFile(user, repo, branch, filename string, progress func(int64, int64)) (string, error) {
	url := fmt.Sprintf("%s/%s/%s/resolve/%s/%s", baseURL, user, repo, branch, filename)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	tmpPath := filepath.Join(config.BinPath(), filename+".partial")
	out, err := os.Create(tmpPath)
	if err != nil {
		return "", err
	}
	defer out.Close()

	size := resp.ContentLength
	written := int64(0)
	buf := make([]byte, 32*1024)

	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			_, werr := out.Write(buf[:n])
			if werr != nil {
				return "", werr
			}
			written += int64(n)
			if progress != nil {
				progress(written, size)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}

	finalPath := filepath.Join(config.BinPath(), filename)
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return "", err
	}

	return finalPath, nil
}

// GetManifest fetches the HuggingFace manifest for a model, which includes
// the recommended GGUF file and optional mmproj file for vision models.
// The tag parameter is typically a quantization level like "Q4_K_M".
// Returns both the parsed manifest and the raw JSON bytes for saving to disk.
func (c *Client) GetManifest(user, repo, tag string) (*Manifest, []byte, error) {
	url := fmt.Sprintf("%s/v2/%s/%s/manifests/%s", baseURL, user, repo, tag)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, nil, err
	}

	// Must include "llama-cpp" in user-agent to get ggufFile/mmprojFile fields
	req.Header.Set("User-Agent", version.UserAgent()+" (llama-cpp compatible)")
	req.Header.Set("Accept", "application/json")

	resp, err := c.doRequest(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	rawJSON, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}

	var manifest Manifest
	if err := json.Unmarshal(rawJSON, &manifest); err != nil {
		return nil, nil, err
	}

	return &manifest, rawJSON, nil
}

// FetchFolderQuantSizes fills in Size for folder-style quantizations by
// summing the sizes of GGUF files in each directory.
func (c *Client) FetchFolderQuantSizes(user, repo, branch string, quants []Quantization) {
	for i := range quants {
		if quants[i].Size == 0 {
			dirFiles, err := c.ListFilesInPath(user, repo, branch, quants[i].Tag)
			if err == nil {
				var total int64
				for _, f := range dirFiles {
					if strings.HasSuffix(f.Path, ".gguf") {
						total += f.Size
					}
				}
				quants[i].Size = total
			}
		}
	}
}
