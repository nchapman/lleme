package hf

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/fileutil"
	"github.com/nchapman/lleme/internal/version"
	"gopkg.in/yaml.v3"
)

type ProgressCallback func(downloaded, total int64, speed float64, eta time.Duration)

type DownloadProgress struct {
	Downloaded int64
	Total      int64
	Speed      float64
	ETA        time.Duration
}

type Downloader struct {
	client     *Client
	progress   ProgressCallback
	startTime  time.Time
	lastUpdate time.Time
	lastBytes  int64
}

func NewDownloaderWithProgress(client *Client, progress ProgressCallback) *Downloader {
	return &Downloader{
		client:   client,
		progress: progress,
	}
}

func (d *Downloader) DownloadModel(user, repo, branch, filename string, destPath string) (*DownloadProgress, error) {
	url := fmt.Sprintf("%s/%s/%s/resolve/%s/%s", baseURL, user, repo, branch, filename)

	partialPath := destPath + ".partial"
	fileSize := int64(0)

	if info, err := os.Stat(partialPath); err == nil {
		fileSize = info.Size()
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", version.UserAgent())
	if d.client.token != "" {
		req.Header.Set("Authorization", "Bearer "+d.client.token)
	}

	if fileSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", fileSize))
	}

	resp, err := d.client.downloadClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	totalSize := fileSize + resp.ContentLength

	flags := os.O_CREATE | os.O_WRONLY
	if resp.StatusCode == http.StatusOK {
		flags |= os.O_TRUNC
	} else {
		flags |= os.O_APPEND
	}

	file, err := os.OpenFile(partialPath, flags, 0644)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	d.startTime = time.Now()
	d.lastUpdate = d.startTime
	d.lastBytes = fileSize

	var progressFn func(int64, int64)
	if d.progress != nil {
		progressFn = func(written, total int64) {
			p := d.calculateProgress(written, total)
			d.progress(p.Downloaded, p.Total, p.Speed, p.ETA)
		}
	}

	written, err := fileutil.StreamBody(resp.Body, file, fileSize, totalSize, progressFn)
	if err != nil {
		return nil, err
	}

	file.Close()

	if err := os.Rename(partialPath, destPath); err != nil {
		return nil, err
	}

	progress := d.calculateProgress(written, totalSize)
	return progress, nil
}

func (d *Downloader) calculateProgress(downloaded, total int64) *DownloadProgress {
	now := time.Now()

	var speed float64
	var eta time.Duration

	if now.Sub(d.lastUpdate) > time.Second {
		deltaBytes := downloaded - d.lastBytes
		deltaTime := now.Sub(d.lastUpdate).Seconds()
		speed = float64(deltaBytes) / deltaTime

		remaining := total - downloaded
		if speed > 0 {
			eta = time.Duration(float64(remaining)/speed) * time.Second
		}

		d.lastUpdate = now
		d.lastBytes = downloaded
	}

	return &DownloadProgress{
		Downloaded: downloaded,
		Total:      total,
		Speed:      speed,
		ETA:        eta,
	}
}

// CalculateSHA256WithProgress computes sha256 hash with optional progress callback.
// The callback receives bytes processed and total size.
func CalculateSHA256WithProgress(filePath string, progress func(processed, total int64)) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return "", err
	}
	totalSize := info.Size()

	hash := sha256.New()
	buf := make([]byte, 32*1024)
	processed := int64(0)

	for {
		n, err := file.Read(buf)
		if n > 0 {
			hash.Write(buf[:n])
			processed += int64(n)
			if progress != nil {
				progress(processed, totalSize)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}

	return hex.EncodeToString(hash.Sum(nil)), nil
}

func GetModelPath(user, repo string) string {
	return filepath.Join(config.ModelsPath(), user, repo)
}

func GetModelFilePath(user, repo, quant string) string {
	modelDir := GetModelPath(user, repo)
	return filepath.Join(modelDir, quant+".gguf")
}

// GetSplitModelDir returns the directory path for split model files.
func GetSplitModelDir(user, repo, quant string) string {
	modelDir := GetModelPath(user, repo)
	return filepath.Join(modelDir, quant)
}

// FindModelFile returns the actual model file path, checking both single file
// and split directory cases. Returns empty string if not found.
func FindModelFile(user, repo, quant string) string {
	// Check for single file first
	singlePath := GetModelFilePath(user, repo, quant)
	if _, err := os.Stat(singlePath); err == nil {
		return singlePath
	}

	// Check for split directory
	splitDir := GetSplitModelDir(user, repo, quant)
	if info, err := os.Stat(splitDir); err == nil && info.IsDir() {
		return FindFirstSplitFile(splitDir)
	}

	return ""
}

// GetMMProjFilePath returns the path where the mmproj file is stored for a model.
// The mmproj file is stored per-quantization since different quants may have different mmproj files.
func GetMMProjFilePath(user, repo, quant string) string {
	modelDir := GetModelPath(user, repo)
	return filepath.Join(modelDir, quant+"-mmproj.gguf")
}

// IsMMProjFileName reports whether a GGUF filename is a vision-encoder companion
// written by GetMMProjFilePath (e.g. "Q4_K_M-mmproj.gguf"). These files are
// loaded via llama-server's --mmproj flag, not served as standalone models, so
// walkers that enumerate downloaded models should skip them.
func IsMMProjFileName(name string) bool {
	return strings.HasSuffix(strings.ToLower(name), "-mmproj.gguf")
}

// GetManifestFilePath returns the path where the manifest is stored for a model.
func GetManifestFilePath(user, repo, quant string) string {
	modelDir := GetModelPath(user, repo)
	return filepath.Join(modelDir, quant+"-manifest.json")
}

// ModelMetadata stores metadata for a downloaded model repository.
type ModelMetadata struct {
	Quants map[string]QuantMetadata `yaml:"quants"`
}

// QuantMetadata stores metadata for a specific quantization.
type QuantMetadata struct {
	LastUsed     time.Time `yaml:"last_used,omitempty"`
	DownloadedAt time.Time `yaml:"downloaded_at,omitempty"`
	// Backend identifies the runtime that serves this quant: "gguf" or "mlx".
	// Empty in legacy metadata files — treat as "gguf" (the only kind lleme
	// could pull before MLX support landed).
	Backend string `yaml:"backend,omitempty"`
}

// GetMetadataPath returns the path to the metadata.yaml file for a model repo.
func GetMetadataPath(user, repo string) string {
	return filepath.Join(GetModelPath(user, repo), "metadata.yaml")
}

// LoadMetadata loads the metadata for a model repo, or returns empty metadata if not found.
func LoadMetadata(user, repo string) (*ModelMetadata, error) {
	path := GetMetadataPath(user, repo)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &ModelMetadata{Quants: make(map[string]QuantMetadata)}, nil
		}
		return nil, err
	}

	var meta ModelMetadata
	if err := yaml.Unmarshal(data, &meta); err != nil {
		return nil, err
	}
	if meta.Quants == nil {
		meta.Quants = make(map[string]QuantMetadata)
	}
	return &meta, nil
}

// SaveMetadata saves the metadata for a model repo.
func SaveMetadata(user, repo string, meta *ModelMetadata) error {
	path := GetMetadataPath(user, repo)
	data, err := yaml.Marshal(meta)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// SetBackendKind records the runtime kind ("gguf" or "mlx") for a quant in
// metadata.yaml. Called at pull time so the proxy can pick the right backend
// without re-detecting the model format.
func SetBackendKind(user, repo, quant, kind string) error {
	meta, err := LoadMetadata(user, repo)
	if err != nil {
		return err
	}

	q := meta.Quants[quant]
	q.Backend = kind
	if q.DownloadedAt.IsZero() {
		q.DownloadedAt = time.Now()
	}
	meta.Quants[quant] = q

	return SaveMetadata(user, repo, meta)
}

// Known backend kinds as persisted in metadata.yaml. These identify the
// on-disk model format; the proxy maps them to a concrete Runtime.
const (
	BackendGGUF = "gguf"
	BackendMLX  = "mlx"
)

// BackendKindForModelName parses a "user/repo:quant" reference and resolves
// its backend kind via metadata.yaml. Falls back to BackendGGUF for any
// parse or read error — callers treat the result as a hint for layering
// backend-specific options, not an authoritative runtime selection.
func BackendKindForModelName(fullName string) string {
	colon := strings.LastIndex(fullName, ":")
	if colon < 0 {
		return BackendGGUF
	}
	repoPart, quant := fullName[:colon], fullName[colon+1:]
	slash := strings.Index(repoPart, "/")
	if slash < 0 {
		return BackendGGUF
	}
	kind, err := GetBackendKind(repoPart[:slash], repoPart[slash+1:], quant)
	if err != nil {
		return BackendGGUF
	}
	return kind
}

// GetBackendKind returns the recorded backend kind for a quant. A missing
// metadata file or an empty `backend:` field resolves to "gguf" (the only
// format lleme supported before MLX). I/O and parse errors are propagated
// so callers can surface them instead of silently falling back. Any
// unrecognized backend string in the file is an error — metadata.yaml is
// local, but corrupt / tampered content shouldn't silently dispatch to a
// default runtime.
func GetBackendKind(user, repo, quant string) (string, error) {
	meta, err := LoadMetadata(user, repo)
	if err != nil {
		return "", err
	}
	q, ok := meta.Quants[quant]
	if !ok || q.Backend == "" {
		return BackendGGUF, nil
	}
	return parseBackendKind(q.Backend)
}

// parseBackendKind accepts only the known kinds. Keeps the on-disk
// vocabulary narrow so downstream switches don't have to re-validate.
func parseBackendKind(s string) (string, error) {
	switch s {
	case BackendGGUF, BackendMLX:
		return s, nil
	default:
		return "", fmt.Errorf("unrecognized backend kind %q", s)
	}
}

// TouchLastUsed updates the last used timestamp for a model.
func TouchLastUsed(user, repo, quant string) error {
	meta, err := LoadMetadata(user, repo)
	if err != nil {
		return err
	}

	q := meta.Quants[quant]
	q.LastUsed = time.Now()
	meta.Quants[quant] = q

	return SaveMetadata(user, repo, meta)
}

// GetLastUsed returns the last used time for a model, or zero time if not tracked.
func GetLastUsed(user, repo, quant string) time.Time {
	meta, err := LoadMetadata(user, repo)
	if err != nil {
		return time.Time{}
	}
	return meta.Quants[quant].LastUsed
}

// FindFirstSplitFile finds the first split file (-00001-of-NNNNN) in a directory.
// Returns empty string if no split file is found.
func FindFirstSplitFile(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasSuffix(name, ".gguf") && strings.Contains(name, "-00001-of-") {
			return filepath.Join(dir, name)
		}
	}
	return ""
}

// FindMMProjFile checks if an mmproj file exists for the given model and returns its path.
// Returns empty string if no mmproj file exists.
func FindMMProjFile(user, repo, quant string) string {
	path := GetMMProjFilePath(user, repo, quant)
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return ""
}

func CleanupPartialFiles() (int, error) {
	binDir := config.BinPath()
	modelsDir := config.ModelsPath()

	dirs := []string{binDir, modelsDir}
	count := 0

	for _, dir := range dirs {
		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if !info.IsDir() && strings.HasSuffix(path, ".partial") {
				if os.Remove(path) == nil {
					count++
				}
			}
			return nil
		})
		if err != nil {
			return count, err
		}
	}

	return count, nil
}
