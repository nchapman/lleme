package hf

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// PullResult contains the result of a model pull operation.
type PullResult struct {
	ModelPath  string
	MMProjPath string // Empty if not a vision model
	IsVision   bool
	TotalSize  int64
	GGUFSize   int64
	MMProjSize int64
}

// PullProgress is called during download and verification phases.
type PullProgress struct {
	Phase   string // "download" or "verify"
	Current int64
	Total   int64
}

// ManifestInfo contains size information about a model from its manifest.
type ManifestInfo struct {
	GGUFSize   int64
	MMProjSize int64
	TotalSize  int64
	IsVision   bool
}

// GetManifestInfo fetches the manifest and returns size information for display.
// Returns the manifest data so it can be passed to PullModel to avoid re-fetching.
func GetManifestInfo(client *Client, user, repo string, quant Quantization) (*ManifestInfo, *Manifest, []byte, error) {
	manifest, manifestJSON, err := client.GetManifest(user, repo, quant.Tag)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	if manifest.GGUFFile == nil {
		return nil, nil, nil, fmt.Errorf("manifest does not contain a GGUF file")
	}

	info := &ManifestInfo{
		GGUFSize:  manifest.GGUFFile.Size,
		TotalSize: manifest.GGUFFile.Size,
		IsVision:  manifest.MMProjFile != nil,
	}
	if info.IsVision {
		info.MMProjSize = manifest.MMProjFile.Size
		info.TotalSize += info.MMProjSize
	}

	return info, manifest, manifestJSON, nil
}

// PullOptions configures a model pull operation.
type PullOptions struct {
	// Pre-fetched manifest to avoid duplicate API calls.
	// If nil, PullModel will fetch it.
	Manifest     *Manifest
	ManifestJSON []byte
}

// fileDownload tracks a file to download and its metadata.
type fileDownload struct {
	file     *ManifestFile
	destPath string
}

// PullModel downloads a model from HuggingFace using the manifest API.
// It handles downloading the GGUF file, optional mmproj for vision models,
// split GGUF files, hash verification, and saving the manifest for future reference.
func PullModel(client *Client, user, repo string, quant Quantization, opts *PullOptions, progress func(PullProgress)) (*PullResult, error) {
	manifest, manifestJSON, err := getOrFetchManifest(client, user, repo, quant, opts)
	if err != nil {
		return nil, err
	}

	splitInfo := ParseSplitFilename(manifest.GGUFFile.RFilename)
	if splitInfo != nil && splitInfo.SplitNo != 0 {
		return nil, fmt.Errorf("manifest references split %d, expected first split", splitInfo.SplitNo+1)
	}

	// Fetch split file info if needed
	if splitInfo != nil {
		splitFiles, err := fetchSplitFileInfo(client, user, repo, splitInfo)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch split file info: %w", err)
		}
		manifest.SplitFiles = splitFiles
	}

	result := calculateResultSizes(manifest, splitInfo)

	// Create model directory
	modelDir := GetModelPath(user, repo)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create model directory: %w", err)
	}

	// Build list of files to download
	files, err := buildFileList(user, repo, quant, manifest, splitInfo, result)
	if err != nil {
		return nil, err
	}

	// Download all files
	if err := downloadAllFiles(client, user, repo, files, result.TotalSize, progress); err != nil {
		cleanupFiles(files, splitInfo, user, repo, quant)
		return nil, err
	}

	if err := verifyAllFiles(files, result.TotalSize, progress); err != nil {
		cleanupFiles(files, splitInfo, user, repo, quant)
		return nil, err
	}

	// Save manifest
	if err := saveManifest(user, repo, quant.Name, manifest, manifestJSON); err != nil {
		return nil, err
	}

	return result, nil
}

// getOrFetchManifest returns the manifest from opts or fetches it.
func getOrFetchManifest(client *Client, user, repo string, quant Quantization, opts *PullOptions) (*Manifest, []byte, error) {
	if opts != nil && opts.Manifest != nil {
		return opts.Manifest, opts.ManifestJSON, nil
	}

	if client == nil {
		return nil, nil, fmt.Errorf("HuggingFace client is required")
	}

	manifest, manifestJSON, err := client.GetManifest(user, repo, quant.Tag)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	if manifest.GGUFFile == nil {
		return nil, nil, fmt.Errorf("manifest does not contain a GGUF file")
	}

	return manifest, manifestJSON, nil
}

// calculateResultSizes computes the PullResult size fields.
func calculateResultSizes(manifest *Manifest, splitInfo *SplitInfo) *PullResult {
	result := &PullResult{
		GGUFSize:  manifest.GGUFFile.Size,
		TotalSize: manifest.GGUFFile.Size,
		IsVision:  manifest.MMProjFile != nil,
	}

	if result.IsVision {
		result.MMProjSize = manifest.MMProjFile.Size
		result.TotalSize += result.MMProjSize
	}

	if splitInfo != nil {
		totalGGUFSize := manifest.GGUFFile.Size
		for _, sf := range manifest.SplitFiles {
			totalGGUFSize += sf.Size
		}
		result.GGUFSize = totalGGUFSize
		result.TotalSize = totalGGUFSize
		if result.IsVision {
			result.TotalSize += result.MMProjSize
		}
	}

	return result
}

// buildFileList creates the list of files to download.
func buildFileList(user, repo string, quant Quantization, manifest *Manifest, splitInfo *SplitInfo, result *PullResult) ([]fileDownload, error) {
	var files []fileDownload

	if splitInfo != nil {
		splitDir := GetSplitModelDir(user, repo, quant.Name)
		if err := os.MkdirAll(splitDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create split directory: %w", err)
		}

		// First split
		firstPath := filepath.Join(splitDir, filepath.Base(manifest.GGUFFile.RFilename))
		result.ModelPath = firstPath
		files = append(files, fileDownload{file: manifest.GGUFFile, destPath: firstPath})

		// Remaining splits
		for _, sf := range manifest.SplitFiles {
			path := filepath.Join(splitDir, filepath.Base(sf.RFilename))
			files = append(files, fileDownload{file: sf, destPath: path})
		}
	} else {
		result.ModelPath = GetModelFilePath(user, repo, quant.Name)
		files = append(files, fileDownload{file: manifest.GGUFFile, destPath: result.ModelPath})
	}

	if result.IsVision {
		result.MMProjPath = GetMMProjFilePath(user, repo, quant.Name)
		files = append(files, fileDownload{file: manifest.MMProjFile, destPath: result.MMProjPath})
	}

	return files, nil
}

// downloadAllFiles downloads all files from HuggingFace.
func downloadAllFiles(client *Client, user, repo string, files []fileDownload, totalSize int64, progress func(PullProgress)) error {
	downloaded := int64(0)

	for i := range files {
		fd := &files[i]

		progressFn := func(current, total int64) {
			if progress != nil {
				progress(PullProgress{
					Phase:   "download",
					Current: downloaded + current,
					Total:   totalSize,
				})
			}
		}

		if err := downloadFromHF(client, user, repo, fd.file, fd.destPath, progressFn); err != nil {
			return err
		}
		downloaded += fd.file.Size
	}

	return nil
}

// downloadFromHF downloads a file from HuggingFace.
func downloadFromHF(client *Client, user, repo string, file *ManifestFile, destPath string, progress func(current, total int64)) error {
	if client == nil {
		return fmt.Errorf("HuggingFace client is required")
	}

	downloader := NewDownloaderWithProgress(client, func(current, total int64, speed float64, eta time.Duration) {
		if progress != nil {
			progress(current, total)
		}
	})

	_, err := downloader.DownloadModel(user, repo, "main", file.RFilename, destPath)
	return err
}

// verifyAllFiles verifies all downloaded files.
func verifyAllFiles(files []fileDownload, totalSize int64, progress func(PullProgress)) error {
	verified := int64(0)

	for i := range files {
		fd := &files[i]

		if fd.file.LFS == nil || fd.file.LFS.SHA256 == "" {
			verified += fd.file.Size
			continue
		}

		progressFn := func(current, total int64) {
			if progress != nil {
				progress(PullProgress{
					Phase:   "verify",
					Current: verified + current,
					Total:   totalSize,
				})
			}
		}

		if err := verifyFile(fd.destPath, fd.file.LFS.SHA256, progressFn); err != nil {
			os.Remove(fd.destPath)
			return fmt.Errorf("verification failed for %s: %w", filepath.Base(fd.destPath), err)
		}

		verified += fd.file.Size
	}

	return nil
}

// verifyFile checks a file's SHA256 hash.
func verifyFile(path, expectedHash string, progress func(current, total int64)) error {
	hash, err := CalculateSHA256WithProgress(path, progress)
	if err != nil {
		return fmt.Errorf("failed to calculate hash: %w", err)
	}
	if !strings.EqualFold(hash, expectedHash) {
		return fmt.Errorf("hash mismatch")
	}
	return nil
}

// cleanupFiles removes downloaded files on error.
func cleanupFiles(files []fileDownload, splitInfo *SplitInfo, user, repo string, quant Quantization) {
	if splitInfo != nil {
		os.RemoveAll(GetSplitModelDir(user, repo, quant.Name))
	} else {
		for _, fd := range files {
			os.Remove(fd.destPath)
		}
	}
}

// saveManifest saves the manifest to disk.
func saveManifest(user, repo, quant string, manifest *Manifest, manifestJSON []byte) error {
	var manifestData []byte
	var err error

	if len(manifest.SplitFiles) > 0 {
		manifestData, err = json.Marshal(manifest)
		if err != nil {
			return fmt.Errorf("failed to marshal manifest: %w", err)
		}
	} else {
		manifestData = manifestJSON
	}

	manifestPath := GetManifestFilePath(user, repo, quant)
	if err := os.WriteFile(manifestPath, manifestData, 0644); err != nil {
		return fmt.Errorf("failed to save manifest: %w", err)
	}

	return nil
}

// fetchSplitFileInfo fetches LFS metadata for all split files (except the first, which is in the manifest).
func fetchSplitFileInfo(client *Client, user, repo string, splitInfo *SplitInfo) ([]*ManifestFile, error) {
	if splitInfo.SplitCount <= 1 {
		return nil, nil
	}

	dirPath := filepath.Dir(splitInfo.Prefix)
	if dirPath == "." {
		dirPath = ""
	}

	files, err := client.ListFilesInPath(user, repo, "main", dirPath)
	if err != nil {
		return nil, fmt.Errorf("failed to list split files: %w", err)
	}

	fileMap := make(map[string]FileTree)
	for _, f := range files {
		fileMap[filepath.Base(f.Path)] = f
	}

	var splitFiles []*ManifestFile
	for i := 1; i < splitInfo.SplitCount; i++ {
		splitPath := SplitPath(splitInfo.Prefix, i, splitInfo.SplitCount)
		splitName := filepath.Base(splitPath)

		ft, ok := fileMap[splitName]
		if !ok {
			return nil, fmt.Errorf("split file %s not found in repository", splitName)
		}

		mf := &ManifestFile{
			RFilename: splitPath,
			Size:      ft.Size,
		}

		if ft.LFS.OID != "" {
			mf.LFS = &ManifestLFS{
				SHA256: ft.LFS.OID,
				Size:   ft.LFS.Size,
			}
		}

		splitFiles = append(splitFiles, mf)
	}

	return splitFiles, nil
}

// CheckForUpdates checks if a local model is up to date with the remote manifest.
func CheckForUpdates(client *Client, user, repo string, quant Quantization) (bool, bool, *Manifest, []byte, error) {
	manifest, manifestJSON, err := client.GetManifest(user, repo, quant.Tag)
	if err != nil {
		return false, false, nil, nil, fmt.Errorf("failed to get manifest: %w", err)
	}

	if manifest.GGUFFile == nil {
		return false, false, nil, nil, fmt.Errorf("manifest does not contain a GGUF file")
	}

	upToDate, saveManifest := isUpToDate(user, repo, quant.Name, manifest)
	return upToDate, saveManifest, manifest, manifestJSON, nil
}

// isUpToDate checks if local files match the remote manifest.
func isUpToDate(user, repo, quant string, remote *Manifest) (bool, bool) {
	manifestPath := GetManifestFilePath(user, repo, quant)
	modelPath := FindModelFile(user, repo, quant)

	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		if modelPath != "" {
			modelInfo, statErr := os.Stat(modelPath)
			if statErr == nil && modelInfo.Size() == remote.GGUFFile.Size {
				if remote.MMProjFile != nil {
					mmprojPath := GetMMProjFilePath(user, repo, quant)
					if _, err := os.Stat(mmprojPath); err != nil {
						return false, false
					}
				}
				return true, true
			}
		}
		return false, false
	}

	var local Manifest
	if err := json.Unmarshal(manifestData, &local); err != nil {
		return false, false
	}

	if !hashesMatch(local.GGUFFile, remote.GGUFFile) {
		return false, false
	}

	if remote.MMProjFile != nil {
		if !hashesMatch(local.MMProjFile, remote.MMProjFile) {
			return false, false
		}
		mmprojPath := GetMMProjFilePath(user, repo, quant)
		if _, err := os.Stat(mmprojPath); err != nil {
			return false, false
		}
	}

	if modelPath == "" {
		return false, false
	}

	return true, false
}

func hashesMatch(local, remote *ManifestFile) bool {
	if local == nil || remote == nil {
		return local == nil && remote == nil
	}
	if local.LFS == nil || remote.LFS == nil {
		return local.Size == remote.Size
	}
	return local.LFS.SHA256 == remote.LFS.SHA256
}

// ProgressDisplay handles progress bar display for pull operations.
type ProgressDisplay interface {
	Start(label string, total int64)
	Update(current, total int64)
	Finish(label string)
	Stop()
}

// ProgressDisplayFactory creates new progress displays.
type ProgressDisplayFactory func() ProgressDisplay

// PullModelWithProgressFactory downloads a model with customizable progress display.
func PullModelWithProgressFactory(client *Client, user, repo string, quant Quantization, opts *PullOptions, factory ProgressDisplayFactory) (*PullResult, error) {
	var progressBar ProgressDisplay
	var currentPhase string

	result, err := PullModel(client, user, repo, quant, opts, func(p PullProgress) {
		if factory == nil {
			return
		}
		if p.Phase != currentPhase {
			if progressBar != nil {
				if currentPhase == "download" {
					progressBar.Finish("Downloaded")
				} else {
					progressBar.Finish("Verified")
				}
			}
			currentPhase = p.Phase
			progressBar = factory()
			if p.Phase == "download" {
				progressBar.Start("", p.Total)
			} else {
				progressBar.Start("Verifying", p.Total)
			}
		}
		if progressBar != nil {
			progressBar.Update(p.Current, p.Total)
		}
	})

	if progressBar != nil {
		if err != nil {
			progressBar.Stop()
		} else if currentPhase == "download" {
			progressBar.Finish("Downloaded")
		} else {
			progressBar.Finish("Verified")
		}
	}

	return result, err
}
