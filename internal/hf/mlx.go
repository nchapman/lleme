package hf

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/nchapman/lleme/internal/binaryrelease"
	"github.com/nchapman/lleme/internal/fileutil"
)

// mlxQuantSuffix matches the quantization suffix on mlx-community repos
// (e.g. "-4bit", "-8bit", "-bf16", "-fp16"). Anchored so we only catch the
// trailing token — avoids false matches inside repo names like "4bit-eval".
var mlxQuantSuffix = regexp.MustCompile(`(?i)-((?:[0-9]+bit)|bf16|fp16|fp32)$`)

// DetectFormat classifies a HuggingFace repo by file listing. It prefers
// GGUF when both are present (historical default for lleme), so accidental
// stragglers in an MLX conversion don't redirect the pull. Returns "" when
// neither format is detectable.
func DetectFormat(files []FileTree) string {
	var hasGGUF, hasConfig, hasSafetensors bool
	for _, f := range files {
		name := filepath.Base(f.Path)
		switch {
		case strings.HasSuffix(name, ".gguf"):
			hasGGUF = true
		case name == "config.json":
			hasConfig = true
		case strings.HasSuffix(name, ".safetensors"):
			hasSafetensors = true
		}
	}
	switch {
	case hasGGUF:
		return BackendGGUF
	case hasConfig && hasSafetensors:
		return BackendMLX
	default:
		return ""
	}
}

// MLXQuantFromRepo derives a quant label from the repo name's trailing
// suffix (e.g. "Qwen3-4B-Instruct-4bit" → "4bit"). Falls back to "default"
// when no suffix matches — some MLX conversions are unquantized.
func MLXQuantFromRepo(repo string) string {
	m := mlxQuantSuffix.FindStringSubmatch(repo)
	if m == nil {
		return "default"
	}
	return strings.ToLower(m[1])
}

// isMLXModelFile reports whether a file in an MLX repo is part of the
// runtime artifact we need to serve it. Keeping this allow-listed avoids
// pulling READMEs, eval data, and PyTorch-side weights we don't use.
func isMLXModelFile(name string) bool {
	switch name {
	case "config.json",
		"tokenizer.json",
		"tokenizer.model",
		"tokenizer_config.json",
		"special_tokens_map.json",
		"generation_config.json",
		"chat_template.jinja",
		"added_tokens.json",
		"vocab.json",
		"merges.txt",
		"model.safetensors.index.json":
		return true
	}
	return strings.HasSuffix(name, ".safetensors")
}

// ListMLXFiles filters a repo tree down to files required by the MLX
// backend, in the order they appear in the tree.
func ListMLXFiles(files []FileTree) []FileTree {
	var out []FileTree
	for _, f := range files {
		if f.Type == "directory" {
			continue
		}
		if isMLXModelFile(filepath.Base(f.Path)) {
			out = append(out, f)
		}
	}
	return out
}

// mlxManifestVersion is the schema version written into the MLX manifest.
// Bump when the manifest shape changes incompatibly; existing callers can
// reject or migrate old versions.
const mlxManifestVersion = "1"

// MLXManifest records what files were downloaded for an MLX quant. Paths
// are relative to the quant directory; sizes are the Content-Length values
// reported by HuggingFace. LFS OIDs would be more authoritative but the
// tree API reports them inconsistently for non-LFS files — size-only
// comparison catches the common case (HF re-upload with new weights) and
// misses the rare one (content-identical, size-identical edit).
type MLXManifest struct {
	Version string            `json:"version"`
	Files   []MLXManifestFile `json:"files"`
}

type MLXManifestFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// GetMLXManifestFilePath returns the sibling path where an MLX manifest is
// stored. Lives beside the quant directory (not inside it) so the manifest
// survives a "rm -rf" of just the model contents the same way GGUF manifests
// do at `<quant>-manifest.json`.
func GetMLXManifestFilePath(user, repo, quant string) string {
	modelDir := GetModelPath(user, repo)
	return filepath.Join(modelDir, quant+"-mlx-manifest.json")
}

// SaveMLXManifest writes manifest atomically.
func SaveMLXManifest(user, repo, quant string, files []FileTree) error {
	entries := make([]MLXManifestFile, 0, len(files))
	for _, f := range files {
		entries = append(entries, MLXManifestFile{Path: f.Path, Size: f.Size})
	}
	m := MLXManifest{Version: mlxManifestVersion, Files: entries}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal mlx manifest: %w", err)
	}
	path := GetMLXManifestFilePath(user, repo, quant)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("mkdir for mlx manifest: %w", err)
	}
	return fileutil.AtomicWriteFile(path, data, 0644)
}

// LoadMLXManifest reads a saved manifest. Returns (nil, nil) when absent
// (pre-manifest pulls or a legacy install) so callers can fall back.
func LoadMLXManifest(user, repo, quant string) (*MLXManifest, error) {
	path := GetMLXManifestFilePath(user, repo, quant)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var m MLXManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse mlx manifest: %w", err)
	}
	return &m, nil
}

// CheckForUpdatesMLX reports whether a locally-pulled MLX quant matches the
// remote tree. Returns (upToDate, saveManifest, remoteFiles, err):
//   - upToDate: every required file exists locally with matching size.
//   - saveManifest: true when we matched via the fallback path (all files
//     present at expected sizes) but no manifest was recorded yet — the
//     caller should persist one so the next check is a cheap JSON compare.
//   - remoteFiles: the filtered MLX file tree from HuggingFace, so a caller
//     that decides to re-download doesn't need to re-list.
//
// Mirrors the GGUF pair (CheckForUpdates + checkSizeFallback).
func CheckForUpdatesMLX(client *Client, user, repo, quant string) (bool, bool, []FileTree, error) {
	if client == nil {
		return false, false, nil, fmt.Errorf("HuggingFace client is required")
	}
	files, err := client.ListFiles(user, repo, "main")
	if err != nil {
		return false, false, nil, fmt.Errorf("list files: %w", err)
	}
	mlxFiles := ListMLXFiles(files)
	if len(mlxFiles) == 0 {
		return false, false, mlxFiles, fmt.Errorf("no MLX files found in %s/%s", user, repo)
	}

	destDir := GetSplitModelDir(user, repo, quant)
	manifest, err := LoadMLXManifest(user, repo, quant)
	if err != nil {
		return false, false, mlxFiles, err
	}

	if manifest != nil {
		return mlxManifestMatches(destDir, mlxFiles, manifest), false, mlxFiles, nil
	}
	// Fallback: no manifest recorded. Check that every expected file is
	// present with matching size. If so, signal saveManifest=true so the
	// caller writes one and subsequent checks are fast.
	return mlxFilesMatchDisk(destDir, mlxFiles), true, mlxFiles, nil
}

// mlxManifestMatches compares the recorded manifest against both the remote
// tree and the local filesystem. Any divergence — remote file added or
// resized, local file missing or resized — signals "not up to date".
func mlxManifestMatches(destDir string, remote []FileTree, manifest *MLXManifest) bool {
	recorded := make(map[string]int64, len(manifest.Files))
	for _, f := range manifest.Files {
		recorded[f.Path] = f.Size
	}
	if len(recorded) != len(remote) {
		return false
	}
	for _, f := range remote {
		size, ok := recorded[f.Path]
		if !ok || size != f.Size {
			return false
		}
		local, err := os.Stat(filepath.Join(destDir, f.Path))
		if err != nil || local.Size() != f.Size {
			return false
		}
	}
	return true
}

// mlxFilesMatchDisk is the manifest-less fallback: every remote file must
// exist locally with the advertised size.
func mlxFilesMatchDisk(destDir string, remote []FileTree) bool {
	for _, f := range remote {
		local, err := os.Stat(filepath.Join(destDir, f.Path))
		if err != nil || local.Size() != f.Size {
			return false
		}
	}
	return true
}

// PullMLXModel downloads all MLX artifacts for a repo into a per-quant
// directory and records backend=mlx in metadata.yaml. The caller selects
// quant (typically via MLXQuantFromRepo) so we can host multiple MLX
// variants of the same repo side-by-side.
func PullMLXModel(client *Client, user, repo, quant string, progress func(PullProgress)) (*PullResult, error) {
	if client == nil {
		return nil, fmt.Errorf("HuggingFace client is required")
	}

	files, err := client.ListFiles(user, repo, "main")
	if err != nil {
		return nil, fmt.Errorf("failed to list files: %w", err)
	}

	mlxFiles := ListMLXFiles(files)
	if len(mlxFiles) == 0 {
		return nil, fmt.Errorf("no MLX files found in %s/%s", user, repo)
	}

	destDir := GetSplitModelDir(user, repo, quant)
	dirExisted := true
	if _, err := os.Stat(destDir); os.IsNotExist(err) {
		dirExisted = false
	}
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create model directory: %w", err)
	}

	var totalSize int64
	for _, f := range mlxFiles {
		totalSize += f.Size
	}

	result := &PullResult{
		ModelPath: destDir,
		TotalSize: totalSize,
	}

	var (
		downloaded int64
		written    []string
	)
	absDest, err := filepath.Abs(destDir)
	if err != nil {
		cleanupMLX(dirExisted, destDir, written)
		return nil, fmt.Errorf("resolve destination: %w", err)
	}
	for _, f := range mlxFiles {
		// f.Path comes verbatim from the HuggingFace tree API — an attacker
		// who controls the repo can return "../../.ssh/id_rsa". SafeJoin
		// refuses absolute paths, .. traversal, and any join that escapes
		// the model directory.
		destPath, err := binaryrelease.SafeJoin(absDest, f.Path)
		if err != nil {
			cleanupMLX(dirExisted, destDir, written)
			return nil, fmt.Errorf("reject MLX file %q: %w", f.Path, err)
		}
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			cleanupMLX(dirExisted, destDir, written)
			return nil, fmt.Errorf("failed to create subdir for %s: %w", f.Path, err)
		}

		progressFn := func(current, total int64) {
			if progress != nil {
				progress(PullProgress{
					Phase:   "download",
					Current: downloaded + current,
					Total:   totalSize,
				})
			}
		}

		if err := downloadFromHF(client, user, repo, &ManifestFile{RFilename: f.Path, Size: f.Size}, destPath, progressFn); err != nil {
			cleanupMLX(dirExisted, destDir, written)
			return nil, fmt.Errorf("download %s: %w", f.Path, err)
		}
		written = append(written, destPath)
		downloaded += f.Size
	}

	if err := SetBackendKind(user, repo, quant, BackendMLX); err != nil {
		return nil, fmt.Errorf("failed to record backend kind: %w", err)
	}

	// Persist the file manifest so subsequent `lleme pull` invocations can
	// short-circuit via CheckForUpdatesMLX. Failure to write the manifest
	// shouldn't undo a successful pull — the next pull just falls back to
	// the size-check path.
	if err := SaveMLXManifest(user, repo, quant, mlxFiles); err != nil {
		// Best-effort: log would be ideal, but we don't import logs here.
		// The next pull will try again.
		_ = err
	}

	return result, nil
}

// cleanupMLX undoes a partial pull. If the destination directory didn't
// exist before this pull, we remove the whole thing; otherwise we only
// delete files we wrote this session so a previous good copy survives.
func cleanupMLX(dirExisted bool, destDir string, written []string) {
	if !dirExisted {
		os.RemoveAll(destDir)
		return
	}
	for _, p := range written {
		os.Remove(p)
	}
}

// PullMLXModelWithProgress is the progress-bar-aware wrapper around
// PullMLXModel, mirroring PullModelWithProgressFactory for GGUF.
func PullMLXModelWithProgress(client *Client, user, repo, quant string, factory ProgressDisplayFactory) (*PullResult, error) {
	var tracker *phaseTracker
	if factory != nil {
		tracker = &phaseTracker{factory: factory}
	}

	result, err := PullMLXModel(client, user, repo, quant, func(p PullProgress) {
		if tracker == nil {
			return
		}
		if p.Phase != tracker.phase {
			tracker.transition(p.Phase, p.Total)
		}
		tracker.update(p.Current, p.Total)
	})

	if tracker != nil {
		tracker.done(err)
	}

	return result, err
}
