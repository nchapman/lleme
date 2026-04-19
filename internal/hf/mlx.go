package hf

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
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

	// TODO: add freshness check using HF file sizes / LFS OIDs so re-pulls
	// can skip unchanged files (mirrors the GGUF checkSizeFallback path).
	var (
		downloaded int64
		written    []string
	)
	for _, f := range mlxFiles {
		destPath := filepath.Join(destDir, f.Path)
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
