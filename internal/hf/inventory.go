package hf

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/nchapman/lleme/internal/config"
)

// loadMetadataFrom is the testable sibling of LoadMetadata that reads from
// an explicit path rather than deriving one from config.ModelsPath(). nil
// return is the "no metadata file" signal, which the caller treats as a
// valid pre-metadata install.
func loadMetadataFrom(path string) *ModelMetadata {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var meta ModelMetadata
	if err := yaml.Unmarshal(data, &meta); err != nil {
		return nil
	}
	if meta.Quants == nil {
		meta.Quants = make(map[string]QuantMetadata)
	}
	return &meta
}

// LocalModel describes a downloaded model ready to serve. It's the
// backend-neutral entry that `lleme list` and `lleme remove` work against.
// Path is the file (single GGUF) or directory (split GGUF, MLX tree) the
// server will be pointed at; Size is the on-disk total, including split
// shards and sibling tokenizer files for MLX.
type LocalModel struct {
	User     string
	Repo     string
	Quant    string
	Backend  string // BackendGGUF | BackendMLX
	Path     string
	Size     int64
	LastUsed time.Time
}

// ListLocalModels enumerates everything pullable from ~/.lleme/models/.
// Source of truth is metadata.yaml; any repo without one (legacy install,
// manual dump) is still picked up via the .gguf fallback scan so users
// don't lose visibility after upgrading.
func ListLocalModels() ([]LocalModel, error) {
	return ListLocalModelsInDir(config.ModelsPath())
}

// ListLocalModelsInDir is the testable form of ListLocalModels that walks an
// explicit modelsDir rather than the configured one. Exported for tests and
// tools; production code calls ListLocalModels.
func ListLocalModelsInDir(modelsDir string) ([]LocalModel, error) {
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read models dir: %w", err)
	}

	var models []LocalModel
	for _, userEntry := range entries {
		if !userEntry.IsDir() {
			continue
		}
		user := userEntry.Name()
		userDir := filepath.Join(modelsDir, user)

		repoEntries, err := os.ReadDir(userDir)
		if err != nil {
			continue
		}
		for _, repoEntry := range repoEntries {
			if !repoEntry.IsDir() {
				continue
			}
			repo := repoEntry.Name()
			models = append(models, listRepoQuants(modelsDir, user, repo)...)
		}
	}
	return models, nil
}

// listRepoQuants returns every quant present for a single user/repo pair.
// Uses metadata.yaml when present (authoritative for backend kind), falls
// back to scanning .gguf files so pre-metadata installs stay visible.
func listRepoQuants(modelsDir, user, repo string) []LocalModel {
	repoDir := filepath.Join(modelsDir, user, repo)
	meta := loadMetadataFrom(filepath.Join(repoDir, "metadata.yaml"))
	seen := make(map[string]bool)
	var out []LocalModel

	if meta != nil {
		for quant, q := range meta.Quants {
			kind := q.Backend
			if kind == "" {
				kind = BackendGGUF
			}
			if m, ok := resolveLocalModel(repoDir, user, repo, quant, kind, q.LastUsed); ok {
				out = append(out, m)
				seen[quant] = true
			}
		}
	}

	// Fallback: a repo without metadata.yaml (or with a quant on disk the
	// metadata never learned about) — scan for .gguf files the way list
	// used to before metadata existed.
	for _, q := range scanLegacyGGUFQuants(repoDir) {
		if seen[q] {
			continue
		}
		if m, ok := resolveLocalModel(repoDir, user, repo, q, BackendGGUF, time.Time{}); ok {
			out = append(out, m)
		}
	}
	return out
}

// resolveLocalModel fills in Path, Size, and LastUsed for a quant of the
// given backend kind under repoDir. Returns (_, false) if nothing is on
// disk for it. repoDir is supplied so the function is testable against an
// arbitrary models tree instead of the configured one.
func resolveLocalModel(repoDir, user, repo, quant, kind string, lastUsed time.Time) (LocalModel, bool) {
	m := LocalModel{User: user, Repo: repo, Quant: quant, Backend: kind, LastUsed: lastUsed}
	switch kind {
	case BackendMLX:
		dir := filepath.Join(repoDir, quant)
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			return m, false
		}
		m.Path = dir
		m.Size = dirSize(dir)
	default: // BackendGGUF or empty/unknown — treat as GGUF
		singlePath := filepath.Join(repoDir, quant+".gguf")
		if info, err := os.Stat(singlePath); err == nil && !info.IsDir() {
			m.Path = singlePath
			m.Size = info.Size()
			if mmproj, err := os.Stat(filepath.Join(repoDir, quant+"-mmproj.gguf")); err == nil {
				m.Size += mmproj.Size()
			}
			break
		}
		splitDir := filepath.Join(repoDir, quant)
		if info, err := os.Stat(splitDir); err == nil && info.IsDir() {
			m.Path = splitDir
			m.Size = dirSize(splitDir)
			break
		}
		return m, false
	}
	if m.LastUsed.IsZero() {
		if info, err := os.Stat(m.Path); err == nil {
			m.LastUsed = info.ModTime()
		}
	}
	return m, true
}

// scanLegacyGGUFQuants discovers quants for a repo with no metadata.yaml.
// Only invoked as a fallback, so it stays narrowly focused on the GGUF
// single-file and split-dir layouts that preceded metadata tracking.
func scanLegacyGGUFQuants(repoDir string) []string {
	entries, err := os.ReadDir(repoDir)
	if err != nil {
		return nil
	}

	var out []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			// Split model stored under user/repo/<quant>/.
			if hasGGUFInside(filepath.Join(repoDir, name)) {
				out = append(out, name)
			}
			continue
		}
		if !strings.HasSuffix(name, ".gguf") {
			continue
		}
		if IsMMProjFileName(name) {
			continue
		}
		out = append(out, strings.TrimSuffix(name, ".gguf"))
	}
	return out
}

func hasGGUFInside(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".gguf") {
			return true
		}
	}
	return false
}

// dirSize sums the sizes of regular files directly inside dir (non-recursive).
// Both GGUF split trees and MLX trees are flat, so recursive walking would
// just be overhead.
func dirSize(dir string) int64 {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	var total int64
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if info, err := e.Info(); err == nil {
			total += info.Size()
		}
	}
	return total
}

// RemoveLocalModel deletes everything on disk for a single quant and
// tidies up empty parent directories. Safe to call for either backend.
func RemoveLocalModel(m LocalModel) error {
	// Primary payload: file for a single GGUF, directory for split GGUF or MLX.
	info, err := os.Stat(m.Path)
	if err != nil {
		return fmt.Errorf("stat %s: %w", m.Path, err)
	}
	if info.IsDir() {
		if err := os.RemoveAll(m.Path); err != nil {
			return fmt.Errorf("remove %s: %w", m.Path, err)
		}
	} else {
		if err := os.Remove(m.Path); err != nil {
			return fmt.Errorf("remove %s: %w", m.Path, err)
		}
	}

	// Associated per-quant files (best-effort — missing is fine).
	_ = os.Remove(GetManifestFilePath(m.User, m.Repo, m.Quant))
	_ = os.Remove(GetMMProjFilePath(m.User, m.Repo, m.Quant))

	// Drop the quant entry from metadata.yaml so it stops appearing in list.
	if meta, err := LoadMetadata(m.User, m.Repo); err == nil {
		delete(meta.Quants, m.Quant)
		if len(meta.Quants) == 0 {
			_ = os.Remove(GetMetadataPath(m.User, m.Repo))
		} else {
			_ = SaveMetadata(m.User, m.Repo, meta)
		}
	}
	return nil
}
