package proxy

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/hf"
)

// DownloadedModel represents a model that has been downloaded locally
type DownloadedModel struct {
	User      string
	Repo      string
	Quant     string
	FullName  string // "user/repo:quant"
	ModelPath string // Absolute path to .gguf file
}

// ModelResolver handles fuzzy matching of model names against downloaded models
type ModelResolver struct {
	modelsPath string
}

// NewModelResolver creates a new model resolver
func NewModelResolver() *ModelResolver {
	return &ModelResolver{
		modelsPath: config.ModelsPath(),
	}
}

// ListDownloadedModels returns all downloaded models
func (r *ModelResolver) ListDownloadedModels() ([]DownloadedModel, error) {
	var models []DownloadedModel
	seenSplitDirs := make(map[string]bool)

	err := filepath.WalkDir(r.modelsPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || filepath.Ext(d.Name()) != ".gguf" {
			return nil
		}

		relPath, err := filepath.Rel(r.modelsPath, path)
		if err != nil {
			return err
		}

		parts := strings.Split(relPath, string(filepath.Separator))
		if len(parts) < 3 {
			return nil
		}

		user, repo := parts[0], parts[1]

		if m, ok := classifyGGUFEntry(user, repo, parts, path, seenSplitDirs); ok {
			models = append(models, m)
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return models, nil
}

// classifyGGUFEntry determines whether a GGUF file is a split or single-file model
// and returns the corresponding DownloadedModel. Returns (model, false) if the file
// should be skipped (e.g., a duplicate split shard).
func classifyGGUFEntry(user, repo string, parts []string, path string, seenSplitDirs map[string]bool) (DownloadedModel, bool) {
	if len(parts) == 4 && hf.SplitFilePattern.MatchString(filepath.Base(path)) {
		quant := parts[2]
		key := filepath.Join(user, repo, quant)
		if seenSplitDirs[key] {
			return DownloadedModel{}, false
		}
		seenSplitDirs[key] = true
		firstPath := hf.FindFirstSplitFile(filepath.Dir(path))
		if firstPath == "" {
			firstPath = path
		}
		return DownloadedModel{
			User:      user,
			Repo:      repo,
			Quant:     quant,
			FullName:  fmt.Sprintf("%s/%s:%s", user, repo, quant),
			ModelPath: firstPath,
		}, true
	}

	quant := strings.TrimSuffix(filepath.Base(path), ".gguf")
	return DownloadedModel{
		User:      user,
		Repo:      repo,
		Quant:     quant,
		FullName:  fmt.Sprintf("%s/%s:%s", user, repo, quant),
		ModelPath: path,
	}, true
}

// ResolveResult contains the result of a model resolution
type ResolveResult struct {
	Model       *DownloadedModel
	Matches     []DownloadedModel // All matching models (for ambiguous case)
	Suggestions []DownloadedModel // Fuzzy suggestions (for no match case)
}

// Resolve attempts to find a downloaded model matching the given query
// Returns:
// - Exact match: Model is set, Matches has 1 item
// - Ambiguous: Model is nil, Matches has multiple items
// - No match: Model is nil, Matches is empty, Suggestions may have items
func (r *ModelResolver) Resolve(query string) (*ResolveResult, error) {
	models, err := r.ListDownloadedModels()
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	if len(models) == 0 {
		return &ResolveResult{}, nil
	}

	query = strings.ToLower(strings.TrimSpace(query))

	for _, try := range []func([]DownloadedModel, string) (*ResolveResult, bool){
		tryExactMatch,
		tryRepoMatch,
		trySuffixMatch,
		tryContainsMatch,
	} {
		if result, ok := try(models, query); ok {
			return result, nil
		}
	}

	return &ResolveResult{Suggestions: fuzzyMatch(query, models)}, nil
}

// tryExactMatch matches the full "user/repo:quant" name.
func tryExactMatch(models []DownloadedModel, query string) (*ResolveResult, bool) {
	for i := range models {
		if strings.ToLower(models[i].FullName) == query {
			return &ResolveResult{Model: &models[i], Matches: []DownloadedModel{models[i]}}, true
		}
	}
	return nil, false
}

// tryRepoMatch matches "user/repo" (without quant), picking the best quant when multiple exist.
// All matches are guaranteed to share the same user/repo, so we pick best quant directly.
func tryRepoMatch(models []DownloadedModel, query string) (*ResolveResult, bool) {
	if strings.Contains(query, ":") {
		return nil, false
	}
	var matches []DownloadedModel
	for i := range models {
		if strings.ToLower(fmt.Sprintf("%s/%s", models[i].User, models[i].Repo)) == query {
			matches = append(matches, models[i])
		}
	}
	if len(matches) == 0 {
		return nil, false
	}
	return &ResolveResult{Model: pickBestQuant(matches), Matches: matches}, true
}

// trySuffixMatch matches "repo:quant" or just "repo".
func trySuffixMatch(models []DownloadedModel, query string) (*ResolveResult, bool) {
	var matches []DownloadedModel
	for i := range models {
		repoQuant := strings.ToLower(fmt.Sprintf("%s:%s", models[i].Repo, models[i].Quant))
		repo := strings.ToLower(models[i].Repo)
		if repoQuant == query || repo == query {
			matches = append(matches, models[i])
		}
	}
	return resolveCandidates(matches)
}

// tryContainsMatch matches any model whose full name contains the query.
func tryContainsMatch(models []DownloadedModel, query string) (*ResolveResult, bool) {
	var matches []DownloadedModel
	for i := range models {
		if strings.Contains(strings.ToLower(models[i].FullName), query) {
			matches = append(matches, models[i])
		}
	}
	return resolveCandidates(matches)
}

// resolveCandidates converts a candidate list into a ResolveResult:
// - 0 candidates: no match
// - 1 candidate: unambiguous match
// - N same-repo candidates: pick best quant
// - N different-repo candidates: ambiguous
func resolveCandidates(candidates []DownloadedModel) (*ResolveResult, bool) {
	switch len(candidates) {
	case 0:
		return nil, false
	case 1:
		return &ResolveResult{Model: &candidates[0], Matches: candidates}, true
	default:
		if allSameRepo(candidates) {
			return &ResolveResult{Model: pickBestQuant(candidates), Matches: candidates}, true
		}
		return &ResolveResult{Matches: candidates}, true
	}
}

// allSameRepo checks if all models are from the same user/repo
func allSameRepo(models []DownloadedModel) bool {
	if len(models) == 0 {
		return true
	}
	first := fmt.Sprintf("%s/%s", models[0].User, models[0].Repo)
	for _, m := range models[1:] {
		if fmt.Sprintf("%s/%s", m.User, m.Repo) != first {
			return false
		}
	}
	return true
}

// pickBestQuant returns the model with the best quantization
func pickBestQuant(models []DownloadedModel) *DownloadedModel {
	if len(models) == 0 {
		return nil
	}

	best := &models[0]
	bestPriority := hf.GetQuantPriority(best.Quant)

	for i := 1; i < len(models); i++ {
		p := hf.GetQuantPriority(models[i].Quant)
		if p < bestPriority {
			best = &models[i]
			bestPriority = p
		}
	}

	return best
}

// fuzzyMatch finds models with similar names (for typo suggestions)
func fuzzyMatch(query string, models []DownloadedModel) []DownloadedModel {
	type scored struct {
		model DownloadedModel
		score int
	}

	var results []scored

	for _, m := range models {
		// Calculate a simple edit distance score
		score := levenshtein(query, strings.ToLower(m.FullName))
		// Also check against just the repo name
		repoScore := levenshtein(query, strings.ToLower(m.Repo))
		if repoScore < score {
			score = repoScore
		}
		// Only include if reasonably close
		if score <= len(query)/2+3 {
			results = append(results, scored{m, score})
		}
	}

	// Sort by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].score < results[j].score
	})

	// Return top 3
	var suggestions []DownloadedModel
	for i := 0; i < len(results) && i < 3; i++ {
		suggestions = append(suggestions, results[i].model)
	}

	return suggestions
}

// levenshtein calculates the edit distance between two strings
func levenshtein(a, b string) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}

	// Create matrix
	matrix := make([][]int, len(a)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(b)+1)
		matrix[i][0] = i
	}
	for j := range matrix[0] {
		matrix[0][j] = j
	}

	// Fill matrix
	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			matrix[i][j] = min(
				matrix[i-1][j]+1,      // deletion
				matrix[i][j-1]+1,      // insertion
				matrix[i-1][j-1]+cost, // substitution
			)
		}
	}

	return matrix[len(a)][len(b)]
}
