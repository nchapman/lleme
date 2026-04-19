package hf

import (
	"regexp"
	"sort"
	"strings"
)

var (
	// Matches quantization suffixes in filenames
	// Supports: Q*, IQ*, TQ*, FP16/32, F16/32, BF16, I4/8, with optional UD- prefix
	quantPattern = regexp.MustCompile(`(?i)[._-]((?:UD-)?(?:Q[0-9]+[^.]+|IQ[0-9]+[^.]*|TQ[0-9]+[^.]*|FP16|FP32|F16|F32|BF16|I[0-9]+))\.gguf`)

	// Preferred quantizations for auto-selection (best balance of quality/size first)
	// UD (Unsloth Dynamic) variants are prioritized when available
	quantOrder = []string{
		"UD-Q4_K_XL",
		"Q4_K_M",
		"UD-Q5_K_XL",
		"Q4_K_S",
		"Q5_K_M",
		"Q5_K_S",
		"UD-Q6_K_XL",
		"Q6_K",
		"UD-Q8_K_XL",
		"Q8_0",
		"UD-Q3_K_XL",
		"Q3_K_M",
		"Q3_K_S",
		"Q3_K_L",
		"UD-Q2_K_XL",
		"Q2_K",
		"Q2_K_S",
		"Q2_K_L",
		"Q4_0",
		"Q4_1",
		"IQ4_XS",
		"IQ4_NL",
		"UD-IQ3_XXS",
		"IQ3_XXS",
		"UD-IQ2_M",
		"UD-IQ2_XXS",
		"IQ2_XXS",
		"IQ2_M",
		"UD-IQ1_M",
		"UD-IQ1_S",
		"IQ1_M",
		"IQ1_S",
		"UD-TQ1_0",
		"TQ1_0",
		"TQ2_0",
		"FP16",
		"F16",
		"BF16",
		"FP32",
		"F32",
	}
)

type Quantization struct {
	Name string // Display name (e.g., "Q4_K_M", "UD-Q4_K_XL")
	Tag  string // Tag for HuggingFace API calls (same as Name)
	File string
	Size int64
}

// ParseQuantization extracts the quantization from a filename.
// Returns the quantization name as it appears in the filename (preserving hyphens).
// Returns "" if no quantization found.
func ParseQuantization(filename string) string {
	matches := quantPattern.FindStringSubmatch(filename)
	if len(matches) < 2 {
		return ""
	}
	return strings.ToUpper(matches[1])
}

// quantDirPattern matches directory names that look like quantization names
// Supports: Q*, IQ*, TQ*, FP16/32, F16/32, BF16, I4/8, with optional UD- prefix
var quantDirPattern = regexp.MustCompile(`^(?i)((?:UD-)?(?:Q[0-9]+[^/]*|IQ[0-9]+[^/]*|TQ[0-9]+[^/]*|FP16|FP32|F16|F32|BF16|I[0-9]+))$`)

func ExtractQuantizations(files []FileTree) []Quantization {
	var quants []Quantization
	seenQuants := make(map[string]bool)

	for _, file := range files {
		// Check for GGUF files
		if strings.HasSuffix(file.Path, ".gguf") {
			name := ParseQuantization(file.Path)
			tag := name
			if name == "" {
				// GGUF file without quantization suffix - use "default"/"latest"
				name = "default"
				tag = "latest"
			}

			if seenQuants[name] {
				continue
			}
			seenQuants[name] = true

			quants = append(quants, Quantization{
				Name: name,
				Tag:  tag,
				File: file.Path,
				Size: file.Size,
			})
			continue
		}

		// Check for directories that look like quantization names
		// These contain split files or nested GGUF files
		if file.Type == "directory" && quantDirPattern.MatchString(file.Path) {
			// Uppercase for display consistency, preserve original for API calls
			name := strings.ToUpper(file.Path)

			if seenQuants[name] {
				continue
			}
			seenQuants[name] = true

			quants = append(quants, Quantization{
				Name: name,
				Tag:  file.Path, // Original case for API calls
				File: "",        // Will be resolved from manifest
				Size: 0,         // Size unknown until manifest fetch
			})
		}
	}

	return quants
}

func GetBestQuantization(quants []Quantization) string {
	if len(quants) == 0 {
		return ""
	}

	for _, preferred := range quantOrder {
		for _, q := range quants {
			if q.Name == preferred {
				return q.Name
			}
		}
	}

	return quants[0].Name
}

func SortQuantizations(quants []Quantization) []Quantization {
	sort.Slice(quants, func(i, j int) bool {
		return quants[i].Name < quants[j].Name
	})
	return quants
}

func FindQuantization(quants []Quantization, name string) (Quantization, bool) {
	for _, q := range quants {
		if strings.EqualFold(q.Name, name) || strings.EqualFold(q.Tag, name) {
			return q, true
		}
	}
	return Quantization{}, false
}

// quantPriorityMap is derived from quantOrder for O(1) lookups
var quantPriorityMap map[string]int

func init() {
	quantPriorityMap = make(map[string]int, len(quantOrder))
	for i, q := range quantOrder {
		quantPriorityMap[strings.ToUpper(q)] = i
	}
}

// GetQuantPriority returns the priority score for a quantization (lower is better).
// Matching is case-insensitive. Returns a high value (1000) for unknown quantizations.
func GetQuantPriority(quant string) int {
	if p, ok := quantPriorityMap[strings.ToUpper(quant)]; ok {
		return p
	}
	return 1000
}

// FormatModelName returns a display name for the model, omitting the quant suffix for "default".
func FormatModelName(user, repo, quant string) string {
	if quant == "" || quant == "default" {
		return user + "/" + repo
	}
	return user + "/" + repo + ":" + quant
}
