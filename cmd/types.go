package cmd

import "time"

// ModelInfo represents a locally downloaded model. Backend / Path mirror
// hf.LocalModel so cmd code can delete by the inventory's authoritative
// answer rather than re-deriving paths per backend.
type ModelInfo struct {
	User     string
	Repo     string
	Quant    string
	Backend  string
	Path     string
	Size     int64
	LastUsed time.Time
}
