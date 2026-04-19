package presets

import (
	"embed"
	"fmt"
	"io/fs"
)

//go:embed all:data
var content embed.FS

// PresetsFS returns the embedded preset data directory as an fs.FS.
// Returns an error only if the embed.FS is misconfigured at build time; callers
// can safely propagate the error rather than crashing the program.
func PresetsFS() (fs.FS, error) {
	fsys, err := fs.Sub(content, "data")
	if err != nil {
		return nil, fmt.Errorf("presets: embedded data directory missing: %w", err)
	}
	return fsys, nil
}
