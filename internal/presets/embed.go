package presets

import (
	"embed"
	"io/fs"
)

//go:embed all:data
var content embed.FS

// PresetsFS returns the embedded preset data directory as an fs.FS.
func PresetsFS() fs.FS {
	fsys, err := fs.Sub(content, "data")
	if err != nil {
		panic("presets: embedded data directory missing: " + err.Error())
	}
	return fsys
}
