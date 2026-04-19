package hf

import (
	"encoding/binary"
	"fmt"
	"io"
	"regexp"
	"strconv"
)

// GGUF value type constants used by SkipGGUFValue.
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// SplitFilePattern matches split GGUF files like "model-00001-of-00002.gguf"
var SplitFilePattern = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}

	if length > 1024*1024 { // Sanity check: 1MB max string length
		return "", fmt.Errorf("string too long: %d", length)
	}

	data := make([]byte, length)
	if _, err := io.ReadFull(r, data); err != nil {
		return "", err
	}

	return string(data), nil
}

// SkipGGUFValue advances the reader past a GGUF value of the given type.
func SkipGGUFValue(r io.Reader, valType uint32) error {
	switch valType {
	case ggufTypeUint8, ggufTypeInt8, ggufTypeBool:
		_, err := io.CopyN(io.Discard, r, 1)
		return err
	case ggufTypeUint16, ggufTypeInt16:
		_, err := io.CopyN(io.Discard, r, 2)
		return err
	case ggufTypeUint32, ggufTypeInt32, ggufTypeFloat32:
		_, err := io.CopyN(io.Discard, r, 4)
		return err
	case ggufTypeUint64, ggufTypeInt64, ggufTypeFloat64:
		_, err := io.CopyN(io.Discard, r, 8)
		return err
	case ggufTypeString:
		_, err := readGGUFString(r)
		return err
	case ggufTypeArray:
		var arrType uint32
		if err := binary.Read(r, binary.LittleEndian, &arrType); err != nil {
			return err
		}
		var arrLen uint64
		if err := binary.Read(r, binary.LittleEndian, &arrLen); err != nil {
			return err
		}
		if arrLen > 1024*1024 { // Sanity check: 1M elements max
			return fmt.Errorf("array too long: %d", arrLen)
		}
		for i := uint64(0); i < arrLen; i++ {
			if err := SkipGGUFValue(r, arrType); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unknown GGUF value type: %d", valType)
	}
}

// SplitInfo contains parsed information from a split filename.
type SplitInfo struct {
	Prefix     string // e.g., "Q4_K_M/model-Q4_K_M"
	SplitNo    int    // 0-based index
	SplitCount int    // Total number of splits
}

// ParseSplitFilename parses a split filename and returns split info.
// Returns nil if the filename doesn't match the split pattern.
func ParseSplitFilename(path string) *SplitInfo {
	matches := SplitFilePattern.FindStringSubmatch(path)
	if matches == nil {
		return nil
	}

	splitNo, err := strconv.Atoi(matches[1])
	if err != nil {
		return nil
	}
	splitCount, err := strconv.Atoi(matches[2])
	if err != nil {
		return nil
	}

	// Extract prefix (everything before the split suffix)
	suffixLen := len(matches[0])
	prefix := path[:len(path)-suffixLen]

	return &SplitInfo{
		Prefix:     prefix,
		SplitNo:    splitNo - 1, // Convert to 0-based
		SplitCount: splitCount,
	}
}

// SplitPath generates the path for a split file given the prefix, split index (0-based),
// and total split count. The format matches llama.cpp: {prefix}-{N:05d}-of-{M:05d}.gguf
func SplitPath(prefix string, splitNo, splitCount int) string {
	return fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, splitNo+1, splitCount)
}
