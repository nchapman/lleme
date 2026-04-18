package hf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
)

// GGUF format constants
const (
	ggufMagic = "GGUF"

	// GGUF value types
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

	// Key for split count
	keySplitCount = "split.count"
)

// SplitFilePattern matches split GGUF files like "model-00001-of-00002.gguf"
var SplitFilePattern = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// GGUFHeader contains the basic header info from a GGUF file.
type GGUFHeader struct {
	Version    uint32
	TensorCnt  int64
	KVCnt      int64
	SplitCount int // 0 if not a split file, otherwise the total number of splits
}

// ReadGGUFHeader reads the GGUF header and key-value metadata from a file.
// It specifically looks for the split.count key to detect split files.
func ReadGGUFHeader(path string) (*GGUFHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return readGGUFHeader(f)
}

func readGGUFHeader(r io.Reader) (*GGUFHeader, error) {
	version, tensorCnt, kvCnt, err := ReadGGUFPreamble(r)
	if err != nil {
		return nil, err
	}

	header := &GGUFHeader{
		Version:   version,
		TensorCnt: int64(tensorCnt),
		KVCnt:     int64(kvCnt),
	}

	for i := uint64(0); i < kvCnt; i++ {
		key, err := readGGUFString(r)
		if err != nil {
			return nil, fmt.Errorf("failed to read key %d: %w", i, err)
		}

		var valType uint32
		if err := binary.Read(r, binary.LittleEndian, &valType); err != nil {
			return nil, fmt.Errorf("failed to read value type for key %q: %w", key, err)
		}

		if key == keySplitCount && valType == ggufTypeUint16 {
			var splitCount uint16
			if err := binary.Read(r, binary.LittleEndian, &splitCount); err != nil {
				return nil, fmt.Errorf("failed to read split.count: %w", err)
			}
			header.SplitCount = int(splitCount)
			return header, nil
		}

		if err := SkipGGUFValue(r, valType); err != nil {
			return nil, fmt.Errorf("failed to skip value for key %q: %w", key, err)
		}
	}

	return header, nil
}

// ReadGGUFPreamble reads and validates the GGUF magic, version, tensor count,
// and KV count from an io.Reader, positioning it at the first KV pair.
func ReadGGUFPreamble(r io.Reader) (version uint32, tensorCnt uint64, kvCnt uint64, err error) {
	magic := make([]byte, 4)
	if _, err = io.ReadFull(r, magic); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to read magic: %w", err)
	}
	if string(magic) != ggufMagic {
		return 0, 0, 0, fmt.Errorf("invalid GGUF magic: %q", string(magic))
	}

	if err = binary.Read(r, binary.LittleEndian, &version); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to read version: %w", err)
	}
	if err = binary.Read(r, binary.LittleEndian, &tensorCnt); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to read tensor count: %w", err)
	}
	if err = binary.Read(r, binary.LittleEndian, &kvCnt); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to read kv count: %w", err)
	}
	return version, tensorCnt, kvCnt, nil
}

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

// SplitPrefix extracts the prefix from a split filename.
// For example, given "model-00001-of-00003.gguf", returns "model".
// Returns empty string if the filename doesn't match the split pattern.
func SplitPrefix(path string, splitNo, splitCount int) string {
	expected := fmt.Sprintf("-%05d-of-%05d.gguf", splitNo+1, splitCount)
	if len(path) > len(expected) && path[len(path)-len(expected):] == expected {
		return path[:len(path)-len(expected)]
	}
	return ""
}
