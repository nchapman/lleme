package fileutil

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStreamBody(t *testing.T) {
	t.Run("streams all bytes with progress", func(t *testing.T) {
		content := strings.Repeat("x", 100_000)
		var dst bytes.Buffer
		var calls []int64
		written, err := StreamBody(strings.NewReader(content), &dst, 0, int64(len(content)), func(n, total int64) {
			calls = append(calls, n)
			if total != int64(len(content)) {
				t.Errorf("progress total = %d, want %d", total, len(content))
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		if written != int64(len(content)) {
			t.Errorf("written = %d, want %d", written, len(content))
		}
		if dst.String() != content {
			t.Error("content mismatch")
		}
		if len(calls) == 0 || calls[len(calls)-1] != int64(len(content)) {
			t.Errorf("final progress = %v, want %d", calls, len(content))
		}
	})

	t.Run("resume offset counts toward written", func(t *testing.T) {
		var dst bytes.Buffer
		written, err := StreamBody(strings.NewReader("tail"), &dst, 100, 104, nil)
		if err != nil {
			t.Fatal(err)
		}
		if written != 104 {
			t.Errorf("written = %d, want 104 (100 offset + 4 body)", written)
		}
	})

	t.Run("propagates read error", func(t *testing.T) {
		wantErr := errors.New("boom")
		_, err := StreamBody(errReader{wantErr}, &bytes.Buffer{}, 0, 0, nil)
		if !errors.Is(err, wantErr) {
			t.Errorf("err = %v, want %v", err, wantErr)
		}
	})

	t.Run("propagates write error", func(t *testing.T) {
		wantErr := errors.New("disk full")
		_, err := StreamBody(strings.NewReader("data"), errWriter{wantErr}, 0, 0, nil)
		if !errors.Is(err, wantErr) {
			t.Errorf("err = %v, want %v", err, wantErr)
		}
	})
}

type errReader struct{ err error }

func (r errReader) Read([]byte) (int, error) { return 0, r.err }

type errWriter struct{ err error }

func (w errWriter) Write([]byte) (int, error) { return 0, w.err }

func TestAtomicWriteFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "metadata.yaml")

	if err := AtomicWriteFile(path, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "hello" {
		t.Errorf("content = %q, want hello", data)
	}

	// Overwrite in place; no temp file left behind.
	if err := AtomicWriteFile(path, []byte("world"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error("temp file left behind after successful write")
	}
}

func TestAtomicWriteFileFailureLeavesOriginal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "file")
	if err := AtomicWriteFile(path, []byte("original"), 0644); err != nil {
		t.Fatal(err)
	}
	// Point the write at a path whose temp file can't be created (parent
	// is a file, not a directory).
	blocked := filepath.Join(path, "child")
	if err := AtomicWriteFile(blocked, []byte("x"), 0644); err == nil {
		t.Fatal("expected error writing under a file path")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "original" {
		t.Errorf("original modified after failed write: %q", data)
	}
}

var _ io.Reader = errReader{}
