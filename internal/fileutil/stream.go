package fileutil

import "io"

// StreamBody reads from body and writes to dst, starting the byte counter at
// startOffset (non-zero for resume). progress receives (bytesWritten, totalSize)
// after each chunk; pass nil to skip. Returns total bytes written.
func StreamBody(body io.Reader, dst io.Writer, startOffset, totalSize int64, progress func(int64, int64)) (int64, error) {
	buf := make([]byte, 32*1024)
	written := startOffset
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return written, werr
			}
			written += int64(n)
			if progress != nil {
				progress(written, totalSize)
			}
		}
		if err == io.EOF {
			return written, nil
		}
		if err != nil {
			return written, err
		}
	}
}
