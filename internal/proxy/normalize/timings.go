package normalize

import (
	"encoding/json"
	"fmt"
	"time"
)

// applyTimings synthesizes a `timings` block on the chunk that
// carries usage, computing predicted_per_second from a wall-clock
// measurement spanning the first content-bearing frame to now.
//
// **Always overwrites** any backend-supplied predicted_n /
// predicted_ms / predicted_per_second. This is the second
// always-overwrite exception in normalize (alongside `model`), and
// the contract is deliberate: users compare backend speeds across
// llama.cpp and SwiftLM, and llama-server's internal decode timer
// measures something subtly different from SwiftLM's (which doesn't
// emit timings at all). A single proxy-side measurement on both
// sides yields apples-to-apples numbers.
//
// Other timings fields the backend supplies (prompt_n, prompt_ms,
// cache_n, etc.) are preserved — we can't measure prompt processing
// from outside the inference engine, so when llama.cpp gives us
// those values they're ground truth.
//
// Triggers only on chunks that carry a non-null usage block with a
// positive completion_tokens. Streaming clients that don't request
// stream_options.include_usage will simply not see timings — that
// matches the backend behavior today.
func (s *streamReader) applyTimings(obj chunkObject) {
	usageRaw, ok := obj["usage"]
	if !ok || isJSONNull(usageRaw) {
		return
	}
	var usage struct {
		CompletionTokens int `json:"completion_tokens"`
	}
	if err := json.Unmarshal(usageRaw, &usage); err != nil || usage.CompletionTokens <= 0 {
		return
	}
	if !s.sawFirstChunk || s.firstContentAt.IsZero() {
		// Final chunk (the one with usage) often arrives without any
		// preceding content frames — e.g. an immediate refusal or a
		// very short generation that fits in the same chunk as usage.
		// We can't synthesize a meaningful rate without a first-token
		// timestamp, so leave timings alone in that case.
		return
	}
	elapsed := time.Since(s.firstContentAt).Seconds()
	if elapsed <= 0 {
		return
	}

	// Preserve backend-supplied non-decode fields (prompt_n,
	// prompt_ms, cache_n, etc.) and overwrite only the predicted_*
	// triplet. If the backend didn't supply timings at all we start
	// from an empty map.
	var timings chunkObject
	if existing, ok := obj["timings"]; ok && !isJSONNull(existing) {
		if err := json.Unmarshal(existing, &timings); err != nil {
			timings = chunkObject{}
		}
	} else {
		timings = chunkObject{}
	}

	rate := float64(usage.CompletionTokens) / elapsed
	elapsedMs := elapsed * 1000
	timings["predicted_n"] = json.RawMessage(fmt.Sprintf("%d", usage.CompletionTokens))
	timings["predicted_ms"] = json.RawMessage(fmt.Sprintf("%.3f", elapsedMs))
	timings["predicted_per_second"] = json.RawMessage(fmt.Sprintf("%.6f", rate))
	// Overwrite predicted_per_token_ms too (llama.cpp emits it
	// alongside the triplet above). Leaving it from the backend
	// would yield a value derived from the backend's predicted_ms,
	// inconsistent with our overwritten one.
	timings["predicted_per_token_ms"] = json.RawMessage(fmt.Sprintf("%.6f", elapsedMs/float64(usage.CompletionTokens)))

	encoded, err := json.Marshal(timings)
	if err != nil {
		return
	}
	obj["timings"] = encoded
}

// markFirstContent captures the wall-clock instant of the first
// content-bearing frame. Called from emitFrame *before*
// applyChunkNormalizers so the timestamp is recorded even if a
// later normalizer drops the frame. We only consider a frame
// "content-bearing" if it has at least one choice with non-empty
// content or reasoning_content — keepalive frames, role-only
// preamble chunks, and the trailing usage chunk don't count.
func (s *streamReader) markFirstContent(obj chunkObject) {
	if s.sawFirstChunk {
		return
	}
	if !chunkHasContent(obj) {
		return
	}
	s.firstContentAt = time.Now()
	s.sawFirstChunk = true
}

// chunkHasContent reports whether a chat-completion chunk carries
// any visible content the model has generated. Used to anchor the
// timings clock to the actual decode start, not to TTFT artifacts
// like the role-only first delta many backends emit.
func chunkHasContent(obj chunkObject) bool {
	choicesRaw, ok := obj["choices"]
	if !ok {
		return false
	}
	var choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
	}
	if err := json.Unmarshal(choicesRaw, &choices); err != nil {
		return false
	}
	for _, c := range choices {
		if c.Delta.Content != "" || c.Delta.ReasoningContent != "" {
			return true
		}
	}
	return false
}
