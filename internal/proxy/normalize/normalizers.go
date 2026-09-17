package normalize

import (
	"bytes"
	"encoding/json"
)

// chunkObject is the parsed top-level of one OpenAI response or SSE
// chunk, kept as a map so unknown fields round-trip untouched. Every
// normalizer mutates this map; the caller re-marshals at the end. Using
// json.RawMessage for values means we never touch nested structure we
// don't have an opinion on.
type chunkObject map[string]json.RawMessage

// applyChunkNormalizers runs all per-chunk passes against an SSE frame's
// parsed payload. The passes are commutative.
func applyChunkNormalizers(obj chunkObject, opts Options) {
	rewriteModel(obj, opts.RequestedModel)
	synthesizeFingerprint(obj, opts.Fingerprint)
	synthesizeUsageDetails(obj)
}

// applyResponseNormalizers runs the non-streaming variants.
func applyResponseNormalizers(obj chunkObject, opts Options) {
	rewriteModel(obj, opts.RequestedModel)
	synthesizeFingerprint(obj, opts.Fingerprint)
	synthesizeUsageDetails(obj)
}

// rewriteModel always sets "model" to the requested name so the response
// matches the request — the contract callers depend on. Empty requested
// model leaves the field alone — happens when callers don't care
// (Anthropic path).
//
// This is the one normalizer that overwrites; the contract is that the
// response model name matches the request.
func rewriteModel(obj chunkObject, model string) {
	if model == "" {
		return
	}
	encoded, err := json.Marshal(model)
	if err != nil {
		return
	}
	obj["model"] = encoded
}

// synthesizeFingerprint inserts a stable system_fingerprint when the
// backend omits one (or sends null). Strict OpenAI SDK consumers expect
// the field to be present and non-null. The value isn't load-bearing —
// presence is.
//
// Additive: never overwrites a backend-supplied string.
func synthesizeFingerprint(obj chunkObject, fingerprint string) {
	if raw, ok := obj["system_fingerprint"]; ok && !isJSONNull(raw) {
		return
	}
	encoded, err := json.Marshal(fingerprint)
	if err != nil {
		return
	}
	obj["system_fingerprint"] = encoded
}

// synthesizeUsageDetails fills in prompt_tokens_details and
// completion_tokens_details on the embedded usage block when missing.
// Strict OpenAI SDK consumers expect both blocks; zero defaults are
// honest when the backend doesn't report cached or reasoning tokens.
//
// No-op when "usage" is absent (typical streaming chunks). Additive
// per-key: if backend supplies one details block but not the other,
// the missing one is filled and the present one is left alone.
func synthesizeUsageDetails(obj chunkObject) {
	raw, ok := obj["usage"]
	if !ok || isJSONNull(raw) {
		return
	}
	var usage chunkObject
	if err := json.Unmarshal(raw, &usage); err != nil {
		return
	}
	changed := false
	if _, has := usage["prompt_tokens_details"]; !has {
		usage["prompt_tokens_details"] = json.RawMessage(`{"cached_tokens":0}`)
		changed = true
	}
	if _, has := usage["completion_tokens_details"]; !has {
		usage["completion_tokens_details"] = json.RawMessage(`{"reasoning_tokens":0}`)
		changed = true
	}
	if !changed {
		return
	}
	encoded, err := json.Marshal(usage)
	if err != nil {
		return
	}
	obj["usage"] = encoded
}

// isJSONNull reports whether a raw value is the JSON literal null. Used
// so absent and explicit-null are treated identically by the synthesis
// passes — both mean "backend didn't give us a value, supply one."
func isJSONNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}
