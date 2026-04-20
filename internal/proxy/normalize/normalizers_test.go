package normalize

import (
	"encoding/json"
	"strings"
	"testing"
)

// parse is a test helper that round-trips a JSON string through
// chunkObject so tests can build inputs as readable JSON literals
// rather than constructing maps by hand.
func parse(t *testing.T, s string) chunkObject {
	t.Helper()
	var obj chunkObject
	if err := json.Unmarshal([]byte(s), &obj); err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return obj
}

func encode(t *testing.T, obj chunkObject) string {
	t.Helper()
	out, err := json.Marshal(obj)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return string(out)
}

func TestFilterPrefillProgress(t *testing.T) {
	tests := []struct {
		name string
		in   string
		keep bool
	}{
		{"chat.completion.chunk passthrough", `{"object":"chat.completion.chunk"}`, true},
		{"chat.completion passthrough", `{"object":"chat.completion"}`, true},
		{"prefill_progress dropped", `{"object":"prefill_progress","fraction":0.5}`, false},
		{"missing object passthrough", `{"id":"x"}`, true},
		{"non-string object passthrough", `{"object":42}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := filterPrefillProgress(parse(t, tt.in))
			if got != tt.keep {
				t.Errorf("filterPrefillProgress(%s) = %v, want %v", tt.in, got, tt.keep)
			}
		})
	}
}

func TestRewriteModel(t *testing.T) {
	tests := []struct {
		name      string
		in        string
		requested string
		wantModel string
	}{
		{"swap", `{"model":"backend-internal-id"}`, "user/repo", "user/repo"},
		{"already matches", `{"model":"user/repo"}`, "user/repo", "user/repo"},
		{"absent gets inserted", `{}`, "user/repo", "user/repo"},
		{"empty requested leaves untouched", `{"model":"x"}`, "", "x"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj := parse(t, tt.in)
			rewriteModel(obj, tt.requested)
			var got string
			if raw, ok := obj["model"]; ok {
				_ = json.Unmarshal(raw, &got)
			}
			if got != tt.wantModel {
				t.Errorf("model = %q, want %q", got, tt.wantModel)
			}
		})
	}
}

func TestSynthesizeFingerprint(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string // empty = expect inserted "test-fp"
	}{
		{"absent", `{}`, "test-fp"},
		{"explicit null", `{"system_fingerprint":null}`, "test-fp"},
		{"present preserved", `{"system_fingerprint":"backend-supplied"}`, "backend-supplied"},
		{"empty string preserved", `{"system_fingerprint":""}`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj := parse(t, tt.in)
			synthesizeFingerprint(obj, "test-fp")
			var got string
			_ = json.Unmarshal(obj["system_fingerprint"], &got)
			if got != tt.want {
				t.Errorf("system_fingerprint = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSynthesizeUsageDetails(t *testing.T) {
	tests := []struct {
		name           string
		in             string
		wantPromptDet  string
		wantComplDet   string
		expectNoChange bool
	}{
		{
			name:           "no usage block",
			in:             `{}`,
			expectNoChange: true,
		},
		{
			name:           "explicit null usage",
			in:             `{"usage":null}`,
			expectNoChange: true,
		},
		{
			name:          "neither details key",
			in:            `{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
			wantPromptDet: `{"cached_tokens":0}`,
			wantComplDet:  `{"reasoning_tokens":0}`,
		},
		{
			name:          "prompt details supplied, completion missing",
			in:            `{"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}`,
			wantPromptDet: `{"cached_tokens":4}`,
			wantComplDet:  `{"reasoning_tokens":0}`,
		},
		{
			name:          "both supplied — no overwrite",
			in:            `{"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":7},"completion_tokens_details":{"reasoning_tokens":3}}}`,
			wantPromptDet: `{"cached_tokens":7}`,
			wantComplDet:  `{"reasoning_tokens":3}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj := parse(t, tt.in)
			before := encode(t, obj)
			synthesizeUsageDetails(obj)
			after := encode(t, obj)
			if tt.expectNoChange {
				if before != after {
					t.Errorf("expected no change; before=%s after=%s", before, after)
				}
				return
			}
			var usage chunkObject
			_ = json.Unmarshal(obj["usage"], &usage)
			if got := string(usage["prompt_tokens_details"]); got != tt.wantPromptDet {
				t.Errorf("prompt_tokens_details = %s, want %s", got, tt.wantPromptDet)
			}
			if got := string(usage["completion_tokens_details"]); got != tt.wantComplDet {
				t.Errorf("completion_tokens_details = %s, want %s", got, tt.wantComplDet)
			}
		})
	}
}

// TestApplyChunkNormalizersIdempotent verifies the headline design
// promise: feeding normalized output back through produces no further
// changes. If this ever fails we've introduced a destructive pass.
func TestApplyChunkNormalizersIdempotent(t *testing.T) {
	inputs := []string{
		`{"object":"chat.completion.chunk","model":"backend-id","choices":[]}`,
		`{"object":"chat.completion.chunk","model":"backend-id","usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		`{"object":"chat.completion","model":"backend-id","system_fingerprint":"existing","usage":{"prompt_tokens":1}}`,
	}
	opts := Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"}
	for _, in := range inputs {
		t.Run(in, func(t *testing.T) {
			first := parse(t, in)
			if !applyChunkNormalizers(first, opts) {
				t.Fatalf("first pass dropped frame unexpectedly")
			}
			firstOut := encode(t, first)

			second := parse(t, firstOut)
			if !applyChunkNormalizers(second, opts) {
				t.Fatalf("second pass dropped frame unexpectedly")
			}
			secondOut := encode(t, second)

			if firstOut != secondOut {
				t.Errorf("not idempotent:\nfirst:  %s\nsecond: %s", firstOut, secondOut)
			}
		})
	}
}

// TestApplyChunkNormalizersAdditive proves we never overwrite a
// backend-supplied system_fingerprint or details block. This is the
// "if backends fix their problems, it just works" guarantee.
func TestApplyChunkNormalizersAdditive(t *testing.T) {
	in := `{"object":"chat.completion","model":"x","system_fingerprint":"backend-fp","usage":{"prompt_tokens":1,"prompt_tokens_details":{"cached_tokens":42},"completion_tokens_details":{"reasoning_tokens":7}}}`
	obj := parse(t, in)
	applyChunkNormalizers(obj, Options{RequestedModel: "user/repo", Fingerprint: "lleme-test"})
	out := encode(t, obj)
	for _, mustContain := range []string{`"backend-fp"`, `"cached_tokens":42`, `"reasoning_tokens":7`} {
		if !strings.Contains(out, mustContain) {
			t.Errorf("backend value lost; output=%s want contains %s", out, mustContain)
		}
	}
}
