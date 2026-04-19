package proxy

// Translation between the Anthropic Messages API (/v1/messages) and the
// OpenAI Chat Completions API (/v1/chat/completions). Anthropic translation
// happens here so backends only need to speak OpenAI; this lets llama-server
// and SwiftLM (which has no Anthropic endpoint) present a consistent surface.
//
// Scope for this pass:
//   - Text content: full support (string or array of {type:"text", text:...}).
//   - Image content: base64 (media_type restricted to Anthropic's accepted set)
//     and url sources both supported.
//   - Tool calling (request-level tools/tool_choice, plus tool_use and
//     tool_result content blocks): rejected with a 400 because the two APIs
//     disagree enough that a partial translation would silently misbehave.
//     Will revisit when SwiftLM's tool-calling story firms up. Reference
//     implementation: llama.cpp's tools/server/server-common.cpp
//     convert_anthropic_to_oai() shows the full mapping (tool_use →
//     tool_calls, tool_result → role:tool, tool_choice {auto|any|tool} →
//     {auto|required|{name:...}}).
//   - Extended-thinking blocks (thinking, redacted_thinking) and document
//     blocks: rejected with a 400. llama.cpp forwards thinking as
//     reasoning_content on the OpenAI side and emits signature_delta on
//     the Anthropic side — worth adopting once SwiftLM's thinking story is
//     clear.
//
// Known limitations relative to Anthropic's public spec (by design for now):
//   - message_start.usage.input_tokens is always 0 in streamed responses.
//     OpenAI with stream_options.include_usage only reports prompt_tokens in
//     the final chunk; backfilling message_start would require buffering the
//     entire stream and break TTFT. llama.cpp cheats by pre-tokenizing on
//     the same process — we can't, without a separate tokenize round-trip.
//   - stop_reason is never "stop_sequence" because OpenAI does not report
//     which stop string matched. We map finish_reason="stop" to "end_turn".
//   - OpenAI finish_reason="content_filter" maps to "end_turn" — Anthropic
//     has no direct analog in the stop_reason enum.
//   - Periodic ping events are not emitted; intermediaries that idle-close
//     SSE connections may drop very long streams.
//   - cache_creation_input_tokens / cache_read_input_tokens are not emitted.
//     Requires backend-side prompt-cache tracking we don't currently have.
//   - count_tokens is a char-ratio approximation, not a real tokenize call.
//     llama.cpp runs the actual tokenizer in-process; our proxy would need
//     a backend round-trip (or to load a tokenizer) for parity.
//   - The anthropic-version request header is not enforced (local proxy).
//   - Forward-compat: unknown top-level fields (service_tier, container,
//     thinking, output_config, etc.) and cache_control hints on supported
//     blocks are silently ignored rather than rejected.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// --- Anthropic request/response types ---

type anthropicMessagesRequest struct {
	Model         string             `json:"model"`
	Messages      []anthropicMessage `json:"messages"`
	System        json.RawMessage    `json:"system,omitempty"` // string or array of {type,text}
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Tools         json.RawMessage    `json:"tools,omitempty"`
	ToolChoice    json.RawMessage    `json:"tool_choice,omitempty"`
	Metadata      json.RawMessage    `json:"metadata,omitempty"` // ignored
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or array of content blocks
}

type anthropicContentBlock struct {
	Type   string                `json:"type"`
	Text   string                `json:"text,omitempty"`
	Source *anthropicImageSource `json:"source,omitempty"`
	// tool_use / tool_result fields intentionally not modeled — rejected at request-parse time.
}

type anthropicImageSource struct {
	Type      string `json:"type"`                 // "base64" | "url"
	MediaType string `json:"media_type,omitempty"` // for base64 (e.g. "image/png")
	Data      string `json:"data,omitempty"`       // for base64
	URL       string `json:"url,omitempty"`        // for url
}

type anthropicMessagesResponse struct {
	ID           string                     `json:"id"`
	Type         string                     `json:"type"` // always "message"
	Role         string                     `json:"role"` // always "assistant"
	Content      []anthropicResponseContent `json:"content"`
	Model        string                     `json:"model"`
	StopReason   string                     `json:"stop_reason,omitempty"`
	StopSequence *string                    `json:"stop_sequence"`
	Usage        anthropicUsage             `json:"usage"`
}

type anthropicResponseContent struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// --- OpenAI types we produce / consume ---

type openAIChatRequest struct {
	Model         string            `json:"model"`
	Messages      []openAIMessage   `json:"messages"`
	MaxTokens     int               `json:"max_tokens,omitempty"`
	Temperature   *float64          `json:"temperature,omitempty"`
	TopP          *float64          `json:"top_p,omitempty"`
	TopK          *int              `json:"top_k,omitempty"`
	Stop          []string          `json:"stop,omitempty"`
	Stream        bool              `json:"stream,omitempty"`
	StreamOptions *openAIStreamOpts `json:"stream_options,omitempty"`
}

// openAIStreamOpts controls streaming extensions we need. include_usage
// asks the backend to emit a final chunk with token counts so the
// Anthropic message_delta event can carry real usage numbers.
type openAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type openAIMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or multimodal array
}

type openAIContentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *openAIImageURL `json:"image_url,omitempty"`
}

type openAIImageURL struct {
	URL string `json:"url"`
}

type openAIChatResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
}

type openAIChoice struct {
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// openAIStreamChunk is a single SSE event body from /v1/chat/completions.
// `Error` is present when a backend interrupts a stream with an error
// payload (OpenAI convention: `{"error": {"message": "...", "type": "..."}}`).
// Anthropic's equivalent is an `event: error` SSE event, which translateAnthropicStream
// emits on demand.
type openAIStreamChunk struct {
	ID      string               `json:"id"`
	Model   string               `json:"model"`
	Choices []openAIStreamChoice `json:"choices"`
	Usage   *openAIUsage         `json:"usage,omitempty"`
	Error   *openAIStreamError   `json:"error,omitempty"`
}

type openAIStreamError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

type openAIStreamChoice struct {
	Index        int         `json:"index"`
	Delta        openAIDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type openAIDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// --- Translation: request ---

// translateAnthropicRequest converts an Anthropic /v1/messages request to an
// OpenAI /v1/chat/completions request. Returns the translated body, whether
// the caller asked for streaming, and a translation error. Translation errors
// carry an HTTP status so the handler can map them to Anthropic error types.
type translateError struct {
	status  int
	errType AnthropicErrorType
	msg     string
}

func (e *translateError) Error() string { return e.msg }

func newBadRequest(msg string) *translateError {
	return &translateError{status: 400, errType: AnthropicInvalidRequest, msg: msg}
}

// maxStopSequences mirrors Anthropic's documented cap (4) so a malicious or
// buggy client can't push unbounded stop lists into the backend.
const maxStopSequences = 4

// allowedImageMediaTypes restricts the media_type clients can pass in image
// blocks. Matches Anthropic's accepted set; defense-in-depth against the
// client-supplied value being embedded into a data URL.
var allowedImageMediaTypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/gif":  true,
	"image/webp": true,
}

// rawJSONPresent reports whether a json.RawMessage is a meaningful value —
// not absent and not the literal `null`. Without this, `"tool_choice": null`
// would be incorrectly treated as present (4 bytes) and rejected.
func rawJSONPresent(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

func translateAnthropicRequest(body []byte) (openAIBody []byte, stream bool, err error) {
	var req anthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, false, newBadRequest("Failed to parse request body as JSON")
	}

	if rawJSONPresent(req.Tools) || rawJSONPresent(req.ToolChoice) {
		return nil, false, newBadRequest("tools/tool_choice are not yet supported in /v1/messages; use /v1/chat/completions directly")
	}

	// Anthropic requires max_tokens. Enforce it so a missing/zero value
	// doesn't silently turn into llama-server's n_predict=-1 (runaway).
	if req.MaxTokens <= 0 {
		return nil, false, newBadRequest("max_tokens: Field required (must be > 0)")
	}

	if len(req.StopSequences) > maxStopSequences {
		return nil, false, newBadRequest(fmt.Sprintf("stop_sequences: at most %d entries allowed", maxStopSequences))
	}

	out := openAIChatRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		Stop:        req.StopSequences,
		Stream:      req.Stream,
	}
	// Ask the backend for usage in the final streaming chunk so we can fill
	// in Anthropic's message_delta.usage instead of always reporting zeros.
	if req.Stream {
		out.StreamOptions = &openAIStreamOpts{IncludeUsage: true}
	}

	if len(req.System) > 0 {
		sysContent, err := flattenAnthropicSystem(req.System)
		if err != nil {
			return nil, false, err
		}
		if sysContent != "" {
			raw, _ := json.Marshal(sysContent)
			out.Messages = append(out.Messages, openAIMessage{
				Role:    "system",
				Content: raw,
			})
		}
	}

	for i, m := range req.Messages {
		translated, err := translateAnthropicMessage(m)
		if err != nil {
			return nil, false, newBadRequest(fmt.Sprintf("messages[%d]: %s", i, err.Error()))
		}
		out.Messages = append(out.Messages, translated)
	}

	b, err := json.Marshal(out)
	if err != nil {
		return nil, false, fmt.Errorf("marshal translated request: %w", err)
	}
	return b, req.Stream, nil
}

// flattenAnthropicSystem accepts either a JSON string or an array of
// {type:"text", text:"..."} blocks and returns the concatenated text.
func flattenAnthropicSystem(raw json.RawMessage) (string, error) {
	// Try string first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	// Fall back to array of content blocks
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", newBadRequest("system: expected string or array of text blocks")
	}
	var parts []string
	for _, b := range blocks {
		if b.Type != "text" {
			return "", newBadRequest("system: only text blocks are supported")
		}
		parts = append(parts, b.Text)
	}
	// Anthropic concatenates system text blocks with a blank-line separator
	// when forming the model's system prompt. Matching that avoids subtle
	// prompt-wording drift for prompt-sensitive models.
	return strings.Join(parts, "\n\n"), nil
}

// translateAnthropicMessage converts a single Anthropic message (role + content)
// into an OpenAI message. Content may be a string or an array of content blocks.
func translateAnthropicMessage(m anthropicMessage) (openAIMessage, error) {
	// Try string content first
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		raw, _ := json.Marshal(s)
		return openAIMessage{Role: m.Role, Content: raw}, nil
	}

	// Array of content blocks
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		return openAIMessage{}, fmt.Errorf("content: expected string or array of blocks")
	}

	// Initialize as empty slice so an empty content array marshals to `[]`
	// rather than `null`. llama-server rejects null content on some message
	// roles; `[]` is structurally valid.
	parts := []openAIContentPart{}
	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, openAIContentPart{Type: "text", Text: b.Text})
		case "image":
			if b.Source == nil {
				return openAIMessage{}, fmt.Errorf("image: missing source")
			}
			var imageURL string
			switch b.Source.Type {
			case "base64":
				if !allowedImageMediaTypes[b.Source.MediaType] {
					return openAIMessage{}, fmt.Errorf("image: unsupported media_type %q", b.Source.MediaType)
				}
				imageURL = fmt.Sprintf("data:%s;base64,%s", b.Source.MediaType, b.Source.Data)
			case "url":
				if b.Source.URL == "" {
					return openAIMessage{}, fmt.Errorf("image: url source missing url")
				}
				imageURL = b.Source.URL
			default:
				return openAIMessage{}, fmt.Errorf("image: unsupported source type %q", b.Source.Type)
			}
			parts = append(parts, openAIContentPart{
				Type:     "image_url",
				ImageURL: &openAIImageURL{URL: imageURL},
			})
		case "tool_use", "tool_result":
			return openAIMessage{}, fmt.Errorf("%s content blocks are not yet supported", b.Type)
		default:
			return openAIMessage{}, fmt.Errorf("unsupported content block type %q", b.Type)
		}
	}

	raw, err := json.Marshal(parts)
	if err != nil {
		return openAIMessage{}, err
	}
	return openAIMessage{Role: m.Role, Content: raw}, nil
}

// --- Translation: non-streaming response ---

// translateOpenAIResponse converts an OpenAI chat completion response into the
// Anthropic /v1/messages response format. messageID is the id to stamp on the
// Anthropic response (typically generated upstream).
func translateOpenAIResponse(body []byte, messageID string) ([]byte, error) {
	var resp openAIChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("decode openai response: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("openai response has no choices")
	}
	choice := resp.Choices[0]

	text, err := extractOpenAIText(choice.Message.Content)
	if err != nil {
		return nil, err
	}

	out := anthropicMessagesResponse{
		ID:         messageID,
		Type:       "message",
		Role:       "assistant",
		Model:      resp.Model,
		StopReason: mapFinishReason(choice.FinishReason),
		Content:    []anthropicResponseContent{{Type: "text", Text: text}},
		Usage: anthropicUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}

	return json.Marshal(out)
}

// extractOpenAIText pulls a plain text string out of an OpenAI message.Content
// value, which can be either a JSON string or an array of content parts.
func extractOpenAIText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []openAIContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("unexpected content shape in assistant response")
	}
	var buf strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			buf.WriteString(p.Text)
		}
	}
	return buf.String(), nil
}

// mapFinishReason maps OpenAI finish_reason to Anthropic stop_reason.
// Empty / unknown values map to "end_turn" (Anthropic's default).
func mapFinishReason(r string) string {
	switch r {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "":
		return ""
	default:
		return "end_turn"
	}
}

// --- Translation: streaming ---

// streamState accumulates the transform as OpenAI SSE events are consumed
// and exposes helpers for emitting the corresponding Anthropic events.
type streamState struct {
	w                io.Writer
	flush            func()
	messageID, model string
	started          bool // emitted message_start + content_block_start
	completionTokens int
	stopReason       string
}

func newStreamState(w io.Writer, messageID, model string) *streamState {
	flush := func() {}
	if f, ok := w.(interface{ Flush() }); ok {
		flush = f.Flush
	}
	return &streamState{w: w, flush: flush, messageID: messageID, model: model}
}

func (s *streamState) writeEvent(event string, data any) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, b); err != nil {
		return err
	}
	s.flush()
	return nil
}

func (s *streamState) ensureStarted() error {
	if s.started {
		return nil
	}
	s.started = true
	err := s.writeEvent("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            s.messageID,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         s.model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         map[string]int{"input_tokens": 0, "output_tokens": 0},
		},
	})
	if err != nil {
		return err
	}
	// Anthropic's documented content_block_start carries a text block with
	// an empty string. SDKs (notably the Python/TS accumulators) construct a
	// TextBlock from this event and expect `text` to be present.
	return s.writeEvent("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]string{"type": "text", "text": ""},
	})
}

// errStreamAborted signals translateAnthropicStream that the upstream
// reported an error mid-stream and an Anthropic error event has already
// been emitted; the stream loop should stop cleanly.
var errStreamAborted = fmt.Errorf("stream aborted by upstream error")

func (s *streamState) applyChunk(chunk *openAIStreamChunk) error {
	// Upstream signaled an error mid-stream. Translate to Anthropic's
	// `event: error` and stop — further chunks would be semantically
	// meaningless after an error.
	if chunk.Error != nil {
		errType := chunk.Error.Type
		if errType == "" {
			errType = string(AnthropicAPIError)
		}
		if err := s.writeEvent("error", map[string]any{
			"type": "error",
			"error": map[string]string{
				"type":    errType,
				"message": chunk.Error.Message,
			},
		}); err != nil {
			return err
		}
		return errStreamAborted
	}
	if chunk.Usage != nil {
		// Only completion/output tokens are forwarded — message_delta.usage
		// per spec is output-only (cumulative). prompt_tokens is known too
		// late to backfill message_start, which is a documented limitation.
		s.completionTokens = chunk.Usage.CompletionTokens
	}
	for _, c := range chunk.Choices {
		if c.Delta.Content != "" {
			if err := s.ensureStarted(); err != nil {
				return err
			}
			if err := s.writeEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]string{"type": "text_delta", "text": c.Delta.Content},
			}); err != nil {
				return err
			}
		}
		if c.FinishReason != nil && *c.FinishReason != "" {
			s.stopReason = mapFinishReason(*c.FinishReason)
		}
	}
	return nil
}

func (s *streamState) finalize() error {
	// Ensure a valid message frame even if upstream produced nothing.
	// After this, s.started is always true and the block needs its stop event.
	if err := s.ensureStarted(); err != nil {
		return err
	}
	if err := s.writeEvent("content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": 0,
	}); err != nil {
		return err
	}
	stopReason := s.stopReason
	if stopReason == "" {
		stopReason = "end_turn"
	}
	// Per Anthropic's spec, message_delta.usage is the cumulative running
	// total and carries only output_tokens; input_tokens belongs in
	// message_start.message.usage (which is zero today — see known
	// limitations in the package comment).
	if err := s.writeEvent("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]int{
			"output_tokens": s.completionTokens,
		},
	}); err != nil {
		return err
	}
	return s.writeEvent("message_stop", map[string]string{"type": "message_stop"})
}

// translateAnthropicStream reads OpenAI /v1/chat/completions SSE events from
// upstream and emits an equivalent Anthropic /v1/messages SSE stream to w.
// messageID is stamped into the emitted events; model is echoed back.
// The writer must be an http.ResponseWriter configured for streaming; the
// caller is responsible for flushing response headers first.
func translateAnthropicStream(w io.Writer, upstream io.Reader, messageID, model string) error {
	state := newStreamState(w, messageID, model)

	scanner := bufio.NewScanner(upstream)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			// Skip malformed chunks rather than aborting the whole stream.
			continue
		}
		if err := state.applyChunk(&chunk); err != nil {
			if errors.Is(err, errStreamAborted) {
				// Error event already emitted; don't tack on trailing
				// message_delta/message_stop frames that would confuse
				// the client about the final state.
				return nil
			}
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return state.finalize()
}

// --- Token counting ---

// countAnthropicTokens returns an approximate token count for an Anthropic
// /v1/messages/count_tokens request body. Anthropic requires this endpoint
// but llama-server and SwiftLM don't expose an OpenAI-equivalent, so we
// approximate with a characters-per-token ratio. The approximation is
// intentionally conservative (overcount slightly) so callers don't plan a
// prompt that then overruns the context window at inference time.
func countAnthropicTokens(body []byte) (int, error) {
	var req anthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return 0, newBadRequest("Failed to parse request body as JSON")
	}

	// Reject tool/tool_choice for the same reason /v1/messages does —
	// and because tool schemas contribute meaningful token counts that
	// our char-ratio approximation would silently undercount.
	if rawJSONPresent(req.Tools) || rawJSONPresent(req.ToolChoice) {
		return 0, newBadRequest("tools/tool_choice are not yet supported in /v1/messages/count_tokens")
	}

	var total int
	if len(req.System) > 0 {
		sys, err := flattenAnthropicSystem(req.System)
		if err != nil {
			return 0, err
		}
		total += approxTokens(sys)
	}
	for _, m := range req.Messages {
		var s string
		if err := json.Unmarshal(m.Content, &s); err == nil {
			total += approxTokens(s)
			continue
		}
		var blocks []anthropicContentBlock
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			continue
		}
		for _, b := range blocks {
			if b.Type == "text" {
				total += approxTokens(b.Text)
			}
			// image tokens are non-trivial to estimate; omit for now
		}
	}
	return total, nil
}

// approxTokens estimates tokens for a plain text string. Empirically, English
// text runs ~3.5 chars/token across modern BPE tokenizers; we use 3.5 and
// round up so the estimate leans toward overcounting.
func approxTokens(s string) int {
	if s == "" {
		return 0
	}
	// ceil(len*2 / 7) == ceil(len / 3.5)
	return (len(s)*2 + 6) / 7
}
