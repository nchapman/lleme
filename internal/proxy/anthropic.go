package proxy

// Translation between the Anthropic Messages API (/v1/messages) and the
// OpenAI Chat Completions API (/v1/chat/completions). Anthropic translation
// happens here so backends only need to speak OpenAI; this lets llama-server
// and SwiftLM (which has no Anthropic endpoint) present a consistent surface.
//
// Scope:
//   - Text content: full support (string or array of {type:"text", text:...}).
//   - Image content: base64 with Anthropic's accepted media_type set. URL
//     sources are rejected with a clear 400 because neither llama-server nor
//     SwiftLM fetch URLs at request time; forwarding one would surface as an
//     opaque backend error.
//   - Tool calling: full round-trip. Request-level tools translate to OpenAI
//     function tools; tool_choice {auto|any|tool|none} maps to OpenAI's
//     {auto|required|function:{name}|none}; assistant tool_use blocks become
//     message.tool_calls; user tool_result blocks split into role:"tool"
//     messages. Responses with finish_reason=="tool_calls" translate back
//     to Anthropic tool_use content blocks with stop_reason="tool_use".
//   - Streaming: tool_call deltas accumulate per OpenAI tool index into their
//     own Anthropic content_block index; function.arguments fragments become
//     input_json_delta partial_json events that SDK accumulators reassemble.
//   - Extended-thinking blocks (thinking, redacted_thinking): dropped rather
//     than rejected. They're advisory; backends that don't understand them
//     would reject the message. Document blocks remain rejected.
//   - Error translation: upstream error types map to Anthropic's enum via
//     normalizeAnthropicErrorType; upstream HTTP status codes map via
//     anthropicErrorTypeForStatus. SDK consumers never see backend-specific
//     error types leaking through.
//
// Known limitations relative to Anthropic's public spec (by design for now):
//   - message_start.usage.input_tokens is always 0 in streamed responses.
//     OpenAI with stream_options.include_usage only reports prompt_tokens in
//     the final chunk; backfilling message_start would require buffering the
//     entire stream and break TTFT.
//   - Periodic `event: ping` frames are emitted every streamPingInterval
//     (default 15s) while a stream is open, so middleware (nginx defaults
//     to a 60s idle-kill; corporate TLS terminators vary) doesn't drop
//     slow-token generation mid-response.
//   - cache_creation_input_tokens / cache_read_input_tokens are not emitted.
//     Requires backend-side prompt-cache tracking we don't currently have.
//   - count_tokens is a char-ratio approximation, not a real tokenize call.
//     Tuned conservative (overcount) so context-budget planners don't
//     under-request and hit a mid-stream context-window overflow.
//   - The anthropic-version request header is not enforced (local proxy).
//   - Forward-compat: unknown top-level fields (service_tier, container,
//     output_config, etc.) and cache_control hints on supported blocks are
//     silently ignored rather than rejected.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
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
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    json.RawMessage    `json:"tool_choice,omitempty"` // object shape varies; parsed separately
	Metadata      json.RawMessage    `json:"metadata,omitempty"`    // ignored
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or array of content blocks
}

// anthropicContentBlock intentionally models every block type we translate —
// tool_use and tool_result are handled in message translation, so they need
// their fields available here. Unused fields tolerate null/empty for block
// types that don't carry them.
type anthropicContentBlock struct {
	Type   string                `json:"type"`
	Text   string                `json:"text,omitempty"`
	Source *anthropicImageSource `json:"source,omitempty"`

	// tool_use (assistant-authored blocks)
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result (user-authored blocks)
	ToolUseID   string          `json:"tool_use_id,omitempty"`
	ToolContent json.RawMessage `json:"content,omitempty"`
	ToolIsError bool            `json:"is_error,omitempty"`
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

// anthropicResponseContent carries either a text block or a tool_use block.
// Fields are emitted conditionally via omitempty so text blocks don't leak
// null tool fields and vice versa.
type anthropicResponseContent struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
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
	Tools         []openAITool      `json:"tools,omitempty"`
	ToolChoice    json.RawMessage   `json:"tool_choice,omitempty"`
}

type openAITool struct {
	Type     string             `json:"type"` // always "function"
	Function openAIToolFunction `json:"function"`
}

type openAIToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// openAIStreamOpts controls streaming extensions we need. include_usage
// asks the backend to emit a final chunk with token counts so the
// Anthropic message_delta event can carry real usage numbers.
type openAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content,omitempty"` // string or multimodal array; may be absent for assistant tool_calls
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"` // set when Role == "tool"
}

type openAIToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"` // always "function"
	Function openAIToolCallFunction `json:"function"`
}

type openAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON-serialized argument object
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
	Index   int           `json:"index"`
	Message openAIMessage `json:"message"`
	// llama-server extension: the stop_sequences entry that terminated
	// generation. Populated when FinishReason == "stop" and the model hit a
	// user-supplied stop string. Not part of the OpenAI spec; we forward it
	// so Anthropic stop_sequence / stop_reason: "stop_sequence" work end-to-end.
	StoppingWord string `json:"stopping_word,omitempty"`
	FinishReason string `json:"finish_reason"`
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
	Index int         `json:"index"`
	Delta openAIDelta `json:"delta"`
	// llama-server extension: see openAIChoice.StoppingWord.
	StoppingWord string  `json:"stopping_word,omitempty"`
	FinishReason *string `json:"finish_reason"`
}

type openAIDelta struct {
	Role      string                `json:"role,omitempty"`
	Content   string                `json:"content,omitempty"`
	ToolCalls []openAIToolCallDelta `json:"tool_calls,omitempty"`
}

// openAIToolCallDelta is the per-chunk slice of a tool call. id / type /
// function.name arrive on the first chunk; function.arguments accumulates
// across subsequent chunks as a string of JSON fragments.
type openAIToolCallDelta struct {
	Index    int                          `json:"index"`
	ID       string                       `json:"id,omitempty"`
	Type     string                       `json:"type,omitempty"`
	Function *openAIToolCallFunctionDelta `json:"function,omitempty"`
}

type openAIToolCallFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
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
// not absent, not the literal `null`, and not an empty structure. Without
// the empty-structure check, `"tool_choice": {}` or `"tools": []` from a
// client that unconditionally attaches the field would be treated as a
// request for tool use.
func rawJSONPresent(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return false
	}
	switch string(trimmed) {
	case "null", "[]", "{}":
		return false
	}
	return true
}

func translateAnthropicRequest(body []byte) (openAIBody []byte, stream bool, err error) {
	var req anthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, false, newBadRequest("Failed to parse request body as JSON")
	}
	if err := validateAnthropicRequest(&req); err != nil {
		return nil, false, err
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
	if err := applyTools(&req, &out); err != nil {
		return nil, false, err
	}
	if err := applySystem(&req, &out); err != nil {
		return nil, false, err
	}
	for i, m := range req.Messages {
		translated, err := translateAnthropicMessage(m)
		if err != nil {
			return nil, false, newBadRequest(fmt.Sprintf("messages[%d]: %s", i, err.Error()))
		}
		out.Messages = append(out.Messages, translated...)
	}

	b, err := json.Marshal(out)
	if err != nil {
		return nil, false, fmt.Errorf("marshal translated request: %w", err)
	}
	return b, req.Stream, nil
}

// validateAnthropicRequest enforces the request invariants we can check
// before translation: max_tokens is required, stop_sequences is capped,
// and tool_choice without tools is a client mistake.
func validateAnthropicRequest(req *anthropicMessagesRequest) error {
	if req.MaxTokens <= 0 {
		return newBadRequest("max_tokens: Field required (must be > 0)")
	}
	if len(req.StopSequences) > maxStopSequences {
		return newBadRequest(fmt.Sprintf("stop_sequences: at most %d entries allowed", maxStopSequences))
	}
	if rawJSONPresent(req.ToolChoice) && len(req.Tools) == 0 {
		return newBadRequest("tool_choice provided without tools")
	}
	return nil
}

// applyTools translates the request-level tools + tool_choice onto out.
func applyTools(req *anthropicMessagesRequest, out *openAIChatRequest) error {
	if len(req.Tools) > 0 {
		translated, err := translateAnthropicTools(req.Tools)
		if err != nil {
			return err
		}
		out.Tools = translated
	}
	if rawJSONPresent(req.ToolChoice) {
		tc, err := translateAnthropicToolChoice(req.ToolChoice)
		if err != nil {
			return err
		}
		out.ToolChoice = tc
	}
	return nil
}

// applySystem flattens the polymorphic system field onto out as a
// role:"system" message when non-empty.
func applySystem(req *anthropicMessagesRequest, out *openAIChatRequest) error {
	if len(req.System) == 0 {
		return nil
	}
	sysContent, err := flattenAnthropicSystem(req.System)
	if err != nil {
		return err
	}
	if sysContent == "" {
		return nil
	}
	raw, _ := json.Marshal(sysContent)
	out.Messages = append(out.Messages, openAIMessage{
		Role:    "system",
		Content: raw,
	})
	return nil
}

// translateAnthropicTools converts Anthropic's tool descriptors to the
// OpenAI `tools` schema. The translation is mechanical: Anthropic's
// {name, description, input_schema} maps 1:1 to OpenAI's
// {type:"function", function:{name, description, parameters}}.
func translateAnthropicTools(tools []anthropicTool) ([]openAITool, error) {
	out := make([]openAITool, 0, len(tools))
	for i, t := range tools {
		if t.Name == "" {
			return nil, newBadRequest(fmt.Sprintf("tools[%d]: name is required", i))
		}
		if len(t.InputSchema) == 0 {
			return nil, newBadRequest(fmt.Sprintf("tools[%d]: input_schema is required", i))
		}
		out = append(out, openAITool{
			Type: "function",
			Function: openAIToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}
	return out, nil
}

// translateAnthropicToolChoice maps Anthropic's tool_choice shape to OpenAI's.
//
// Anthropic accepts:
//
//	"auto"                            // model decides (also: {"type":"auto"})
//	{"type":"any"}                    // model must call a tool
//	{"type":"tool","name":"foo"}      // model must call this specific tool
//	{"type":"none"}                   // disallow tool use
//
// OpenAI accepts:
//
//	"auto" | "required" | "none"
//	{"type":"function","function":{"name":"foo"}}
func translateAnthropicToolChoice(raw json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)

	// String shorthand — only "auto" is documented for Anthropic. "required"
	// / "none" are not in the Anthropic spec but pass through harmlessly to
	// OpenAI backends that accept them.
	if len(trimmed) > 0 && trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, newBadRequest("tool_choice: invalid string value")
		}
		switch s {
		case "auto", "required", "none":
			return json.Marshal(s)
		}
		return nil, newBadRequest(fmt.Sprintf("tool_choice: unsupported string value %q", s))
	}

	var tc struct {
		Type string `json:"type"`
		Name string `json:"name,omitempty"`
	}
	if err := json.Unmarshal(trimmed, &tc); err != nil {
		return nil, newBadRequest("tool_choice: expected object with `type` field")
	}
	switch tc.Type {
	case "auto", "":
		return json.Marshal("auto")
	case "any":
		return json.Marshal("required")
	case "none":
		return json.Marshal("none")
	case "tool":
		if tc.Name == "" {
			return nil, newBadRequest(`tool_choice: {"type":"tool"} requires "name"`)
		}
		return json.Marshal(map[string]any{
			"type":     "function",
			"function": map[string]string{"name": tc.Name},
		})
	}
	return nil, newBadRequest(fmt.Sprintf("tool_choice: unsupported type %q", tc.Type))
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

// translateAnthropicMessage converts a single Anthropic message into one or
// more OpenAI messages. A single Anthropic user message carrying N
// tool_result blocks fans out to N OpenAI role:"tool" messages (plus
// optional user-text preamble); an assistant message with tool_use blocks
// collapses into one OpenAI assistant message with tool_calls.
func translateAnthropicMessage(m anthropicMessage) ([]openAIMessage, error) {
	// String content is the simple case — just wrap it.
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		raw, _ := json.Marshal(s)
		return []openAIMessage{{Role: m.Role, Content: raw}}, nil
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		return nil, fmt.Errorf("content: expected string or array of blocks")
	}

	if m.Role == "assistant" {
		return translateAssistantBlocks(blocks)
	}
	return translateUserBlocks(blocks)
}

// translateAssistantBlocks handles an assistant turn. The allowed block
// types are text and tool_use (plus thinking/redacted_thinking, which we
// drop — they're advisory and backends that don't support them would
// reject the message). Tool calls become openAIToolCalls on a single
// assistant message.
func translateAssistantBlocks(blocks []anthropicContentBlock) ([]openAIMessage, error) {
	var textParts []string
	var toolCalls []openAIToolCall
	for _, b := range blocks {
		switch b.Type {
		case "text":
			textParts = append(textParts, b.Text)
		case "tool_use":
			if b.ID == "" || b.Name == "" {
				return nil, fmt.Errorf("tool_use: id and name are required")
			}
			args := "{}"
			if len(b.Input) > 0 {
				args = string(b.Input)
			}
			toolCalls = append(toolCalls, openAIToolCall{
				ID:   b.ID,
				Type: "function",
				Function: openAIToolCallFunction{
					Name:      b.Name,
					Arguments: args,
				},
			})
		case "thinking", "redacted_thinking":
			// Drop — upstream reasoning markers that aren't part of the
			// conversation the backend needs to see.
		default:
			return nil, fmt.Errorf("unsupported assistant content block type %q", b.Type)
		}
	}
	msg := openAIMessage{Role: "assistant"}
	if len(textParts) > 0 {
		raw, _ := json.Marshal(strings.Join(textParts, ""))
		msg.Content = raw
	}
	msg.ToolCalls = toolCalls
	return []openAIMessage{msg}, nil
}

// translateUserBlocks handles a user turn. Tool results split out into
// their own role:"tool" messages (OpenAI requires one tool message per
// tool_call_id); text and image blocks remain on a single user message,
// preserving ordering.
func translateUserBlocks(blocks []anthropicContentBlock) ([]openAIMessage, error) {
	var out []openAIMessage
	parts := []openAIContentPart{}

	flushUser := func() {
		if len(parts) == 0 {
			return
		}
		raw, _ := json.Marshal(parts)
		out = append(out, openAIMessage{Role: "user", Content: raw})
		parts = []openAIContentPart{}
	}

	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, openAIContentPart{Type: "text", Text: b.Text})
		case "image":
			p, err := translateImageBlock(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, p)
		case "tool_result":
			// Tool results must go on their own role:"tool" message. Flush
			// any pending user content first so the conversation ordering
			// is preserved.
			flushUser()
			toolMsg, err := translateToolResultBlock(b)
			if err != nil {
				return nil, err
			}
			out = append(out, toolMsg)
		default:
			return nil, fmt.Errorf("unsupported user content block type %q", b.Type)
		}
	}
	flushUser()
	return out, nil
}

// translateImageBlock converts an Anthropic image block to OpenAI's
// image_url content part. URL sources are refused with a clear 400 because
// llama-server and SwiftLM don't fetch remote URLs — forwarding would
// produce an opaque backend error. Base64 sources are reformatted into a
// data URL.
func translateImageBlock(b anthropicContentBlock) (openAIContentPart, error) {
	if b.Source == nil {
		return openAIContentPart{}, fmt.Errorf("image: missing source")
	}
	switch b.Source.Type {
	case "base64":
		if !allowedImageMediaTypes[b.Source.MediaType] {
			return openAIContentPart{}, fmt.Errorf("image: unsupported media_type %q", b.Source.MediaType)
		}
		return openAIContentPart{
			Type: "image_url",
			ImageURL: &openAIImageURL{
				URL: fmt.Sprintf("data:%s;base64,%s", b.Source.MediaType, b.Source.Data),
			},
		}, nil
	case "url":
		return openAIContentPart{}, fmt.Errorf("image: url sources are not supported by the selected backend; re-send as base64")
	default:
		return openAIContentPart{}, fmt.Errorf("image: unsupported source type %q", b.Source.Type)
	}
}

// translateToolResultBlock builds a role:"tool" OpenAI message from one
// Anthropic tool_result block. The content is flattened to a string —
// OpenAI's tool role does not carry structured blocks. Errors are prefixed
// with "[error] " so the model can distinguish them from successful results.
func translateToolResultBlock(b anthropicContentBlock) (openAIMessage, error) {
	if b.ToolUseID == "" {
		return openAIMessage{}, fmt.Errorf("tool_result: tool_use_id is required")
	}
	text, err := flattenToolResultContent(b.ToolContent)
	if err != nil {
		return openAIMessage{}, err
	}
	if b.ToolIsError {
		text = "[error] " + text
	}
	raw, _ := json.Marshal(text)
	return openAIMessage{
		Role:       "tool",
		ToolCallID: b.ToolUseID,
		Content:    raw,
	}, nil
}

// flattenToolResultContent accepts the polymorphic Anthropic tool_result
// content: a bare string, an array of content blocks, or a single object.
// Images inside tool_result are intentionally dropped — OpenAI's tool role
// can't carry them, and the model sees the text explanation alone.
func flattenToolResultContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", fmt.Errorf("tool_result.content: expected string or array of blocks")
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n"), nil
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

	// Build content blocks: text first (if any), then tool_use blocks for
	// each tool_call. Omit the text block when there's no text AND there
	// are tool calls — a bare tool_use response shouldn't carry an empty
	// assistant turn.
	var content []anthropicResponseContent
	if text != "" || len(choice.Message.ToolCalls) == 0 {
		content = append(content, anthropicResponseContent{Type: "text", Text: text})
	}
	for _, tc := range choice.Message.ToolCalls {
		input := json.RawMessage(tc.Function.Arguments)
		if len(input) == 0 {
			input = json.RawMessage(`{}`)
		}
		content = append(content, anthropicResponseContent{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}

	stopReason := mapFinishReasonForResponse(choice)
	var stopSequence *string
	if stopReason == "stop_sequence" && choice.StoppingWord != "" {
		sw := choice.StoppingWord
		stopSequence = &sw
	}

	out := anthropicMessagesResponse{
		ID:           messageID,
		Type:         "message",
		Role:         "assistant",
		Model:        resp.Model,
		StopReason:   stopReason,
		StopSequence: stopSequence,
		Content:      content,
		Usage: anthropicUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}

	return json.Marshal(out)
}

// mapFinishReasonForResponse extends the generic mapFinishReason so the
// caller can opt into the "stop_sequence" Anthropic value when llama-server
// reports which user-supplied stop string matched (via choice.StoppingWord).
// Without this signal, every match degrades to the generic "end_turn".
func mapFinishReasonForResponse(choice openAIChoice) string {
	if choice.FinishReason == "stop" && choice.StoppingWord != "" {
		return "stop_sequence"
	}
	if choice.FinishReason == "tool_calls" || choice.FinishReason == "function_call" {
		return "tool_use"
	}
	return mapFinishReason(choice.FinishReason)
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

// normalizeAnthropicErrorType constrains an upstream-supplied error type
// string to Anthropic's enumerated set. Unknown values fall back to
// api_error rather than propagating backend-specific strings ("model_error",
// "BadRequestError", vendor prefixes) into SDK consumers that switch on type.
func normalizeAnthropicErrorType(raw string) AnthropicErrorType {
	switch AnthropicErrorType(raw) {
	case AnthropicInvalidRequest, AnthropicAuthentication, AnthropicPermission,
		AnthropicNotFound, AnthropicRequestTooLarge, AnthropicRateLimit,
		AnthropicAPIError, AnthropicOverloaded:
		return AnthropicErrorType(raw)
	}
	return AnthropicAPIError
}

// anthropicErrorTypeForStatus maps an HTTP status from the backend to the
// Anthropic error type our error response carries. Keeps users' monitoring
// consistent regardless of which backend failed.
func anthropicErrorTypeForStatus(status int) AnthropicErrorType {
	switch status {
	case 400:
		return AnthropicInvalidRequest
	case 401:
		return AnthropicAuthentication
	case 403:
		return AnthropicPermission
	case 404:
		return AnthropicNotFound
	case 413:
		return AnthropicRequestTooLarge
	case 429:
		return AnthropicRateLimit
	case 503:
		return AnthropicOverloaded
	}
	return AnthropicAPIError
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
//
// Anthropic's streaming model puts every content piece (text and each
// tool_use) on its own numbered content_block, with start/delta/stop
// events. OpenAI's stream delivers text as `delta.content` and tool calls
// as `delta.tool_calls[].function.arguments` string fragments across chunks.
// We map each OpenAI tool_call.index to a freshly allocated Anthropic
// content-block index; the text block (index 0) is allocated lazily on
// the first text delta. openToolBlocks tracks which OpenAI tool indices
// have already emitted content_block_start so we don't repeat it.
type streamState struct {
	// writeMu serializes every SSE write because a parallel ping goroutine
	// may interleave with the main chunk-processing loop. Nothing else on
	// streamState is shared across goroutines.
	writeMu          sync.Mutex
	w                io.Writer
	flush            func()
	messageID, model string
	completionTokens int
	stopReason       string
	stopSequence     string // populated when llama-server reports a stop word match

	started        bool // emitted message_start
	textOpen       bool // text content block (Anthropic index = textIndex) started
	textIndex      int  // Anthropic content_block index for the text block
	nextBlockIndex int  // next Anthropic content_block index to allocate

	// openAI tool_call.index → Anthropic content_block index
	toolBlockIndex map[int]int
	// Anthropic content_block index → whether content_block_stop still needed
	openBlocks map[int]bool
}

func newStreamState(w io.Writer, messageID, model string) *streamState {
	flush := func() {}
	if f, ok := w.(interface{ Flush() }); ok {
		flush = f.Flush
	}
	return &streamState{
		w:              w,
		flush:          flush,
		messageID:      messageID,
		model:          model,
		toolBlockIndex: make(map[int]int),
		openBlocks:     make(map[int]bool),
	}
}

func (s *streamState) writeEvent(event string, data any) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, b); err != nil {
		return err
	}
	s.flush()
	return nil
}

// ensureMessageStarted emits the message_start frame once. Callers invoke
// this before allocating any content block, and finalize invokes it so an
// empty upstream still produces a well-formed Anthropic message.
func (s *streamState) ensureMessageStarted() error {
	if s.started {
		return nil
	}
	s.started = true
	return s.writeEvent("message_start", map[string]any{
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
}

// ensureTextBlock emits content_block_start for the text block on first use.
// The text block always sits at index 0 (per Anthropic convention) unless
// a tool block already claimed it — which can't happen here because we
// allocate indices deterministically in the first ensure call.
func (s *streamState) ensureTextBlock() error {
	if s.textOpen {
		return nil
	}
	if err := s.ensureMessageStarted(); err != nil {
		return err
	}
	s.textIndex = s.nextBlockIndex
	s.nextBlockIndex++
	s.textOpen = true
	s.openBlocks[s.textIndex] = true
	// Anthropic's documented content_block_start carries a text block with
	// an empty string. SDKs (notably the Python/TS accumulators) construct a
	// TextBlock from this event and expect `text` to be present.
	return s.writeEvent("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         s.textIndex,
		"content_block": map[string]string{"type": "text", "text": ""},
	})
}

// ensureToolBlock emits content_block_start for an OpenAI tool_call index
// the first time we see it, returning the Anthropic content_block index.
// Name is required by Anthropic; an OpenAI stream that omits it on the
// first frame produces an empty-name tool_use block — non-ideal, but
// better than dropping the tool call entirely.
func (s *streamState) ensureToolBlock(openAIIdx int, toolID, toolName string) (int, error) {
	if idx, ok := s.toolBlockIndex[openAIIdx]; ok {
		return idx, nil
	}
	if err := s.ensureMessageStarted(); err != nil {
		return 0, err
	}
	idx := s.nextBlockIndex
	s.nextBlockIndex++
	s.toolBlockIndex[openAIIdx] = idx
	s.openBlocks[idx] = true
	err := s.writeEvent("content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    toolID,
			"name":  toolName,
			"input": map[string]any{},
		},
	})
	return idx, err
}

// errStreamAborted signals translateAnthropicStream that the upstream
// reported an error mid-stream and an Anthropic error event has already
// been emitted; the stream loop should stop cleanly.
var errStreamAborted = fmt.Errorf("stream aborted by upstream error")

// startPings runs a goroutine that emits an `event: ping` frame every
// `interval` until the returned stop function is called. Pings are
// content-free — their only purpose is to keep idle-sensitive
// intermediaries from dropping the connection. writeEvent serializes with
// the main loop via writeMu so frames never interleave mid-JSON.
//
// startPings only fires the ticker after message_start has been written;
// emitting a ping before any other frame would be out-of-sequence per
// Anthropic's stream grammar. Returns a no-op stop when interval<=0 so
// tests can disable pings without a special path.
func (s *streamState) startPings(interval time.Duration) func() {
	if interval <= 0 {
		return func() {}
	}
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				s.writeMu.Lock()
				started := s.started
				s.writeMu.Unlock()
				if !started {
					continue
				}
				// Ignore write errors: a broken connection is caught by
				// the main loop's next write too; emitting a best-effort
				// ping keeps the design simple.
				_ = s.writeEvent("ping", map[string]string{"type": "ping"})
			}
		}
	}()
	return func() {
		close(stop)
		<-done
	}
}

// emitError writes a synthetic Anthropic `event: error` frame. Used for
// conditions the upstream didn't report as a structured error but that we
// still want the SDK to see — e.g. a mid-stream I/O failure on the
// upstream connection.
func (s *streamState) emitError(errType, message string) error {
	return s.writeEvent("error", map[string]any{
		"type": "error",
		"error": map[string]string{
			"type":    errType,
			"message": message,
		},
	})
}

func (s *streamState) applyChunk(chunk *openAIStreamChunk) error {
	// Upstream signaled an error mid-stream. Translate to Anthropic's
	// `event: error` and stop — further chunks would be semantically
	// meaningless after an error.
	if chunk.Error != nil {
		if err := s.emitError(string(normalizeAnthropicErrorType(chunk.Error.Type)), chunk.Error.Message); err != nil {
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
		if err := s.applyTextDelta(c.Delta.Content); err != nil {
			return err
		}
		if err := s.applyToolCallDeltas(c.Delta.ToolCalls); err != nil {
			return err
		}
		if c.FinishReason != nil && *c.FinishReason != "" {
			s.stopReason = mapFinishReasonForStream(*c.FinishReason, c.StoppingWord)
			if c.StoppingWord != "" {
				s.stopSequence = c.StoppingWord
			}
		}
	}
	return nil
}

func (s *streamState) applyTextDelta(text string) error {
	if text == "" {
		return nil
	}
	if err := s.ensureTextBlock(); err != nil {
		return err
	}
	return s.writeEvent("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": s.textIndex,
		"delta": map[string]string{"type": "text_delta", "text": text},
	})
}

func (s *streamState) applyToolCallDeltas(deltas []openAIToolCallDelta) error {
	for _, d := range deltas {
		name := ""
		args := ""
		if d.Function != nil {
			name = d.Function.Name
			args = d.Function.Arguments
		}
		idx, err := s.ensureToolBlock(d.Index, d.ID, name)
		if err != nil {
			return err
		}
		if args == "" {
			continue
		}
		if err := s.writeEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": idx,
			"delta": map[string]string{
				"type":         "input_json_delta",
				"partial_json": args,
			},
		}); err != nil {
			return err
		}
	}
	return nil
}

// mapFinishReasonForStream extends mapFinishReason with llama-server's
// stopping_word signal, same shape as the non-streaming path.
func mapFinishReasonForStream(finish, stoppingWord string) string {
	if finish == "stop" && stoppingWord != "" {
		return "stop_sequence"
	}
	return mapFinishReason(finish)
}

func (s *streamState) finalize() error {
	// Ensure a valid message frame even if upstream produced nothing.
	if err := s.ensureMessageStarted(); err != nil {
		return err
	}
	// If neither text nor tool blocks opened, Anthropic still needs at
	// least one content block in the message — emit an empty text block so
	// SDK accumulators don't receive a malformed empty message.
	if len(s.openBlocks) == 0 {
		if err := s.ensureTextBlock(); err != nil {
			return err
		}
	}
	// Stop every open block in allocation order (stable output; SDK
	// accumulators walk by index).
	for i := 0; i < s.nextBlockIndex; i++ {
		if !s.openBlocks[i] {
			continue
		}
		if err := s.writeEvent("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": i,
		}); err != nil {
			return err
		}
		delete(s.openBlocks, i)
	}
	stopReason := s.stopReason
	if stopReason == "" {
		stopReason = "end_turn"
	}
	var stopSequence any
	if s.stopSequence != "" {
		stopSequence = s.stopSequence
	}
	// Per Anthropic's spec, message_delta.usage is the cumulative running
	// total and carries only output_tokens; input_tokens belongs in
	// message_start.message.usage (which is zero today — see known
	// limitations in the package comment).
	if err := s.writeEvent("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": stopSequence,
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
//
// Uses bufio.Reader.ReadBytes instead of bufio.Scanner because Scanner has
// a hard token-length cap (defaults to 64 KiB; even raising it to 1 MiB is
// fragile). A single `data:` line from a reasoning model or a tool-call
// backend can exceed any static limit; silently dropping the rest of the
// stream is the worst outcome. Each line is read as an independent
// allocation, bounded only by Go's memory, and any I/O error that happens
// mid-stream gets surfaced as a synthetic Anthropic error event instead of
// a silent truncation.
// streamPingInterval is how often we emit an Anthropic `event: ping` while
// waiting on a slow model. Many L7 proxies (nginx default 60s, corporate
// SSL terminators) kill idle SSE connections — without pings, a 70B model
// generating at a few tok/s can produce back-to-back chunks separated by
// enough idle time to trip the timeout. 15s is well below every common
// threshold we've seen.
// streamPingInterval is how often we emit an Anthropic `event: ping`
// while waiting on a slow model. var rather than const so tests can
// override it without sleeping for 15 seconds.
var streamPingInterval = 15 * time.Second

func translateAnthropicStream(w io.Writer, upstream io.Reader, messageID, model string) error {
	state := newStreamState(w, messageID, model)
	stopPings := state.startPings(streamPingInterval)
	defer stopPings()
	reader := bufio.NewReader(upstream)

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if stop, handleErr := handleStreamLine(state, line); handleErr != nil {
				return handleErr
			} else if stop {
				return nil
			}
		}
		if err == io.EOF {
			return state.finalize()
		}
		if err != nil {
			// Mid-stream I/O error: emit an error frame so SDK consumers see
			// an explicit failure instead of a stream that just stops.
			_ = state.emitError("api_error", fmt.Sprintf("upstream stream error: %v", err))
			return err
		}
	}
}

// handleStreamLine processes one SSE line. Returns (stop=true, nil) when
// an upstream error event already emitted an Anthropic error frame and the
// caller should return cleanly without tacking on trailing message_delta /
// message_stop frames.
func handleStreamLine(state *streamState, line []byte) (bool, error) {
	trimmed := bytes.TrimRight(line, "\r\n")
	if !bytes.HasPrefix(trimmed, []byte("data:")) {
		return false, nil
	}
	payload := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
		return false, nil
	}
	var chunk openAIStreamChunk
	if err := json.Unmarshal(payload, &chunk); err != nil {
		// Skip malformed chunks rather than aborting the whole stream.
		return false, nil
	}
	if err := state.applyChunk(&chunk); err != nil {
		if errors.Is(err, errStreamAborted) {
			return true, nil
		}
		return false, err
	}
	return false, nil
}

// --- Token counting ---

// countAnthropicTokens returns an approximate token count for an Anthropic
// /v1/messages/count_tokens request body. Anthropic requires this endpoint
// but llama-server and SwiftLM don't expose an OpenAI-equivalent that works
// without loading the model, so we approximate.
//
// The estimate is deliberately pessimistic:
//   - ~2.5 chars/token covers the common adversarial cases (whitespace-
//     heavy code, JSON, CJK), where real BPE tokenizers run well below the
//     English-prose 3.5–4.0 ratio.
//   - perMessageOverhead captures the chat-template framing tokens (role
//     markers, separators, turn boundaries) that don't show up in content
//     strings but are real context-window consumption.
//   - Tool schemas contribute too — they're serialized to JSON and fed
//     into the system prompt by most chat templates.
//
// This biases toward overcount. That's the correct direction: aider and
// other planners drop/summarize history based on this number; an
// undercount leads to mid-stream context-window overruns, which are
// user-visible crashes.
func countAnthropicTokens(body []byte) (int, error) {
	var req anthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return 0, newBadRequest("Failed to parse request body as JSON")
	}

	var total int
	if len(req.System) > 0 {
		sys, err := flattenAnthropicSystem(req.System)
		if err != nil {
			return 0, err
		}
		total += approxTokens(sys)
	}
	for _, t := range req.Tools {
		// Approximate a tool descriptor by its JSON footprint — name,
		// description, and schema all get serialized into the chat template.
		b, _ := json.Marshal(t)
		total += approxTokens(string(b))
	}
	for _, m := range req.Messages {
		total += perMessageOverhead
		total += approxMessageTokens(m)
	}
	return total, nil
}

// approxMessageTokens sums the token footprint of every block in one
// message. Text and tool_use inputs count as JSON strings; tool_result
// content flattens via the same path as translation. Image blocks
// intentionally skip the estimator — per-image Anthropic token costs are
// model-dependent and callers sending images should not rely on
// count_tokens for budgeting.
func approxMessageTokens(m anthropicMessage) int {
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return approxTokens(s)
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		return 0
	}
	var total int
	for _, b := range blocks {
		switch b.Type {
		case "text":
			total += approxTokens(b.Text)
		case "tool_use":
			total += approxTokens(b.Name) + approxTokens(string(b.Input))
		case "tool_result":
			if s, err := flattenToolResultContent(b.ToolContent); err == nil {
				total += approxTokens(s)
			}
		}
	}
	return total
}

const (
	// charsPerToken is the inverse of our chars/token ratio — 2.5 → 2.
	// Kept as an integer so the rounding stays deterministic.
	charsPerToken = 5
	// perMessageOverhead is the per-turn token cost of the chat template
	// framing (role markers, BOS/EOS, separators). Conservative; real
	// overhead varies 4–8 tokens across common templates.
	perMessageOverhead = 8
)

// approxTokens estimates tokens for a plain text string using ceil(len*2 / 5).
// 2.5 chars/token leans pessimistic enough to cover non-English / code /
// JSON inputs where modern BPE tokenizers compress poorly.
func approxTokens(s string) int {
	if s == "" {
		return 0
	}
	// ceil(len * 2 / 5)
	return (len(s)*2 + charsPerToken - 1) / charsPerToken
}
