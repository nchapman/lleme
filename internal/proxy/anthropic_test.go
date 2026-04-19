package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestTranslateAnthropicRequest_StringContent(t *testing.T) {
	in := `{
		"model": "some/model:Q4",
		"max_tokens": 64,
		"messages": [
			{"role": "user", "content": "hi"}
		]
	}`

	out, stream, err := translateAnthropicRequest([]byte(in))
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if stream {
		t.Errorf("stream = true, want false")
	}

	var got openAIChatRequest
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("decode openai req: %v", err)
	}
	if got.Model != "some/model:Q4" {
		t.Errorf("model = %q, want some/model:Q4", got.Model)
	}
	if got.MaxTokens != 64 {
		t.Errorf("max_tokens = %d, want 64", got.MaxTokens)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Fatalf("messages = %+v, want 1 user message", got.Messages)
	}

	var contentStr string
	if err := json.Unmarshal(got.Messages[0].Content, &contentStr); err != nil {
		t.Fatalf("content should be string: %v", err)
	}
	if contentStr != "hi" {
		t.Errorf("content = %q, want hi", contentStr)
	}
}

func TestTranslateAnthropicRequest_SystemAndSamplingParams(t *testing.T) {
	temp := 0.4
	topP := 0.9
	topK := 40
	in := anthropicMessagesRequest{
		Model:         "m",
		MaxTokens:     32,
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"</s>", "<|endoftext|>"},
		Stream:        true,
		System:        json.RawMessage(`"Be brief."`),
		Messages: []anthropicMessage{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}
	body, _ := json.Marshal(in)

	out, stream, err := translateAnthropicRequest(body)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if !stream {
		t.Errorf("stream = false, want true")
	}

	var got openAIChatRequest
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Temperature == nil || *got.Temperature != 0.4 {
		t.Errorf("temperature = %v, want 0.4", got.Temperature)
	}
	if got.TopP == nil || *got.TopP != 0.9 {
		t.Errorf("top_p = %v, want 0.9", got.TopP)
	}
	if got.TopK == nil || *got.TopK != 40 {
		t.Errorf("top_k = %v, want 40", got.TopK)
	}
	if len(got.Stop) != 2 || got.Stop[0] != "</s>" {
		t.Errorf("stop = %v, want [</s>, <|endoftext|>]", got.Stop)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("got %d messages, want 2 (system + user)", len(got.Messages))
	}
	if got.Messages[0].Role != "system" {
		t.Errorf("first message role = %q, want system", got.Messages[0].Role)
	}
	var sys string
	_ = json.Unmarshal(got.Messages[0].Content, &sys)
	if sys != "Be brief." {
		t.Errorf("system content = %q, want Be brief.", sys)
	}
}

func TestTranslateAnthropicRequest_SystemArrayFlattened(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"system": [
			{"type": "text", "text": "Part one."},
			{"type": "text", "text": "Part two."}
		],
		"messages": [{"role": "user", "content": "q"}]
	}`)

	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got openAIChatRequest
	_ = json.Unmarshal(out, &got)
	var sys string
	_ = json.Unmarshal(got.Messages[0].Content, &sys)
	if sys != "Part one.\n\nPart two." {
		t.Errorf("system = %q, want joined parts", sys)
	}
}

func TestTranslateAnthropicRequest_ToolUseAssistantContentRejected(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"messages": [{
			"role": "assistant",
			"content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]
		}]
	}`)

	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400 translateError", err)
	}
}

func TestTranslateAnthropicRequest_MaxTokensRequired(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400 translateError for missing max_tokens", err)
	}
}

func TestTranslateAnthropicRequest_ToolChoiceNullIsAccepted(t *testing.T) {
	// A literal null for tools/tool_choice must not be treated as "present".
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"tool_choice": null,
		"tools": null,
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	if _, _, err := translateAnthropicRequest(in); err != nil {
		t.Fatalf("null tool_choice/tools should be accepted, got: %v", err)
	}
}

func TestTranslateAnthropicRequest_StopSequencesCapped(t *testing.T) {
	in := anthropicMessagesRequest{
		Model:         "m",
		MaxTokens:     5,
		StopSequences: []string{"a", "b", "c", "d", "e"},
		Messages:      []anthropicMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	body, _ := json.Marshal(in)
	_, _, err := translateAnthropicRequest(body)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400 for too many stop_sequences", err)
	}
}

func TestTranslateAnthropicRequest_DisallowedImageMediaType(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "image", "source": {"type": "base64", "media_type": "image/bmp", "data": "AAAA"}}
			]
		}]
	}`)
	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400 for disallowed media_type", err)
	}
}

func TestTranslateAnthropicRequest_StreamingAsksForUsage(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"stream": true,
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got openAIChatRequest
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.StreamOptions == nil || !got.StreamOptions.IncludeUsage {
		t.Errorf("stream_options.include_usage not set; got %+v", got.StreamOptions)
	}
}

func TestTranslateAnthropicRequest_ContentBlocksTextAndImage(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "describe"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
			]
		}]
	}`)

	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got openAIChatRequest
	_ = json.Unmarshal(out, &got)

	var parts []openAIContentPart
	if err := json.Unmarshal(got.Messages[0].Content, &parts); err != nil {
		t.Fatalf("content should be array: %v", err)
	}
	if len(parts) != 2 {
		t.Fatalf("got %d parts, want 2", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "describe" {
		t.Errorf("parts[0] = %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Fatalf("parts[1] = %+v", parts[1])
	}
	if parts[1].ImageURL.URL != "data:image/png;base64,AAAA" {
		t.Errorf("image url = %q", parts[1].ImageURL.URL)
	}
}

func TestTranslateAnthropicRequest_ToolsRejected(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"tools": [{"name": "get_weather"}],
		"messages": [{"role": "user", "content": "hi"}]
	}`)

	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400 translateError", err)
	}
	if !strings.Contains(te.msg, "tools") {
		t.Errorf("error message = %q, should mention tools", te.msg)
	}
}

func TestTranslateAnthropicRequest_ToolResultContentRejected(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"messages": [{
			"role": "user",
			"content": [{"type": "tool_result", "tool_use_id": "x", "content": "42"}]
		}]
	}`)

	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400 translateError", err)
	}
}

func TestTranslateOpenAIResponse_Basic(t *testing.T) {
	openAI := `{
		"id": "chatcmpl-xyz",
		"model": "foo/bar:Q4",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Hello world"},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
	}`

	out, err := translateOpenAIResponse([]byte(openAI), "msg_abc")
	if err != nil {
		t.Fatalf("translate: %v", err)
	}

	var got anthropicMessagesResponse
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.ID != "msg_abc" {
		t.Errorf("id = %q, want msg_abc", got.ID)
	}
	if got.Type != "message" || got.Role != "assistant" {
		t.Errorf("type/role = %q/%q", got.Type, got.Role)
	}
	if got.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", got.StopReason)
	}
	if len(got.Content) != 1 || got.Content[0].Text != "Hello world" {
		t.Errorf("content = %+v", got.Content)
	}
	if got.Usage.InputTokens != 12 || got.Usage.OutputTokens != 3 {
		t.Errorf("usage = %+v", got.Usage)
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := map[string]string{
		"stop":           "end_turn",
		"length":         "max_tokens",
		"tool_calls":     "tool_use",
		"function_call":  "tool_use",
		"content_filter": "end_turn",
		"":               "",
		"unexpected":     "end_turn",
	}
	for in, want := range cases {
		if got := mapFinishReason(in); got != want {
			t.Errorf("mapFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestTranslateAnthropicStream_HappyPath(t *testing.T) {
	// Two deltas, then a finish chunk with usage.
	upstream := strings.Join([]string{
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"Hel"}}]}`,
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		`data: [DONE]`,
		"",
	}, "\n\n")

	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg_1", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}

	got := buf.String()
	wantEvents := []string{
		"event: message_start",
		"event: content_block_start",
		"event: content_block_delta",
		`"text":"Hel"`,
		`"text":"lo"`,
		"event: content_block_stop",
		"event: message_delta",
		`"stop_reason":"end_turn"`,
		// Per Anthropic spec, message_delta.usage carries only output_tokens.
		`"output_tokens":2`,
		"event: message_stop",
	}
	for _, w := range wantEvents {
		if !strings.Contains(got, w) {
			t.Errorf("stream output missing %q\n--- output ---\n%s", w, got)
		}
	}

	// message_start must precede any content_block_delta.
	startIdx := strings.Index(got, "message_start")
	deltaIdx := strings.Index(got, "content_block_delta")
	if startIdx < 0 || deltaIdx < 0 || startIdx > deltaIdx {
		t.Errorf("message_start must come before content_block_delta (start=%d, delta=%d)", startIdx, deltaIdx)
	}
}

func TestTranslateAnthropicStream_EmptyUpstreamStillEmitsFrame(t *testing.T) {
	// Backend closed without sending anything — we must still emit a valid
	// Anthropic frame so the client doesn't hang.
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(""), "msg_x", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	got := buf.String()
	for _, ev := range []string{"message_start", "content_block_stop", "message_delta", "message_stop"} {
		if !strings.Contains(got, "event: "+ev) {
			t.Errorf("missing %q in output:\n%s", ev, got)
		}
	}
}

func TestTranslateAnthropicStream_MappedFinishReason(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"length"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if !strings.Contains(buf.String(), `"stop_reason":"max_tokens"`) {
		t.Errorf("expected max_tokens stop_reason, got:\n%s", buf.String())
	}
}

func TestCountAnthropicTokens(t *testing.T) {
	// A 7-char string should yield ceil(7/3.5) = 2 tokens.
	body := []byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "abcdefg"}]
	}`)
	n, err := countAnthropicTokens(body)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 2 {
		t.Errorf("count = %d, want 2", n)
	}
}

func TestCountAnthropicTokens_IncludesSystemAndBlocks(t *testing.T) {
	body := []byte(`{
		"model": "m",
		"system": "abcdefg",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "hijklmn"},
				{"type": "text", "text": "opqrstu"}
			]
		}]
	}`)
	// 3 segments × 7 chars → ceil(7/3.5)=2 tokens each → 6 total.
	n, err := countAnthropicTokens(body)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 6 {
		t.Errorf("count = %d, want 6", n)
	}
}

// --- Streaming spec-shape lock-ins ---

// Decodes the SSE stream output into a slice of {event, data} records so
// tests can assert shape at the JSON-object level instead of substring
// hunting. The data string still holds the raw JSON payload.
type sseEvent struct {
	name string
	data string
}

func decodeSSE(t *testing.T, raw string) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, block := range strings.Split(raw, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var ev sseEvent
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				ev.name = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				ev.data = strings.TrimPrefix(line, "data: ")
			}
		}
		if ev.name != "" {
			events = append(events, ev)
		}
	}
	return events
}

func findEvent(events []sseEvent, name string) *sseEvent {
	for i := range events {
		if events[i].name == name {
			return &events[i]
		}
	}
	return nil
}

// content_block_start must carry the documented Anthropic shape:
// {"type":"text","text":""}. The empty text key is required — Python/TS
// SDK accumulators construct a TextBlock from this event and assume it.
func TestTranslateAnthropicStream_ContentBlockStartShape(t *testing.T) {
	upstream := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n"
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	ev := findEvent(decodeSSE(t, buf.String()), "content_block_start")
	if ev == nil {
		t.Fatal("no content_block_start emitted")
	}
	var payload struct {
		Type         string `json:"type"`
		Index        int    `json:"index"`
		ContentBlock struct {
			Type string  `json:"type"`
			Text *string `json:"text"` // pointer so we can tell "missing" from ""
		} `json:"content_block"`
	}
	if err := json.Unmarshal([]byte(ev.data), &payload); err != nil {
		t.Fatalf("decode: %v\ndata=%s", err, ev.data)
	}
	if payload.ContentBlock.Type != "text" {
		t.Errorf("content_block.type = %q, want text", payload.ContentBlock.Type)
	}
	if payload.ContentBlock.Text == nil {
		t.Error(`content_block.text missing; Anthropic spec requires text:""`)
	} else if *payload.ContentBlock.Text != "" {
		t.Errorf(`content_block.text = %q, want ""`, *payload.ContentBlock.Text)
	}
}

// Per spec, message_delta.usage carries only output_tokens (cumulative).
// input_tokens belongs in message_start and must NOT appear here.
func TestTranslateAnthropicStream_MessageDeltaUsageShape(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	ev := findEvent(decodeSSE(t, buf.String()), "message_delta")
	if ev == nil {
		t.Fatal("no message_delta emitted")
	}
	var payload struct {
		Usage map[string]any `json:"usage"`
	}
	if err := json.Unmarshal([]byte(ev.data), &payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, present := payload.Usage["input_tokens"]; present {
		t.Errorf("message_delta.usage must not contain input_tokens; got %v", payload.Usage)
	}
	if payload.Usage["output_tokens"] != float64(1) {
		t.Errorf("output_tokens = %v, want 1", payload.Usage["output_tokens"])
	}
}

// message_start always carries usage with input_tokens — we emit 0 as a
// known limitation (prompt_tokens aren't known until the final usage chunk,
// too late to backfill without buffering the whole stream).
func TestTranslateAnthropicStream_MessageStartUsageZero(t *testing.T) {
	upstream := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n"
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	ev := findEvent(decodeSSE(t, buf.String()), "message_start")
	if ev == nil {
		t.Fatal("no message_start emitted")
	}
	var payload struct {
		Message struct {
			ID           string  `json:"id"`
			Type         string  `json:"type"`
			Role         string  `json:"role"`
			Model        string  `json:"model"`
			StopReason   *string `json:"stop_reason"`
			StopSequence *string `json:"stop_sequence"`
			Usage        struct {
				InputTokens  *int `json:"input_tokens"`
				OutputTokens *int `json:"output_tokens"`
			} `json:"usage"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(ev.data), &payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if payload.Message.ID != "msg" || payload.Message.Type != "message" || payload.Message.Role != "assistant" {
		t.Errorf("message_start skeleton wrong: %+v", payload.Message)
	}
	if payload.Message.Usage.InputTokens == nil {
		t.Fatal("message_start.usage.input_tokens missing; spec requires the key present")
	}
	// Currently 0 by design (see package-level known limitations). If a
	// buffering strategy lands, this assertion is the signal to update the
	// docs and the test together — it's the whole point of the lock-in.
	if *payload.Message.Usage.InputTokens != 0 {
		t.Errorf("message_start.usage.input_tokens = %d; documented as 0 — update docs + this test if buffering intentionally landed",
			*payload.Message.Usage.InputTokens)
	}
}

// An upstream chunk that carries both `content` and `finish_reason` on the
// same choice must still emit the content delta AND the final stop_reason.
func TestTranslateAnthropicStream_ContentAndFinishOnSameChunk(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"bye"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	events := decodeSSE(t, buf.String())
	if findEvent(events, "content_block_delta") == nil {
		t.Error("content_block_delta missing when content arrived alongside finish_reason")
	}
	md := findEvent(events, "message_delta")
	if md == nil || !strings.Contains(md.data, `"stop_reason":"end_turn"`) {
		t.Errorf("message_delta.stop_reason not end_turn: %+v", md)
	}
}

// Malformed or adversarial upstream: two finish chunks. Must emit exactly
// one message_stop and one message_delta; state must not double-emit.
func TestTranslateAnthropicStream_MultipleFinishChunks(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":"stop"}]}`,
		`data: {"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":"length"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	events := decodeSSE(t, buf.String())
	countEv := func(name string) int {
		n := 0
		for _, e := range events {
			if e.name == name {
				n++
			}
		}
		return n
	}
	if got := countEv("message_stop"); got != 1 {
		t.Errorf("message_stop count = %d, want 1", got)
	}
	if got := countEv("message_delta"); got != 1 {
		t.Errorf("message_delta count = %d, want 1", got)
	}
	// Which stop_reason wins is unspecified for this pathological input —
	// only the frame-count invariant matters. We intentionally don't pin
	// first-vs-last here.
}

// --- Request-side: forward-compat and rejection coverage ---

// A trailing assistant message (prefill) is a legitimate Anthropic pattern;
// the translator must pass it through so the backend can continue from it.
func TestTranslateAnthropicRequest_PrefillAssistantPassedThrough(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 20,
		"messages": [
			{"role": "user", "content": "begin"},
			{"role": "assistant", "content": "ok, here we go:"}
		]
	}`)
	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got openAIChatRequest
	_ = json.Unmarshal(out, &got)
	if len(got.Messages) != 2 {
		t.Fatalf("got %d messages, want 2", len(got.Messages))
	}
	if got.Messages[1].Role != "assistant" {
		t.Errorf("second message role = %q, want assistant (prefill)", got.Messages[1].Role)
	}
}

// Prompt-caching fields (cache_control on system blocks) must not cause a 400.
// We silently ignore them; clients that always attach them still work.
func TestTranslateAnthropicRequest_CacheControlOnSystemIgnored(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"system": [
			{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
		],
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	if _, _, err := translateAnthropicRequest(in); err != nil {
		t.Fatalf("cache_control on system should be ignored, got: %v", err)
	}
}

// cache_control on a text content block must not cause a 400.
func TestTranslateAnthropicRequest_CacheControlOnContentIgnored(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "cached bit", "cache_control": {"type": "ephemeral"}}
			]
		}]
	}`)
	if _, _, err := translateAnthropicRequest(in); err != nil {
		t.Fatalf("cache_control on content should be ignored, got: %v", err)
	}
}

// Unknown top-level fields (service_tier, metadata, thinking, container)
// must be silently ignored for forward-compat with newer API shapes.
func TestTranslateAnthropicRequest_UnknownTopLevelFieldsIgnored(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"service_tier": "standard_only",
		"metadata": {"user_id": "u_123"},
		"container": "c_abc",
		"thinking": {"type": "enabled"},
		"cache_control": {"type": "ephemeral"},
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	if _, _, err := translateAnthropicRequest(in); err != nil {
		t.Fatalf("unknown top-level fields should be ignored, got: %v", err)
	}
}

// Document content blocks aren't supported by our translator (backend can't
// consume them anyway). Reject with a clear 400 mentioning the type.
func TestTranslateAnthropicRequest_DocumentBlockRejected(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AAA="}}
			]
		}]
	}`)
	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400", err)
	}
	if !strings.Contains(te.msg, "document") {
		t.Errorf("error should mention block type; got %q", te.msg)
	}
}

// Extended-thinking blocks and redacted_thinking blocks are not translated.
func TestTranslateAnthropicRequest_ThinkingBlockRejected(t *testing.T) {
	for _, blockType := range []string{"thinking", "redacted_thinking"} {
		t.Run(blockType, func(t *testing.T) {
			body := []byte(`{
				"model": "m",
				"max_tokens": 5,
				"messages": [{
					"role": "assistant",
					"content": [{"type": "` + blockType + `", "thinking": "..."}]
				}]
			}`)
			_, _, err := translateAnthropicRequest(body)
			var te *translateError
			if !errors.As(err, &te) || te.status != 400 {
				t.Fatalf("got err %v, want 400", err)
			}
			// Make sure we fail for the right reason — not an unrelated
			// 400 that happens to match status.
			if !strings.Contains(te.msg, blockType) {
				t.Errorf("error message should mention %q; got %q", blockType, te.msg)
			}
		})
	}
}

// Anthropic supports `source.type: "url"` for images (added 2024); llama.cpp
// passes the URL through verbatim. We do the same — no download, no
// validation; let the backend fetch.
func TestTranslateAnthropicRequest_ImageURLSourcePassthrough(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
			]
		}]
	}`)
	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("url image should be accepted, got: %v", err)
	}
	var got openAIChatRequest
	_ = json.Unmarshal(out, &got)
	var parts []openAIContentPart
	_ = json.Unmarshal(got.Messages[0].Content, &parts)
	if len(parts) != 1 || parts[0].ImageURL == nil {
		t.Fatalf("expected one image_url part; got %+v", parts)
	}
	if parts[0].ImageURL.URL != "https://example.com/x.png" {
		t.Errorf("image url = %q, want passthrough", parts[0].ImageURL.URL)
	}
}

// A url-source block without a url should still produce a clear 400.
func TestTranslateAnthropicRequest_ImageURLSourceMissingURL(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "image", "source": {"type": "url"}}
			]
		}]
	}`)
	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400", err)
	}
}

// tool_choice without tools should still be rejected (consistency — the
// path is conceptually tool-related either way).
func TestTranslateAnthropicRequest_ToolChoiceWithoutToolsRejected(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"tool_choice": {"type": "auto"},
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400", err)
	}
}

// System array containing an image (or any non-text) block must be rejected;
// Anthropic system is text-only.
func TestTranslateAnthropicRequest_SystemWithNonTextBlockRejected(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"system": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA="}}],
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400", err)
	}
}

// Content-block ordering is significant (image before text, text before
// image) — verify the translator preserves the order the client sent.
func TestTranslateAnthropicRequest_ContentBlockOrderPreserved(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA="}},
				{"type": "text", "text": "what is this?"}
			]
		}]
	}`)
	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got openAIChatRequest
	_ = json.Unmarshal(out, &got)
	var parts []openAIContentPart
	_ = json.Unmarshal(got.Messages[0].Content, &parts)
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2", len(parts))
	}
	if parts[0].Type != "image_url" || parts[1].Type != "text" {
		t.Errorf("order not preserved: %+v", parts)
	}
}

// --- Response: multi-choice and error shape ---

// Only the first OpenAI choice should populate the Anthropic response;
// subsequent choices must be dropped rather than concatenated.
func TestTranslateOpenAIResponse_MultipleChoicesTakesFirst(t *testing.T) {
	openAI := `{
		"id": "c",
		"model": "m",
		"choices": [
			{"index": 0, "message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"},
			{"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"}
		],
		"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
	}`
	out, err := translateOpenAIResponse([]byte(openAI), "msg_1")
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got anthropicMessagesResponse
	_ = json.Unmarshal(out, &got)
	if len(got.Content) != 1 || got.Content[0].Text != "first" {
		t.Errorf("response should carry only first choice; got %+v", got.Content)
	}
	// Explicitly guard against a future "concat choices" refactor.
	if strings.Contains(string(out), "second") {
		t.Errorf("response leaked choice[1] text; got: %s", string(out))
	}
}

// AnthropicError marshalling: the top-level request_id field is what clients
// correlate with. Ensure it survives serialization.
func TestAnthropicError_RequestIDInBody(t *testing.T) {
	e := AnthropicError{
		Type:      "error",
		RequestID: "req_abc123",
		Error: AnthropicErrorDetail{
			Type:    AnthropicInvalidRequest,
			Message: "oops",
		},
	}
	raw, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["request_id"] != "req_abc123" {
		t.Errorf("request_id not present in marshalled error body; got %+v", got)
	}
	errMap, ok := got["error"].(map[string]any)
	if !ok || errMap["type"] != string(AnthropicInvalidRequest) {
		t.Errorf("error.type wrong: %+v", got["error"])
	}
}

// Empty messages array: we don't auto-reject (let the backend complain if
// it cares). Lock in this behavior so a change is deliberate.
func TestTranslateAnthropicRequest_EmptyMessagesAccepted(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 5,
		"messages": []
	}`)
	if _, _, err := translateAnthropicRequest(in); err != nil {
		t.Fatalf("empty messages should translate without error; got: %v", err)
	}
}

// count_tokens must reject tools for consistency with /v1/messages — and
// because tool schemas contribute substantial tokens that our char-ratio
// approximation would silently undercount, leading to context overflows.
func TestCountAnthropicTokens_ToolsRejected(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"tools": [{"name": "f", "description": "d", "input_schema": {}}],
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	_, err := countAnthropicTokens(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400 for tools in count_tokens", err)
	}
}

// Upstream error chunks are translated to Anthropic `event: error` frames.
// After that, no further content or message_* frames should be emitted.
func TestTranslateAnthropicStream_UpstreamErrorBecomesErrorEvent(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		`data: {"error":{"message":"backend exploded","type":"api_error"}}`,
		// A spurious trailing chunk after the error must not produce output.
		`data: {"choices":[{"index":0,"delta":{"content":"never"}}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")

	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	events := decodeSSE(t, buf.String())

	errEv := findEvent(events, "error")
	if errEv == nil {
		t.Fatal("expected event: error to be emitted on upstream error")
	}
	if !strings.Contains(errEv.data, "backend exploded") {
		t.Errorf("error event missing upstream message; got: %s", errEv.data)
	}

	// After the error, we must not emit message_delta/message_stop — that
	// would confuse clients about the final state.
	for _, name := range []string{"message_delta", "message_stop"} {
		if findEvent(events, name) != nil {
			t.Errorf("%s must not follow an error event", name)
		}
	}
	// Content from after the error chunk must not be present.
	if strings.Contains(buf.String(), "never") {
		t.Errorf("content after error should not be forwarded")
	}
}

// Event ordering: content_block_start must precede the first
// content_block_delta. The ensureStarted guard is load-bearing; a
// regression here would break every strict SDK accumulator.
func TestTranslateAnthropicStream_EventOrdering(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"A"}}]}`,
		`data: {"choices":[{"index":0,"delta":{"content":"B"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	events := decodeSSE(t, buf.String())

	indexOf := func(name string) int {
		for i, e := range events {
			if e.name == name {
				return i
			}
		}
		return -1
	}
	var (
		iStart = indexOf("message_start")
		iCBS   = indexOf("content_block_start")
		iCBD   = indexOf("content_block_delta")
		iCBE   = indexOf("content_block_stop")
		iMD    = indexOf("message_delta")
		iMS    = indexOf("message_stop")
	)
	if iStart < 0 || iCBS < 0 || iCBD < 0 || iCBE < 0 || iMD < 0 || iMS < 0 {
		t.Fatalf("missing event; got order: %+v", events)
	}
	// Canonical Anthropic order.
	if iStart >= iCBS || iCBS >= iCBD || iCBD >= iCBE || iCBE >= iMD || iMD >= iMS {
		t.Errorf("out-of-order events: start=%d cbs=%d cbd=%d cbe=%d md=%d ms=%d", iStart, iCBS, iCBD, iCBE, iMD, iMS)
	}
}

// stop_sequence must serialize as null (not omitted) on non-streaming
// responses. We never know which sequence matched — see known limitations
// — so emitting null is the honest, spec-compliant signal.
func TestTranslateOpenAIResponse_StopSequenceNullPresent(t *testing.T) {
	openAI := `{
		"id": "c",
		"model": "m",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
	}`
	out, err := translateOpenAIResponse([]byte(openAI), "msg_x")
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	// Substring check — we want the literal `"stop_sequence":null` key to
	// be present, not omitted. A struct decode would silently hide omission.
	if !strings.Contains(string(out), `"stop_sequence":null`) {
		t.Errorf(`response should include "stop_sequence":null key; got %s`, string(out))
	}
}
