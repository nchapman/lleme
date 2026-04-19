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
		`"input_tokens":5`,
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
