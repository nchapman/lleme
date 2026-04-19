package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
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

// Assistant tool_use blocks translate to an OpenAI assistant message with
// tool_calls — Anthropic's {id, name, input} maps 1:1 to OpenAI's
// {id, type:"function", function:{name, arguments}}.
func TestTranslateAnthropicRequest_AssistantToolUseTranslated(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"messages": [{
			"role": "assistant",
			"content": [
				{"type": "text", "text": "thinking..."},
				{"type": "tool_use", "id": "t1", "name": "fetch", "input": {"url": "https://x"}}
			]
		}]
	}`)

	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got struct {
		Messages []struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name, Arguments string
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(got.Messages))
	}
	m := got.Messages[0]
	if m.Role != "assistant" || m.Content != "thinking..." {
		t.Errorf("role/content: %q %q", m.Role, m.Content)
	}
	if len(m.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %d, want 1", len(m.ToolCalls))
	}
	tc := m.ToolCalls[0]
	if tc.ID != "t1" || tc.Type != "function" || tc.Function.Name != "fetch" {
		t.Errorf("tool call shape: %+v", tc)
	}
	if !strings.Contains(tc.Function.Arguments, `"url"`) {
		t.Errorf("arguments lost input: %q", tc.Function.Arguments)
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

// Tools are translated into OpenAI's tools schema: Anthropic's
// {name, description, input_schema} becomes {type:"function", function:
// {name, description, parameters}}. Clients like Claude Code / aider
// / cline rely on this; without it they can't use the proxy at all.
func TestTranslateAnthropicRequest_ToolsTranslated(t *testing.T) {
	in := []byte(`{
		"model": "m",
		"max_tokens": 10,
		"tools": [{
			"name": "get_weather",
			"description": "Return the weather",
			"input_schema": {"type": "object", "properties": {"loc": {"type": "string"}}}
		}],
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name, Description string
				Parameters        map[string]any
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(got.Tools))
	}
	tool := got.Tools[0]
	if tool.Type != "function" || tool.Function.Name != "get_weather" {
		t.Errorf("tool shape: %+v", tool)
	}
	if tool.Function.Description == "" {
		t.Errorf("description dropped")
	}
	if tool.Function.Parameters["type"] != "object" {
		t.Errorf("parameters lost: %+v", tool.Function.Parameters)
	}
}

// Each tool_choice shape must land on the correct OpenAI form so the
// backend sees the same intent. Pins the 1:1 mapping — a client sending
// {type:"any"} expects forced tool use and we must translate to "required",
// not drop it.
func TestTranslateAnthropicRequest_ToolChoiceShapes(t *testing.T) {
	base := func(tc string) []byte {
		return []byte(`{
			"model": "m", "max_tokens": 10,
			"tools": [{"name": "f", "input_schema": {}}],
			"tool_choice": ` + tc + `,
			"messages": [{"role": "user", "content": "hi"}]
		}`)
	}
	// Compared after re-parsing so map key order doesn't matter.
	decode := func(raw string) any {
		var v any
		if err := json.Unmarshal([]byte(raw), &v); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		return v
	}
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"string auto", `"auto"`, `"auto"`},
		{"object auto", `{"type":"auto"}`, `"auto"`},
		{"object any -> required", `{"type":"any"}`, `"required"`},
		{"object none", `{"type":"none"}`, `"none"`},
		{"object tool with name", `{"type":"tool","name":"f"}`, `{"type":"function","function":{"name":"f"}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, _, err := translateAnthropicRequest(base(tc.in))
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			var got struct {
				ToolChoice json.RawMessage `json:"tool_choice"`
			}
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(decode(string(got.ToolChoice)), decode(tc.want)) {
				t.Errorf("tool_choice = %s, want %s", got.ToolChoice, tc.want)
			}
		})
	}
}

// Empty tools/tool_choice are treated as absent — a client attaching
// `tools: []` unconditionally (common Go/Python SDK wrapper pattern) must
// not be rejected or forced into tool mode.
func TestTranslateAnthropicRequest_EmptyToolsAreAbsent(t *testing.T) {
	in := []byte(`{
		"model": "m", "max_tokens": 10,
		"tools": [], "tool_choice": {},
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if bytes.Contains(out, []byte(`"tools"`)) {
		t.Error("empty tools should not appear in translated request")
	}
	if bytes.Contains(out, []byte(`"tool_choice"`)) {
		t.Error("empty tool_choice should not appear in translated request")
	}
}

// tool_result content blocks translate to role:"tool" OpenAI messages;
// this is the end-half of the Claude Code tool loop. Multiple tool_result
// blocks in one user turn fan out to one OpenAI message each.
func TestTranslateAnthropicRequest_ToolResultTranslated(t *testing.T) {
	in := []byte(`{
		"model": "m", "max_tokens": 10,
		"messages": [
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "a", "name": "f", "input": {}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "a", "content": "42"},
				{"type": "text", "text": "one more thing"}
			]}
		]
	}`)
	out, _, err := translateAnthropicRequest(in)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var got struct {
		Messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			ToolCallID string          `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.Messages) != 3 {
		t.Fatalf("messages = %d, want 3 (assistant + tool + user)", len(got.Messages))
	}
	if got.Messages[1].Role != "tool" || got.Messages[1].ToolCallID != "a" {
		t.Errorf("second message: role=%s id=%s, want tool/a", got.Messages[1].Role, got.Messages[1].ToolCallID)
	}
	if !bytes.Contains(got.Messages[1].Content, []byte("42")) {
		t.Errorf("tool content missing: %s", got.Messages[1].Content)
	}
	if got.Messages[2].Role != "user" {
		t.Errorf("third message: role=%s, want user", got.Messages[2].Role)
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

// When llama-server reports a stopping_word on a finish_reason=="stop",
// translate to Anthropic's stop_reason="stop_sequence" with the matched
// string carried on stop_sequence. Without this, clients that branch on
// "stop_sequence" (e.g. aider-style multi-model agents) never see the
// signal.
func TestTranslateOpenAIResponse_StopSequenceForwarded(t *testing.T) {
	openAI := `{
		"model": "m",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "stop here"},
			"finish_reason": "stop",
			"stopping_word": "\nHuman:"
		}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
	}`
	out, err := translateOpenAIResponse([]byte(openAI), "msg_abc")
	if err != nil {
		t.Fatal(err)
	}
	var got anthropicMessagesResponse
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got.StopReason != "stop_sequence" {
		t.Errorf("stop_reason = %q, want stop_sequence", got.StopReason)
	}
	if got.StopSequence == nil || *got.StopSequence != "\nHuman:" {
		t.Errorf("stop_sequence = %v, want \\nHuman:", got.StopSequence)
	}
}

// An OpenAI response with finish_reason="tool_calls" must produce an
// Anthropic response whose content includes tool_use blocks and whose
// stop_reason is "tool_use" — the contract Claude Code's tool loop walks.
func TestTranslateOpenAIResponse_ToolCalls(t *testing.T) {
	openAI := `{
		"model": "m",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "",
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {"name": "fetch", "arguments": "{\"url\":\"https://x\"}"}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
	}`
	out, err := translateOpenAIResponse([]byte(openAI), "msg_abc")
	if err != nil {
		t.Fatal(err)
	}
	var got anthropicMessagesResponse
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", got.StopReason)
	}
	if len(got.Content) != 1 || got.Content[0].Type != "tool_use" {
		t.Fatalf("content shape: %+v", got.Content)
	}
	if got.Content[0].ID != "call_1" || got.Content[0].Name != "fetch" {
		t.Errorf("tool_use fields: %+v", got.Content[0])
	}
	var input map[string]any
	if err := json.Unmarshal(got.Content[0].Input, &input); err != nil {
		t.Fatalf("input parse: %v", err)
	}
	if input["url"] != "https://x" {
		t.Errorf("input lost arguments: %+v", input)
	}
}

// Streaming: tool_call deltas must produce content_block_start with
// content_block.type="tool_use" and input_json_delta partial_json events
// that SDK accumulators stitch back into a JSON argument object.
func TestTranslateAnthropicStream_ToolCallDeltas(t *testing.T) {
	upstream := strings.Join([]string{
		// First chunk introduces the tool call (id + name)
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"fetch","arguments":""}}]}}]}`,
		// Args arrive in two fragments
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"url\":"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"https://x\"}"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	events := decodeSSE(t, buf.String())

	// message_start present, tool_use block_start present, at least two
	// input_json_delta events, one block_stop, stop_reason tool_use.
	var blockStartFound, blockStopFound bool
	var partialJSON []string
	var stopReason string
	for _, ev := range events {
		switch ev.name {
		case "content_block_start":
			if strings.Contains(ev.data, `"type":"tool_use"`) {
				blockStartFound = true
				if !strings.Contains(ev.data, `"id":"call_1"`) {
					t.Errorf("tool block_start missing id: %s", ev.data)
				}
				if !strings.Contains(ev.data, `"name":"fetch"`) {
					t.Errorf("tool block_start missing name: %s", ev.data)
				}
			}
		case "content_block_delta":
			if strings.Contains(ev.data, `"type":"input_json_delta"`) {
				var payload struct {
					Delta struct {
						Partial string `json:"partial_json"`
					} `json:"delta"`
				}
				_ = json.Unmarshal([]byte(ev.data), &payload)
				partialJSON = append(partialJSON, payload.Delta.Partial)
			}
		case "content_block_stop":
			blockStopFound = true
		case "message_delta":
			if strings.Contains(ev.data, `"stop_reason":"tool_use"`) {
				stopReason = "tool_use"
			}
		}
	}
	if !blockStartFound {
		t.Error("missing tool_use content_block_start")
	}
	if !blockStopFound {
		t.Error("missing content_block_stop")
	}
	if stopReason != "tool_use" {
		t.Error("message_delta did not report stop_reason=tool_use")
	}
	if len(partialJSON) < 2 {
		t.Errorf("expected ≥2 input_json_delta events, got %d: %v", len(partialJSON), partialJSON)
	}
	// The concatenated fragments must be a valid JSON object.
	joined := strings.Join(partialJSON, "")
	var args map[string]any
	if err := json.Unmarshal([]byte(joined), &args); err != nil {
		t.Errorf("concatenated partial_json not valid JSON: %v (got %q)", err, joined)
	}
}

// Upstream error events with a non-Anthropic type must be normalized to the
// api_error enum so SDK consumers don't see a leaked backend-specific type.
func TestTranslateAnthropicStream_ErrorTypeNormalized(t *testing.T) {
	upstream := `data: {"error":{"message":"boom","type":"VendorSpecificError"}}` + "\n\n"
	var buf bytes.Buffer
	if err := translateAnthropicStream(&buf, strings.NewReader(upstream), "msg", "m"); err != nil {
		t.Fatalf("stream: %v", err)
	}
	for _, ev := range decodeSSE(t, buf.String()) {
		if ev.name != "error" {
			continue
		}
		if !strings.Contains(ev.data, `"type":"api_error"`) {
			t.Errorf("error type not normalized: %s", ev.data)
		}
	}
}

// generateMessageID must produce values independent of request IDs and
// the required msg_ prefix. We call it N times and require uniqueness —
// a regression where it derives from the request ID would collide when
// called twice in quick succession with the same request.
func TestGenerateMessageID(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 16; i++ {
		id := generateMessageID()
		if !strings.HasPrefix(id, "msg_") {
			t.Errorf("id missing prefix: %q", id)
		}
		if seen[id] {
			t.Errorf("duplicate id: %q", id)
		}
		seen[id] = true
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

// countAnthropicTokens is deliberately conservative (overcount). We don't
// pin exact numbers because the estimator is tuned for planning safety, not
// parity with a real tokenizer — instead we assert sane ranges plus the
// invariant that additional content never decreases the count.
func TestCountAnthropicTokens(t *testing.T) {
	body := []byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "abcdefg"}]
	}`)
	n, err := countAnthropicTokens(body)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	// 7 chars → at least 3 content tokens, plus per-message overhead.
	// Upper bound guards against a runaway estimator.
	if n < 3 || n > 20 {
		t.Errorf("count = %d, want 3 ≤ n ≤ 20", n)
	}
}

// Adding system content and content blocks must never reduce the estimate.
// Regression guard: the old estimator silently dropped the per-message
// overhead, letting a longer request appear cheaper than a shorter one.
func TestCountAnthropicTokens_MonotonicInContent(t *testing.T) {
	small := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	big := []byte(`{
		"model": "m",
		"system": "you are helpful",
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "hijklmn"},
				{"type": "text", "text": "opqrstu"}
			]},
			{"role": "assistant", "content": "ack"}
		]
	}`)
	nSmall, _ := countAnthropicTokens(small)
	nBig, _ := countAnthropicTokens(big)
	if nBig <= nSmall {
		t.Errorf("big request not larger: small=%d big=%d", nSmall, nBig)
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

// Extended-thinking and redacted_thinking blocks are advisory; we drop
// them from the translated request rather than reject, so an assistant
// turn that carries them (from an earlier round with a reasoning-enabled
// model) still gets through to backends that don't understand them.
func TestTranslateAnthropicRequest_ThinkingBlockDropped(t *testing.T) {
	for _, blockType := range []string{"thinking", "redacted_thinking"} {
		t.Run(blockType, func(t *testing.T) {
			body := []byte(`{
				"model": "m",
				"max_tokens": 5,
				"messages": [{
					"role": "assistant",
					"content": [
						{"type": "` + blockType + `", "thinking": "secret"},
						{"type": "text", "text": "final answer"}
					]
				}]
			}`)
			out, _, err := translateAnthropicRequest(body)
			if err != nil {
				t.Fatalf("thinking block should be dropped, got err: %v", err)
			}
			// The thinking text must not leak into the backend request;
			// only the surrounding text block remains.
			if bytes.Contains(out, []byte("secret")) {
				t.Errorf("thinking content leaked into translated request: %s", out)
			}
			if !bytes.Contains(out, []byte("final answer")) {
				t.Errorf("surrounding text dropped: %s", out)
			}
		})
	}
}

// Anthropic supports `source.type: "url"` for images, but llama-server and
// SwiftLM don't fetch URLs at load time — passing one through would surface
// as an opaque backend error. We reject with a clear actionable message.
func TestTranslateAnthropicRequest_ImageURLSourceRejected(t *testing.T) {
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
	_, _, err := translateAnthropicRequest(in)
	var te *translateError
	if !errors.As(err, &te) || te.status != 400 {
		t.Fatalf("got err %v, want 400", err)
	}
	if !strings.Contains(te.msg, "url sources") {
		t.Errorf("error should explain the URL rejection; got %q", te.msg)
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

// count_tokens estimates tool schema cost too — otherwise a client using
// count_tokens to size its context budget will undercount a tool-heavy
// request and trip context-window errors at inference time. Hard number
// pinning is avoided (estimator is heuristic) in favor of an inequality.
func TestCountAnthropicTokens_IncludesToolSchemas(t *testing.T) {
	withTools := []byte(`{
		"model": "m",
		"tools": [{"name": "f", "description": "do the thing", "input_schema": {"type":"object","properties":{"x":{"type":"string","description":"the x"}}}}],
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	withoutTools := []byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	nWith, err := countAnthropicTokens(withTools)
	if err != nil {
		t.Fatalf("with tools: %v", err)
	}
	nWithout, err := countAnthropicTokens(withoutTools)
	if err != nil {
		t.Fatalf("without: %v", err)
	}
	if nWith <= nWithout {
		t.Errorf("tools didn't add to count: with=%d without=%d", nWith, nWithout)
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
