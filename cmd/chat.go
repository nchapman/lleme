package cmd

import (
	"context"
	"fmt"
	"strings"

	"github.com/nchapman/lleme/internal/config"
	"github.com/nchapman/lleme/internal/options"
	"github.com/nchapman/lleme/internal/server"
	"github.com/nchapman/lleme/internal/ui"
)

// ChatSession handles one-shot chat completions for CLI prompts.
type ChatSession struct {
	api      *server.APIClient
	model    string
	persona  *config.Persona
	resolver *options.Resolver
	messages []server.ChatMessage

	// Options
	systemPrompt     string
	maxTokens        int
	temp             float64
	topP             float64
	topK             int
	repeatPenalty    float64
	presencePenalty  float64
	frequencyPenalty float64
	minP             float64
}

// NewChatSession creates a new chat session.
func NewChatSession(api *server.APIClient, model string, cfg *config.Config, persona *config.Persona) *ChatSession {
	return &ChatSession{
		api:      api,
		model:    model,
		persona:  persona,
		resolver: options.NewResolver(persona, cfg),
		messages: []server.ChatMessage{},
	}
}

// SetSystemPrompt sets the system prompt for the session.
func (s *ChatSession) SetSystemPrompt(prompt string) {
	s.systemPrompt = prompt
}

// SetSamplingOptions sets the sampling parameters for generation.
func (s *ChatSession) SetSamplingOptions(temp, topP, minP, repeatPenalty, presencePenalty, frequencyPenalty float64, topK, maxTokens int) {
	s.temp = temp
	s.topP = topP
	s.minP = minP
	s.repeatPenalty = repeatPenalty
	s.presencePenalty = presencePenalty
	s.frequencyPenalty = frequencyPenalty
	s.topK = topK
	s.maxTokens = maxTokens
}

// Run sends the prompt to the model and streams the response.
func (s *ChatSession) Run(prompt string) error {
	s.initSystemPrompt()
	s.messages = append(s.messages, server.ChatMessage{Role: "user", Content: prompt})
	return s.streamResponse()
}

// initSystemPrompt sets up the initial system message.
func (s *ChatSession) initSystemPrompt() {
	sysPrompt := s.systemPrompt
	if sysPrompt == "" && s.persona != nil && s.persona.System != "" {
		sysPrompt = s.persona.System
	}
	if sysPrompt == "" {
		sysPrompt = config.DefaultSystemPrompt()
	}
	s.messages = []server.ChatMessage{{Role: "system", Content: sysPrompt}}
}

// streamResponse sends the chat completion request and streams output.
func (s *ChatSession) streamResponse() error {
	req := &server.ChatCompletionRequest{
		Model:           s.model,
		Messages:        s.messages,
		Stream:          true,
		MaxTokens:       s.maxTokens,
		ReasoningFormat: "auto",
	}

	// Apply options: session > persona > config > default
	req.Temperature = s.resolver.ResolveFloat(s.temp, "temp")
	req.TopP = s.resolver.ResolveFloat(s.topP, "top-p")
	req.TopK = s.resolver.ResolveInt(s.topK, "top-k")
	req.MinP = s.resolver.ResolveFloat(s.minP, "min-p")
	req.RepeatPenalty = s.resolver.ResolveFloat(s.repeatPenalty, "repeat-penalty")
	req.PresencePenalty = s.resolver.ResolveFloat(s.presencePenalty, "presence-penalty")
	req.FrequencyPenalty = s.resolver.ResolveFloat(s.frequencyPenalty, "frequency-penalty")

	var fullResponse strings.Builder
	hadReasoning := false
	inReasoning := false

	cb := server.StreamCallback{
		ReasoningCallback: func(reasoning string) {
			inReasoning = true
			hadReasoning = true
			fmt.Print(ui.Muted(reasoning))
		},
		ContentCallback: func(content string) {
			if inReasoning {
				fmt.Print("\n\n")
				inReasoning = false
			}
			fullResponse.WriteString(content)
			fmt.Print(content)
		},
	}

	err := s.api.StreamChatCompletion(context.Background(), req, cb)

	if hadReasoning && fullResponse.Len() == 0 {
		fmt.Println()
	}

	if err != nil {
		return err
	}

	fmt.Println()
	return nil
}
