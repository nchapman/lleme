package chat

import (
	"strings"
	"testing"
)

func TestHandleSet(t *testing.T) {
	tests := []struct {
		name         string
		option       string
		value        string
		wantErr      bool
		wantMessage  string
		check        func(m *Model) bool
		checkMessage string
	}{
		{
			name:         "float option",
			option:       "temp",
			value:        "0.7",
			wantMessage:  "Set temp = 0.7",
			check:        func(m *Model) bool { return m.options.Temp == 0.7 },
			checkMessage: "options.Temp not applied",
		},
		{
			name:         "temperature alias",
			option:       "temperature",
			value:        "1.5",
			wantMessage:  "Set temperature = 1.5",
			check:        func(m *Model) bool { return m.options.Temp == 1.5 },
			checkMessage: "temperature alias did not map to Temp",
		},
		{
			name:         "int option",
			option:       "top-k",
			value:        "40",
			wantMessage:  "Set top-k = 40",
			check:        func(m *Model) bool { return m.options.TopK == 40 },
			checkMessage: "options.TopK not applied",
		},
		{
			name:         "reload option sets pendingReload",
			option:       "ctx-size",
			value:        "8192",
			wantMessage:  "Set ctx-size = 8192 (use /reload to apply)",
			check:        func(m *Model) bool { return m.pendingReload && m.options.CtxSize == 8192 && m.options.CtxSizeSet },
			checkMessage: "ctx-size/pendingReload not applied",
		},
		{
			name:         "presence penalty",
			option:       "presence-penalty",
			value:        "0.5",
			wantMessage:  "Set presence-penalty = 0.5",
			check:        func(m *Model) bool { return m.options.PresencePenalty == 0.5 },
			checkMessage: "PresencePenalty not applied",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{}
			result := m.handleSet(tt.option, tt.value)
			if result.IsError {
				t.Fatalf("unexpected error: %s", result.Message)
			}
			if result.Message != tt.wantMessage {
				t.Errorf("message = %q, want %q", result.Message, tt.wantMessage)
			}
			if !tt.check(m) {
				t.Error(tt.checkMessage)
			}
		})
	}
}

func TestHandleSetErrors(t *testing.T) {
	tests := []struct {
		name   string
		option string
		value  string
	}{
		{"unknown option", "bogus", "1"},
		{"invalid float", "temp", "abc"},
		{"invalid int", "top-k", "1.5"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{}
			result := m.handleSet(tt.option, tt.value)
			if !result.IsError {
				t.Errorf("expected error for %s %s", tt.option, tt.value)
			}
		})
	}
}

func TestHandleCommandResults(t *testing.T) {
	t.Run("unknown command errors", func(t *testing.T) {
		m := &Model{}
		cmd := m.handleCommand("/bogus")
		msg, ok := cmd().(CommandResultMsg)
		if !ok {
			t.Fatalf("cmd() = %T, want CommandResultMsg", cmd())
		}
		if !msg.IsError || !strings.Contains(msg.Message, "Unknown command") {
			t.Errorf("msg = %+v, want unknown-command error", msg)
		}
	})

	t.Run("set routes through handleSet", func(t *testing.T) {
		m := &Model{}
		cmd := m.handleCommand("/set temp 0.9")
		msg, ok := cmd().(CommandResultMsg)
		if !ok {
			t.Fatalf("cmd() = %T, want CommandResultMsg", cmd())
		}
		if msg.IsError || m.options.Temp != 0.9 {
			t.Errorf("msg = %+v, temp = %v", msg, m.options.Temp)
		}
	})

	t.Run("set without value shows usage", func(t *testing.T) {
		m := &Model{}
		cmd := m.handleCommand("/set temp")
		msg, ok := cmd().(CommandResultMsg)
		if !ok {
			t.Fatalf("cmd() = %T, want CommandResultMsg", cmd())
		}
		if !msg.IsError || !strings.Contains(msg.Message, "Usage") {
			t.Errorf("msg = %+v, want usage error", msg)
		}
	})
}
