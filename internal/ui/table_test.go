package ui

import (
	"strings"
	"testing"
)

func TestNewTable(t *testing.T) {
	tbl := NewTable()
	if tbl == nil {
		t.Fatal("Expected non-nil Table")
	}
	if tbl.indent != 2 {
		t.Errorf("Expected default indent of 2, got %d", tbl.indent)
	}
}

func TestTableIndent(t *testing.T) {
	tbl := NewTable().Indent(4)
	if tbl.indent != 4 {
		t.Errorf("Expected indent of 4, got %d", tbl.indent)
	}
}

func TestTableAddColumn(t *testing.T) {
	tbl := NewTable().
		AddColumn("Name", 20, AlignLeft).
		AddColumn("Size", 10, AlignRight)

	if len(tbl.columns) != 2 {
		t.Fatalf("Expected 2 columns, got %d", len(tbl.columns))
	}
	if tbl.columns[0].Header != "Name" {
		t.Errorf("Expected first column header 'Name', got '%s'", tbl.columns[0].Header)
	}
	if tbl.columns[1].Align != AlignRight {
		t.Error("Expected second column to be right-aligned")
	}
}

func TestTableAddRow(t *testing.T) {
	tbl := NewTable().
		AddColumn("A", 5, AlignLeft).
		AddRow("1", "2").
		AddRow("3", "4")

	if len(tbl.rows) != 2 {
		t.Fatalf("Expected 2 rows, got %d", len(tbl.rows))
	}
	if tbl.rows[0][0] != "1" {
		t.Errorf("Expected first cell '1', got '%s'", tbl.rows[0][0])
	}
}

func TestTableFormatCell(t *testing.T) {
	tbl := NewTable()

	t.Run("left align pads right", func(t *testing.T) {
		col := Column{Header: "Test", Width: 10, Align: AlignLeft}
		result := tbl.formatCell("hi", col)
		if result != "hi        " {
			t.Errorf("Expected 'hi        ', got '%s'", result)
		}
	})

	t.Run("right align pads left", func(t *testing.T) {
		col := Column{Header: "Test", Width: 10, Align: AlignRight}
		result := tbl.formatCell("hi", col)
		if result != "        hi" {
			t.Errorf("Expected '        hi', got '%s'", result)
		}
	})

	t.Run("truncates long text with ellipsis", func(t *testing.T) {
		col := Column{Header: "Test", Width: 10, Align: AlignLeft}
		result := tbl.formatCell("this is a very long string", col)
		if result != "this is..." {
			t.Errorf("Expected 'this is...', got '%s'", result)
		}
	})

	t.Run("truncates without ellipsis for tiny width", func(t *testing.T) {
		col := Column{Header: "Test", Width: 3, Align: AlignLeft}
		result := tbl.formatCell("hello", col)
		if result != "hel" {
			t.Errorf("Expected 'hel', got '%s'", result)
		}
	})
}

func TestTableRender(t *testing.T) {
	t.Run("returns empty string for no columns", func(t *testing.T) {
		tbl := NewTable()
		if tbl.Render() != "" {
			t.Error("Expected empty string for table with no columns")
		}
	})

	t.Run("renders header and rows", func(t *testing.T) {
		tbl := NewTable().Indent(0).
			AddColumn("Name", 10, AlignLeft).
			AddColumn("Value", 5, AlignRight).
			AddRow("foo", "123").
			AddRow("bar", "456")

		result := tbl.Render()

		// Check that it contains the data
		if !strings.Contains(result, "foo") {
			t.Error("Expected result to contain 'foo'")
		}
		if !strings.Contains(result, "123") {
			t.Error("Expected result to contain '123'")
		}
		if !strings.Contains(result, "bar") {
			t.Error("Expected result to contain 'bar'")
		}

		// Check row count (header + 2 data rows = 3 lines)
		lines := strings.Split(strings.TrimSuffix(result, "\n"), "\n")
		if len(lines) != 3 {
			t.Errorf("Expected 3 lines, got %d", len(lines))
		}
	})

	t.Run("handles missing row values", func(t *testing.T) {
		tbl := NewTable().Indent(0).
			AddColumn("A", 5, AlignLeft).
			AddColumn("B", 5, AlignLeft).
			AddRow("x") // Only one value for two columns

		result := tbl.Render()
		if !strings.Contains(result, "x") {
			t.Error("Expected result to contain 'x'")
		}
		// Should not panic with missing values
	})
}

func TestAlignment(t *testing.T) {
	if AlignLeft != 0 {
		t.Errorf("Expected AlignLeft to be 0, got %d", AlignLeft)
	}
	if AlignRight != 1 {
		t.Errorf("Expected AlignRight to be 1, got %d", AlignRight)
	}
}

func TestTableAutoWidth(t *testing.T) {
	t.Run("auto-sizes based on content", func(t *testing.T) {
		tbl := NewTable().Indent(0).
			AddColumn("NAME", 0, AlignLeft). // Auto-width
			AddColumn("SIZE", 6, AlignRight)
		tbl.AddRow("short", "100")
		tbl.AddRow("this-is-a-long-model-name", "200")

		result := tbl.Render()

		// Should contain the full long name without truncation
		if !strings.Contains(result, "this-is-a-long-model-name") {
			t.Error("Expected auto-width column to contain full long name")
		}
	})

	t.Run("uses header width as minimum", func(t *testing.T) {
		tbl := NewTable().Indent(0).
			AddColumn("LONGHEADER", 0, AlignLeft) // Auto-width, header is 10 chars
		tbl.AddRow("x") // Short value

		result := tbl.Render()
		lines := strings.Split(strings.TrimSuffix(result, "\n"), "\n")

		// Data row should be padded to at least header length
		if len(lines) < 2 {
			t.Fatal("Expected at least 2 lines")
		}
		// "x" should be padded to match "LONGHEADER" width
		if !strings.Contains(lines[1], "x         ") {
			t.Errorf("Expected 'x' to be padded to header width, got '%s'", lines[1])
		}
	})
}
