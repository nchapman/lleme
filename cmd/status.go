package cmd

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/nchapman/lleme/internal/llama"
	"github.com/nchapman/lleme/internal/proxy"
	"github.com/nchapman/lleme/internal/ui"
	"github.com/spf13/cobra"
)

var statusCmd = &cobra.Command{
	Use:     "status",
	Aliases: []string{"ps"},
	Short:   "Show server status and loaded models",
	GroupID: "model",
	Run: func(cmd *cobra.Command, args []string) {
		state := proxy.GetRunningProxyState()
		if state == nil {
			fmt.Println(ui.Muted("Server is not running"))
			fmt.Println()
			fmt.Println("Start it with: lleme server start")
			fmt.Println("Or use: lleme run <model> (will auto-start server)")
			return
		}

		// Get detailed status from proxy API
		proxyURL := fmt.Sprintf("http://%s:%d", state.Host, state.Port)
		status, err := getProxyStatus(proxyURL)
		if err != nil {
			// Fall back to basic info
			fmt.Println(ui.Header("Server Status"))
			fmt.Printf("  %-12s %s\n", "Address", proxyURL)
			fmt.Printf("  %-12s %d\n", "PID", state.PID)
			fmt.Printf("  %-12s %s\n", "Started", formatTimeSince(state.StartedAt))
			fmt.Println()
			fmt.Printf("%s Could not fetch detailed status: %v\n", ui.Muted("Note:"), err)
			return
		}

		// Pretty print status
		fmt.Println(ui.Header("Server Status"))
		fmt.Printf("  %-12s %s\n", "Address", proxyURL)
		fmt.Printf("  %-12s %d\n", "PID", state.PID)
		fmt.Printf("  %-12s %s\n", "Uptime", formatUptime(time.Duration(status.UptimeSeconds)*time.Second))
		fmt.Printf("  %-12s %d\n", "Max models", status.MaxModels)
		fmt.Println()

		if len(status.Models) == 0 {
			fmt.Println(ui.Muted("No models loaded"))
			fmt.Println()
			fmt.Println("Use 'lleme run <model>' to load a model")
			return
		}

		fmt.Println(ui.Header("Loaded Models"))
		fmt.Println()

		table := ui.NewTable().
			AddColumn("MODEL", 0, ui.AlignLeft).
			AddColumn("PORT", 5, ui.AlignRight).
			AddColumn("STATUS", 0, ui.AlignLeft).
			AddColumn("UNLOADS", 7, ui.AlignLeft)

		// Calculate idle timeout in minutes for "unload in" display
		idleTimeoutMins := 10.0 // default
		if status.IdleTimeout != "" {
			if d, err := time.ParseDuration(status.IdleTimeout); err == nil {
				idleTimeoutMins = d.Minutes()
			}
		}

		for _, m := range status.Models {
			unloadIn := formatUnloadTime(m.IdleMinutes, idleTimeoutMins)
			table.AddRow(m.ModelName, fmt.Sprintf("%d", m.Port), m.Status, unloadIn)
		}

		fmt.Print(table.Render())

		// Footer with model count and llama.cpp credit
		fmt.Println()
		modelWord := "model"
		if len(status.Models) != 1 {
			modelWord = "models"
		}
		installed, _ := llama.GetInstalledVersion()
		if installed != nil {
			fmt.Printf("%d %s loaded %s %s\n", len(status.Models), modelWord, ui.Muted("•"), ui.LlamaCppCredit(installed.TagName))
		} else {
			fmt.Printf("%d %s loaded\n", len(status.Models), modelWord)
		}
	},
}

func getProxyStatus(proxyURL string) (*proxy.ProxyStatus, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(proxyURL + "/api/status")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	var status proxy.ProxyStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, err
	}

	return &status, nil
}

func formatUptime(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%d seconds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%d minutes", int(d.Minutes()))
	}
	if d < 24*time.Hour {
		hours := int(d.Hours())
		mins := int(d.Minutes()) % 60
		return fmt.Sprintf("%dh %dm", hours, mins)
	}
	return fmt.Sprintf("%.1f days", d.Hours()/24)
}

func formatUnloadTime(idleMinutes, timeoutMinutes float64) string {
	remaining := timeoutMinutes - idleMinutes
	if remaining <= 0 {
		return "soon"
	}
	if remaining < 1 {
		secs := int(remaining * 60)
		return fmt.Sprintf("%ds", secs)
	}
	if remaining < 60 {
		return fmt.Sprintf("%.0fm", remaining)
	}
	return fmt.Sprintf("%.1fh", remaining/60)
}

func formatTimeSince(t time.Time) string {
	if t.IsZero() {
		return "-"
	}
	diff := time.Since(t)
	if diff < time.Minute {
		return "just now"
	}
	if diff < time.Hour {
		return fmt.Sprintf("%dm ago", int(diff.Minutes()))
	}
	if diff < 24*time.Hour {
		return fmt.Sprintf("%dh ago", int(diff.Hours()))
	}
	return t.Format("Jan 2 15:04")
}

func init() {
	rootCmd.AddCommand(statusCmd)
}
