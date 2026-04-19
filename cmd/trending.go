package cmd

import (
	"github.com/spf13/cobra"
)

var trendingCmd = &cobra.Command{
	Use:     "trending",
	Short:   "Show trending models on Hugging Face",
	GroupID: "discovery",
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		searchCmd.Run(cmd, []string{})
	},
}

func init() {
	rootCmd.AddCommand(trendingCmd)
}
