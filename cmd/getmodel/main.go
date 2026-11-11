// Command getmodel downloads BAAI/bge-base-en-v1.5 (official ONNX export) from Hugging Face into ./models, once.
// After that the weights live on disk and nothing downloads at runtime.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/knights-analytics/hugot"

	"github.com/Praroop1435/Vaani/internal/embed"
)

func main() {
	if _, err := os.Stat(filepath.Join(embed.DefaultModelDir, "model.onnx")); err == nil {
		fmt.Println("already saved:", embed.DefaultModelDir)
		return
	}
	if err := os.MkdirAll(filepath.Dir(embed.DefaultModelDir), 0o755); err != nil {
		log.Fatal(err)
	}
	opts := hugot.NewDownloadOptions()
	opts.OnnxFilePath = "onnx/model.onnx"
	path, err := hugot.DownloadModel(context.Background(), "BAAI/bge-base-en-v1.5", filepath.Dir(embed.DefaultModelDir), opts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("saved:", path)
}
