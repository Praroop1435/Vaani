// Command getmodel downloads the two local models (official ONNX exports) from Hugging Face into
// ./models, once: BAAI/bge-base-en-v1.5 (embedder) and the cross-encoder re-ranker.
// After that the weights live on disk and nothing downloads at runtime.
//
//	go run -tags ORT ./cmd/getmodel
//	go run -tags ORT ./cmd/getmodel -repo BAAI/bge-reranker-base   any other repo with onnx/model.onnx
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/knights-analytics/hugot"

	"github.com/Praroop1435/Vaani/internal/embed"
)

func main() {
	repo := flag.String("repo", "", "download only this Hugging Face repo")
	flag.Parse()
	repos := []string{"BAAI/bge-base-en-v1.5", embed.RerankRepo}
	if *repo != "" {
		repos = []string{*repo}
	}
	for _, r := range repos {
		dir := filepath.Join("models", strings.ReplaceAll(r, "/", "_"))
		if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err == nil {
			fmt.Println("already saved:", dir)
			continue
		}
		if err := os.MkdirAll("models", 0o755); err != nil {
			log.Fatal(err)
		}
		opts := hugot.NewDownloadOptions()
		opts.OnnxFilePath = "onnx/model.onnx"
		path, err := hugot.DownloadModel(context.Background(), r, "models", opts)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("saved:", path)
	}
}
