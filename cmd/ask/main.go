// Command ask answers one question through the full pipeline and prints the reply as it is
// verified, then the trace (steps with waiting / working / retrying time) as JSON.
// Spends 1-3 real Groq requests unless -base points at the stand-in.
//
//	go run -tags ORT ./cmd/ask "How many attention heads does the base Transformer use?"
//	go run -tags ORT ./cmd/ask -setting C -no-understand "..."
//	go run -tags ORT ./cmd/ask -cache results/answer_cache.json "..."   opt-in semantic answer cache
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

func main() {
	base := flag.String("base", llm.GroqBaseURL, "OpenAI-style endpoint (Groq, or the stand-in)")
	model := flag.String("model", "qwen/qwen3.8-27b", "model id")
	setting := flag.String("setting", "R", "retrieval setting: A, B, C, H10, H15 or R")
	noUnderstand := flag.Bool("no-understand", false, "skip the query-rewrite model call")
	cachePath := flag.String("cache", "", "semantic answer cache file (off when empty)")
	cacheThreshold := flag.Float64("cache-threshold", float64(pipeline.DefaultCacheThreshold), "min cosine for a cache hit")
	flag.Parse()
	question := strings.Join(flag.Args(), " ")
	if question == "" {
		log.Fatal("usage: ask [flags] <question>")
	}
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		key = envFile(".env", "GROQ_API_KEY")
	}

	ctx := context.Background()
	e, err := embed.New(ctx, embed.DefaultModelDir)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	s := retrieve.Named[*setting]
	ix, err := retrieve.Load("data/index", s.ChunkSize)
	if err != nil {
		log.Fatal(err)
	}
	if code, d, err := (&llm.Client{BaseURL: *base}).Warm(ctx); err == nil {
		fmt.Fprintf(os.Stderr, "warm-up: HTTP %d in %s\n", code, d.Round(time.Millisecond))
	}
	cfg := pipeline.Defaults
	cfg.Setting, cfg.Understand = s, !*noUnderstand
	ret := &retrieve.Retriever{Embedder: e, Indexes: map[int]*retrieve.Index{s.ChunkSize: ix}}
	if err := ret.LoadReranker(); err != nil {
		log.Fatal(err)
	}
	p := &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: *base, APIKey: key, Model: *model},
		Gov: governor.New(governor.Limits{RPM: 30, TPM: 8000, RPD: 1000}), // Groq free tier, gpt-oss-120b (docs, 2026-11-09)
		Ret: ret,
		Cfg: cfg,
	}
	var c *pipeline.Cache
	if *cachePath != "" {
		if c, err = pipeline.LoadCache(*cachePath, float32(*cacheThreshold)); err != nil {
			log.Fatal(err)
		}
	}
	tr, cr := p.AnswerCached(ctx, c, "ask-1", question, func(piece string) { fmt.Print(piece) })
	fmt.Println()
	js, _ := json.MarshalIndent(tr, "", "  ")
	fmt.Fprintln(os.Stderr, string(js))
	if c != nil {
		js, _ = json.Marshal(map[string]any{"cache": cr, "stats": c.Stats(), "entries": c.Len()})
		fmt.Fprintln(os.Stderr, string(js))
		if err := c.Save(*cachePath); err != nil {
			log.Fatal(err)
		}
	}
}

func envFile(path, name string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if v, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), name+"="); ok {
			return strings.Trim(v, `"'`)
		}
	}
	return ""
}
