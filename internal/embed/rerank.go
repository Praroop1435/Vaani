package embed

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
)

const (
	RerankRepo     = "cross-encoder/ms-marco-MiniLM-L6-v2"
	RerankModelDir = "models/cross-encoder_ms-marco-MiniLM-L6-v2" // saved by cmd/getmodel
)

// Reranker scores (question, passage) pairs with a cross-encoder: it reads both together, so it
// can tell "the passage answers this" from "the passage shares words with this".
type Reranker struct {
	p  *pipelines.CrossEncoderPipeline
	mu sync.Mutex // one inference at a time, like Embedder
}

// NewReranker loads the cross-encoder into the embedder's ONNX session. It never downloads.
func (e *Embedder) NewReranker(modelDir string) (*Reranker, error) {
	p, err := hugot.NewPipeline(e.session, hugot.CrossEncoderConfig{
		ModelPath: modelDir,
		Name:      fmt.Sprintf("rerank:%s:%d", modelDir, RerankBatch),
		Options:   []hugot.CrossEncoderOption{pipelines.WithBatchSize(RerankBatch)},
	})
	if err != nil {
		return nil, fmt.Errorf("embed: load re-ranker %s (run cmd/getmodel once): %w", modelDir, err)
	}
	return &Reranker{p: p}, nil
}

// Scores returns one relevance score per passage, in the order given (higher is better).
func (r *Reranker) Scores(ctx context.Context, question string, passages []string) ([]float32, error) {
	passages = r.fit(question, passages)
	r.mu.Lock()
	out, err := r.p.RunPipeline(ctx, question, passages)
	r.mu.Unlock()
	if err != nil {
		return nil, err
	}
	s := make([]float32, len(passages))
	for _, res := range out.Results {
		s[res.Index] = res.Score
	}
	return s, nil
}

// fit cuts each passage's tail until question + passage fit the model's 512 positions; the
// model fails outright on longer input. A 300-token chunk only passes with a long question.
func (r *Reranker) fit(question string, passages []string) []string {
	count := func(texts []string) []int {
		batch := &backends.PipelineBatch{}
		backends.TokenizeInputs(batch, r.p.Model.Tokenizer, texts)
		n := make([]int, len(texts))
		for i, in := range batch.Input {
			n[i] = len(in.TokenIDs)
		}
		return n
	}
	room := MaxTokens - 8 - count([]string{question})[0] // 8: spare for the pair's separator tokens
	out := make([]string, len(passages))
	for i, n := range count(passages) {
		out[i] = passages[i]
		for n > room {
			w := strings.Fields(out[i])
			out[i] = strings.Join(w[:len(w)*room/n*9/10], " ")
			n = count(out[i : i+1])[0]
		}
	}
	return out
}

// RerankBatch is pairs per ONNX run.
var RerankBatch = 16
