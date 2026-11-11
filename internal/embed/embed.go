// Package embed turns text into BGE vectors in-process, using BAAI's official ONNX weights
// (Hugging Face BAAI/bge-base-en-v1.5) loaded from disk through ONNX Runtime.
// Build and test with `-tags ORT`.
package embed

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

const (
	DefaultModelDir = "models/BAAI_bge-base-en-v1.5" // saved by `go run -tags ORT ./cmd/getmodel`
	ORTLibDir       = "/opt/homebrew/lib"            // `brew install onnxruntime`
	MaxTokens       = 512
	Dim             = 768
	// BGE v1.5 retrieval convention: prefix queries, never passages.
	QueryPrefix = "Represent this sentence for searching relevant passages: "
)

type Embedder struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
	mu       sync.Mutex // shortcut: one inference at a time; revisit if burst tests show the embedder queueing
}

// New loads the model from modelDir. It never downloads.
func New(ctx context.Context, modelDir string) (*Embedder, error) {
	session, err := hugot.NewORTSession(ctx, options.WithOnnxLibraryPath(ORTLibDir))
	if err != nil {
		return nil, fmt.Errorf("embed: ONNX Runtime session (built with -tags ORT?): %w", err)
	}
	p, err := hugot.NewPipeline(session, hugot.FeatureExtractionConfig{
		ModelPath: modelDir,
		Name:      "bge-base",
	})
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("embed: load %s (run cmd/getmodel once): %w", modelDir, err)
	}
	return &Embedder{session: session, pipeline: p}, nil
}

func (e *Embedder) Close() error { return e.session.Destroy() }

// CountTokens returns BERT tokens per text, including [CLS] and [SEP]. Ingest uses it to size chunks.
func (e *Embedder) CountTokens(texts []string) []int {
	batch := &backends.PipelineBatch{}
	backends.TokenizeInputs(batch, e.pipeline.Model.Tokenizer, texts)
	n := make([]int, len(texts))
	for i, in := range batch.Input {
		n[i] = len(in.TokenIDs)
	}
	return n
}

// Embed returns one unit-length vector per text. Text over 512 tokens is an error, never silently cut.
// Pooling is the [CLS] token, as BGE was trained (hugot's default mean pooling would be wrong for BGE).
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	for i, n := range e.CountTokens(texts) {
		if n > MaxTokens {
			return nil, fmt.Errorf("embed: text %d has %d tokens, max %d", i, n, MaxTokens)
		}
	}
	e.mu.Lock()
	raw, err := e.pipeline.RunRaw(ctx, texts)
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i, tokens := range raw.HiddenStates {
		out[i] = normalize(tokens[0]) // [CLS] is always the first token
	}
	return out, nil
}

func normalize(v []float32) []float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(s))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / n
	}
	return out
}

// Cosine of two unit vectors is their dot product.
func Cosine(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
