package test

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
)

// Runs BAAI's official bge-base-en-v1.5 in-process from ../models. Costs no Groq budget.
//   go test -tags ORT ./test -v

var (
	embOnce sync.Once
	emb     *embed.Embedder
	embErr  error
)

func embedder(t *testing.T) *embed.Embedder {
	t.Helper()
	dir := "../" + embed.DefaultModelDir
	if _, err := os.Stat(dir + "/model.onnx"); err != nil {
		t.Skip("model not saved: run `go run -tags ORT ./cmd/getmodel` once")
	}
	embOnce.Do(func() { emb, embErr = embed.New(context.Background(), dir) })
	if embErr != nil {
		t.Fatal(embErr)
	}
	return emb
}

func TestEmbedShape(t *testing.T) {
	e := embedder(t)
	vecs, err := e.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range vecs {
		if len(v) != embed.Dim {
			t.Fatalf("dim = %d, want %d", len(v), embed.Dim)
		}
		if n := math.Sqrt(float64(embed.Cosine(v, v))); math.Abs(n-1) > 1e-3 {
			t.Errorf("norm = %f, want 1", n)
		}
	}
}

func TestEmbedRanksRelevantPassageFirst(t *testing.T) {
	e := embedder(t)
	q := embed.QueryPrefix + "How many attention heads does the base Transformer use?"
	passages := []string{
		"In this work we employ h = 8 parallel attention layers, or heads.", // relevant
		"Moshi is a speech-text foundation model for real-time dialogue.",
		"H-Net uses a dynamic chunking mechanism to segment raw bytes.",
	}
	vecs, err := e.Embed(context.Background(), append([]string{q}, passages...))
	if err != nil {
		t.Fatal(err)
	}
	scores := make([]float32, len(passages))
	for i := range passages {
		scores[i] = embed.Cosine(vecs[0], vecs[i+1])
	}
	t.Logf("scores relevant=%.3f moshi=%.3f hnet=%.3f", scores[0], scores[1], scores[2])
	if scores[0] <= max(scores[1], scores[2]) {
		t.Errorf("relevant passage not ranked first: %v", scores)
	}
}

func TestEmbedRejectsOver512Tokens(t *testing.T) {
	e := embedder(t)
	long := strings.Repeat("attention ", 700)
	if n := e.CountTokens([]string{long})[0]; n != 702 { // 700 words + [CLS] + [SEP]
		t.Errorf("CountTokens = %d, want 702", n)
	}
	if _, err := e.Embed(context.Background(), []string{long}); err == nil {
		t.Error("over-long chunk was accepted; it would be silently truncated")
	}
}

func TestEmbedQueryLatency(t *testing.T) {
	e := embedder(t)
	ctx := context.Background()
	q := []string{embed.QueryPrefix + "What does the routing module in dynamic chunking predict?"}
	if _, err := e.Embed(ctx, q); err != nil { // warm-up
		t.Fatal(err)
	}
	var ms []float64
	for range 30 {
		start := time.Now()
		if _, err := e.Embed(ctx, q); err != nil {
			t.Fatal(err)
		}
		ms = append(ms, float64(time.Since(start).Microseconds())/1000)
	}
	slices.Sort(ms)
	p50, p95 := ms[len(ms)/2], ms[len(ms)*95/100]
	t.Logf("query embed latency over %d calls: p50=%.1fms p95=%.1fms max=%.1fms", len(ms), p50, p95, ms[len(ms)-1])
	if p50 > 200 {
		t.Errorf("p50 %.1fms is too slow for the request path", p50)
	}
}

// Cross-check: an independent implementation (Ollama's GGUF conversion) must give near-identical vectors.
// Skips when Ollama is not running.
func TestEmbedMatchesOllamaConversion(t *testing.T) {
	e := embedder(t)
	texts := []string{
		embed.QueryPrefix + "How many attention heads does the base Transformer use?",
		"Moshi is a speech-text foundation model for real-time dialogue.",
	}
	body, _ := json.Marshal(map[string]any{"model": "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:F16", "input": texts})
	resp, err := http.Post("http://localhost:11434/api/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Skip("Ollama not running")
	}
	defer resp.Body.Close()
	var ref struct{ Embeddings [][]float32 }
	if err := json.NewDecoder(resp.Body).Decode(&ref); err != nil || len(ref.Embeddings) != len(texts) {
		t.Skipf("Ollama reply unusable: %v", err)
	}
	ours, err := e.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range texts {
		sim := embed.Cosine(ours[i], ref.Embeddings[i])
		t.Logf("text %d: cosine(official ONNX, Ollama GGUF) = %.5f", i, sim)
		if sim < 0.99 {
			t.Errorf("text %d: vectors disagree (%.4f): pooling or weights differ", i, sim)
		}
	}
}
