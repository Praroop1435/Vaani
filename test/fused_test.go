package test

import (
	"context"
	"strings"
	"testing"

	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

func TestPaperOf(t *testing.T) {
	for q, want := range map[string]string{
		"What hardware were the Transformer models trained on?":  "1706.03762v7",
		"What learning rate does Moshi's Depth Transformer use?": "2410.00037v1", // a component, not the paper
		"How many parameters does Helium have?":                  "2410.00037v1",
		"What two techniques make up dynamic chunking in H-Net?": "2507.07955v2",
		"Does H-Net use Transformer layers?":                     "2507.07955v2",
		"How do Moshi and H-Net differ?":                         "", // two papers: no scope
		"What GPUs were used for training?":                      "",
		"What does the transformers library do?":                 "", // whole words only
	} {
		if got := retrieve.PaperOf(q); got != want {
			t.Errorf("PaperOf(%q) = %q, want %q", q, got, want)
		}
	}
}

// fakeScorer prefers passages containing its word, like a cross-encoder that read the answer.
type fakeScorer string

func (f fakeScorer) Scores(_ context.Context, _ string, passages []string) ([]float32, error) {
	out := make([]float32, len(passages))
	for i, p := range passages {
		if strings.Contains(p, string(f)) {
			out[i] = 1
		}
	}
	return out, nil
}

// The q04 failure: asked about the Transformer, the answer cited Moshi's GPUs. With scope the
// Moshi chunk can't be returned at all, and the re-rank lifts the chunk the embedding ranked last.
func TestFusedScopesToThePaperAndReranks(t *testing.T) {
	ix := &retrieve.Index{
		Chunks: []ingest.Chunk{
			{ID: "moshi", Paper: "2410.00037v1", Text: "trained on H100 GPUs"},
			{ID: "tf-other", Paper: "1706.03762v7", Text: "the Transformer uses attention"},
			{ID: "tf-gpus", Paper: "1706.03762v7", Text: "one machine with 8 NVIDIA P100 GPUs"},
		},
		Vecs: [][]float32{{1, 0}, {0.9, 0.1}, {0.1, 0.9}}, // the embedding likes them in this order
	}
	q := []retrieve.Query{{Text: "What hardware were the Transformer models trained on?", Vec: []float32{1, 0}}}
	s := retrieve.Setting{Name: "t", K: 1, Hybrid: true, Candidates: 3, Rerank: true, Scope: true}
	hits, err := ix.SearchFused(context.Background(), q, q[0].Text, q[0].Text, s, fakeScorer("P100"))
	if err != nil || len(hits) != 1 || hits[0].Chunk.ID != "tf-gpus" {
		t.Fatalf("want the Transformer GPU chunk, got %+v err %v", hits, err)
	}
	if hits[0].Score < 0.09 || hits[0].Score > 0.11 {
		t.Errorf("Score must stay the plain cosine (0.1) for the off-topic threshold, got %v", hits[0].Score)
	}
	s.Rerank, s.K = false, 3
	hits, _ = ix.SearchFused(context.Background(), q, q[0].Text, q[0].Text, s, nil)
	for _, h := range hits {
		if h.Chunk.Paper != "1706.03762v7" {
			t.Errorf("scope let %s through", h.Chunk.ID)
		}
	}
}

// The pipeline-mechanics tests (limits, retries, formats, cache) were written against setting A
// and don't load the cross-encoder; setting R is covered here and by cmd/evalretrieval -fix.
func init() { pipeline.Defaults.Setting = retrieve.A }
