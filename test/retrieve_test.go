package test

import (
	"context"
	"os"
	"regexp"
	"testing"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

func TestSearchOrdersByCosine(t *testing.T) {
	unit := func(x, y float32) []float32 { // 2-d unit vectors are enough for ordering
		v := make([]float32, embed.Dim)
		v[0], v[1] = x, y
		return v
	}
	ix := &retrieve.Index{
		Chunks: []ingest.Chunk{{ID: "far"}, {ID: "best"}, {ID: "mid"}},
		Vecs:   [][]float32{unit(0, 1), unit(1, 0), unit(0.8, 0.6)},
	}
	hits := ix.Search(unit(1, 0), 2)
	if len(hits) != 2 || hits[0].Chunk.ID != "best" || hits[1].Chunk.ID != "mid" {
		t.Errorf("got %+v", hits)
	}
	if got := ix.Search(unit(1, 0), 10); len(got) != 3 {
		t.Errorf("k larger than the index should return everything, got %d", len(got))
	}
}

func TestWindowsCutSentencesAndKeepTablesWhole(t *testing.T) {
	chunks := []ingest.Chunk{
		{Kind: "text", Text: "One is here. Two is here. Three is here. Four is here."},
		{Kind: "table", Text: "Table 1 (page 2): Caption. | a | b |"},
		{Kind: "text", Text: "Short. Chunk."},
	}
	ws := ingest.Windows(chunks, 2, 1)
	want := []ingest.Window{
		{Parent: 0, Text: "One is here. Two is here."},
		{Parent: 0, Text: "Two is here. Three is here."},
		{Parent: 0, Text: "Three is here. Four is here."},
		{Parent: 1, Text: "Table 1 (page 2): Caption. | a | b |"},
		{Parent: 2, Text: "Short. Chunk."},
	}
	if len(ws) != len(want) {
		t.Fatalf("got %+v", ws)
	}
	for i := range want {
		if ws[i] != want[i] {
			t.Errorf("window %d = %+v, want %+v", i, ws[i], want[i])
		}
	}
}

func TestSearchWindowsReturnsParentsOnce(t *testing.T) {
	unit := func(x, y float32) []float32 {
		v := make([]float32, embed.Dim)
		v[0], v[1] = x, y
		return v
	}
	ix := &retrieve.Index{
		Chunks:    []ingest.Chunk{{ID: "p0"}, {ID: "p1"}},
		WinParent: []int{0, 0, 1},
		WinVecs:   [][]float32{unit(0, 1), unit(1, 0), unit(0.8, 0.6)}, // p0's 2nd window is the best match
	}
	hits := ix.SearchWindows(unit(1, 0), 5)
	if len(hits) != 2 || hits[0].Chunk.ID != "p0" || hits[0].Score < 0.99 || hits[1].Chunk.ID != "p1" {
		t.Errorf("got %+v", hits)
	}
}

func TestLoadRejectsMismatchedVectors(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(dir+"/chunks_7.json", []byte(`[{"id":"a"},{"id":"b"}]`), 0o644)
	os.WriteFile(dir+"/vectors_7.f32", make([]byte, embed.Dim*4), 0o644) // one vector for two chunks
	if _, err := retrieve.Load(dir, 7); err == nil {
		t.Error("stale vectors file was accepted")
	}
}

// Questions whose answer location is known from reading the papers. A setting passes a question
// when some retrieved chunk contains the answer text.
// answer is a regexp: the papers write one fact several ways ("12.5 Hz", "12.5Hz", "12.5 frames per second").
var retrievalProbes = []struct{ question, paper, answer string }{
	{"How many parallel attention heads does the base Transformer use?", "1706.03762v7", `h = 8`},
	{"What BLEU score does the big Transformer get on English-to-German newstest2014?", "1706.03762v7", `28\.4`},
	{"What frame rate does the Mimi codec run at?", "2410.00037v1", `12\.5\s*(Hz|hz|frames per second)`},
	{"What does the ratio loss in H-Net prevent?", "2507.07955v2", `trivial solutions`},
	{"What is Inner Monologue in Moshi?", "2410.00037v1", `Inner Monologue`},
}

func TestRetrieveRealIndex(t *testing.T) {
	e := embedder(t)
	r := &retrieve.Retriever{Embedder: e, Indexes: map[int]*retrieve.Index{}}
	settings := []retrieve.Setting{retrieve.A, retrieve.B, retrieve.C}
	for _, s := range settings {
		ix, err := retrieve.Load("../data/index", s.ChunkSize)
		if err != nil {
			t.Skipf("index not built (run cmd/ingest): %v", err)
		}
		r.Indexes[s.ChunkSize] = ix
	}
	ctx := context.Background()
	foundBy := map[string]int{}
	for _, s := range settings {
		found := 0
		for _, p := range retrievalProbes {
			res, err := r.Retrieve(ctx, p.question, s)
			if err != nil {
				t.Fatal(err)
			}
			rank := -1
			for i, h := range res.Hits {
				if h.Chunk.Paper == p.paper && regexp.MustCompile(p.answer).MatchString(h.Chunk.Text) && rank < 0 {
					rank = i + 1
				}
			}
			if rank > 0 {
				found++
			}
			top := res.Hits[0]
			t.Logf("%s | rank=%2d top=%.3f %s §%s %s | embed=%.1fms search=%.2fms | %s",
				s.Name, rank, top.Score, top.Chunk.ID, top.Chunk.SectionNumber, top.Chunk.Kind, res.EmbedMs, res.SearchMs, p.question)
		}
		t.Logf("%s: answer chunk retrieved for %d/%d questions", s.Name, found, len(retrievalProbes))
		foundBy[s.Name] = found
		// Smoke-test floor, not a quality target (the benchmark measures quality). Known miss: A and B
		// rank the "h = 8" chunk 25-27 (formula-heavy chunk); C ranks the Mimi rate 5th.
		if found < 3 {
			t.Errorf("%s found only %d/%d answer chunks", s.Name, found, len(retrievalProbes))
		}
	}

	if foundBy[retrieve.C.Name] < foundBy[retrieve.A.Name] {
		t.Errorf("small-to-big (C) found %d, fewer than its plain parent index A (%d)", foundBy[retrieve.C.Name], foundBy[retrieve.A.Name])
	}

	// Off-library questions: their top score shows where a "papers don't say" threshold could sit.
	for _, q := range []string{"What is the capital of France?", "How do I bake sourdough bread?", "What GPU did the Moshi authors use to train Helium?"} {
		res, _ := r.Retrieve(ctx, q, retrieve.A)
		t.Logf("off-topic? top=%.3f %s | %s", res.Hits[0].Score, res.Hits[0].Chunk.ID, q)
	}
}
