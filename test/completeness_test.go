package test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/fakellm"
	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

func oraclePipeline(t *testing.T, s retrieve.Setting) (*pipeline.Pipeline, *eval.Benchmark, map[string]ingest.Chunk) {
	t.Helper()
	b, err := eval.Load("../data/eval/benchmark.json")
	if err != nil {
		t.Fatal(err)
	}
	ix, err := retrieve.Load("../data/index", s.ChunkSize)
	if err != nil {
		t.Skipf("index not built: %v", err)
	}
	srv := httptest.NewServer(&fakellm.Oracle{Bench: b})
	t.Cleanup(srv.Close)
	cfg := pipeline.Defaults
	cfg.Setting = s
	byID := map[string]ingest.Chunk{}
	for _, c := range ix.Chunks {
		byID[c.ID] = c
	}
	return &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: srv.URL, Model: "oracle"},
		Gov: governor.New(governor.Limits{RPM: 1000, TPM: 1_000_000, RPD: 100_000}),
		Ret: &retrieve.Retriever{Embedder: embedder(t), Indexes: map[int]*retrieve.Index{s.ChunkSize: ix}},
		Cfg: cfg,
	}, b, byID
}

// The live smoke test's q14 answered 2 of 3 parts. With parts and targeted retrieval, the missing
// "tokens Helium was pretrained on" part must trigger one targeted search that finds 2.1T.
func TestPipelineCompletesQ14WithTargetedRetrieval(t *testing.T) {
	p, b, byID := oraclePipeline(t, retrieve.A)
	q := b.Questions[slices.IndexFunc(b.Questions, func(q eval.Question) bool { return q.ID == "q14" })]
	tr := p.Answer(context.Background(), "q14", q.Question, nil)
	if len(tr.Parts) != 3 {
		t.Fatalf("parts %q", tr.Parts)
	}
	if !slices.ContainsFunc(tr.Steps, func(s pipeline.Step) bool { return s.Name == "retrieve_targeted" }) || len(tr.Targeted) == 0 {
		t.Fatalf("no targeted retrieval; steps %+v", tr.Steps)
	}
	if !strings.Contains(tr.Answer, "2.1T") || len(tr.Missing) != 0 || tr.ModelCalls != 3 {
		t.Errorf("answer %q missing %q calls %d", tr.Answer, tr.Missing, tr.ModelCalls)
	}
	var cited []ingest.Chunk
	for _, c := range tr.Citations {
		cited = append(cited, byID[c.ChunkID])
	}
	if g := eval.GradeAnswer(q, tr.Answer, tr.Refused, cited); g.Grade != "correct" || !g.CiteOK {
		t.Errorf("grade %+v for %q", g, tr.Answer)
	}
}

// scripted model that records every request body
type recorder struct {
	mu      sync.Mutex
	bodies  []string
	replies []string
}

func (rc *recorder) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	rc.mu.Lock()
	rc.bodies = append(rc.bodies, string(body))
	reply := "NOT IN PAPERS"
	if len(rc.replies) > 0 {
		reply, rc.replies = rc.replies[0], rc.replies[1:]
	}
	rc.mu.Unlock()
	streamReply(reply, 0)(w)
}

// "[P2] MISSING" has no [S#] tag: it must still end its unit, not swallow the [P3] sentence after it,
// and the targeted round must ask for P2 only.
func TestMissingPartIsParsedAndRetargeted(t *testing.T) {
	rc := &recorder{replies: []string{
		`{"query": "Moshi Helium", "parts": ["what Moshi is", "something the passages lack", "what Moshi is again"]}`,
		"[P1] Moshi is a speech-text foundation model [S1]. [P2] MISSING [P3] Moshi is a dialogue model [S1].",
		"[P2] MISSING",
	}}
	srv := httptest.NewServer(rc)
	defer srv.Close()
	ix, err := retrieve.Load("../data/index", 300)
	if err != nil {
		t.Skip(err)
	}
	p := &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: srv.URL, Model: "scripted"},
		Gov: governor.New(governor.Limits{RPM: 1000, TPM: 1_000_000, RPD: 100_000}),
		Ret: &retrieve.Retriever{Embedder: embedder(t), Indexes: map[int]*retrieve.Index{300: ix}},
		Cfg: pipeline.Defaults,
	}
	tr := p.Answer(context.Background(), "m1", "What is Moshi?", nil)
	if len(rc.bodies) != 3 {
		t.Fatalf("%d model calls, want 3 (understand, draft, targeted draft)", len(rc.bodies))
	}
	last := rc.bodies[2]
	if !strings.Contains(last, `[P2] something the passages lack`) || strings.Contains(last, `what Moshi is`) {
		t.Errorf("targeted draft must ask for P2 only:\n%s", last)
	}
	if !strings.Contains(tr.Answer, "Moshi is a dialogue model") || !slices.Equal(tr.Missing, []string{"something the passages lack"}) {
		t.Errorf("answer %q missing %q", tr.Answer, tr.Missing)
	}
	if !strings.HasSuffix(tr.Answer, "I could not find something the passages lack in the papers.") {
		t.Errorf("an unanswered part must be said plainly: %q", tr.Answer)
	}
}

// Smoke test: two passages from page 8 rendered as "[Attention Is All You Need, p.8] [Attention Is All You Need, p.8]".
func TestCitationLabelsAreDistinctOnSamePage(t *testing.T) {
	cites := map[string]pipeline.Citation{
		"S1": pipeline.NewCitation("S1", ingest.Chunk{ID: "1706.03762v7#c030", Paper: "1706.03762v7", PageStart: 8, PageEnd: 8, SectionNumber: "6.1", Section: "Machine Translation"}),
		"S2": pipeline.NewCitation("S2", ingest.Chunk{ID: "1706.03762v7#t001", Paper: "1706.03762v7", PageStart: 8, PageEnd: 8, SectionNumber: "5.4", Section: "Regularization", Kind: "table", Table: "Table 2"}),
	}
	got := pipeline.Render("It reaches 28.4 BLEU [S1][S2].", cites)
	if got != "It reaches 28.4 BLEU [Attention Is All You Need, p.8, §6.1; Attention Is All You Need, p.8, Table 2]." {
		t.Errorf("got %q", got)
	}
	if got := pipeline.Render("Twice the same passage [S1] [S1].", cites); strings.Count(got, "§6.1") != 1 {
		t.Errorf("one passage cited twice must show one label: %q", got)
	}
	if cites["S1"].Tag == cites["S2"].Tag || cites["S1"].ChunkID == cites["S2"].ChunkID {
		t.Error("internal ids must be unique per passage")
	}
}
