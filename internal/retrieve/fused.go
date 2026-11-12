package retrieve

// Setting R: the retrieval the pipeline uses after the first real run showed A's weak points.
//
//  1. Scope: a question that names one paper searches only that paper (q04 cited Moshi's GPUs
//     for a Transformer question).
//  2. Wide fetch: BGE top-Candidates + BM25 top-Candidates, both over "paper | section" + text,
//     so a chunk that never names its own paper can still match one.
//  3. Re-rank: a cross-encoder reads question and passage together and keeps the best K.

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"regexp"
	"sort"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/ingest"
)

var R = Setting{Name: "R: paper scope, BGE+BM25 top-20 with paper/section prefix, cross-encoder re-rank, k=6",
	ChunkSize: 300, K: 6, Hybrid: true, Candidates: 20, Meta: true, Rerank: true, Scope: true}

// Scorer is the cross-encoder (embed.Reranker); an interface so tests can fake it.
type Scorer interface {
	Scores(ctx context.Context, question string, passages []string) ([]float32, error)
}

// names are how people refer to each paper. shortcut: hand-written for this 3-paper library;
// a larger library needs these derived at ingest (title words, coined terms).
var names = []struct {
	paper string
	re    *regexp.Regexp
}{
	{"2410.00037v1", regexp.MustCompile(`(?i)\b(moshi|mimi|helium|inner monologue)\b`)},
	{"2507.07955v2", regexp.MustCompile(`(?i)\b(h-?net|dynamic chunking)\b`)},
	{"1706.03762v7", regexp.MustCompile(`(?i)\b(transformer|attention is all you need)\b`)},
}

// PaperOf returns the one paper the question names, or "" when it names none or several.
// "Transformer" alone means the Transformer paper; next to Moshi or H-Net it is their component.
func PaperOf(question string) string {
	var found []string
	for _, n := range names {
		if n.re.MatchString(question) {
			found = append(found, n.paper)
		}
	}
	if len(found) == 2 && found[1] == "1706.03762v7" {
		found = found[:1]
	}
	if len(found) == 1 {
		return found[0]
	}
	return ""
}

var titles = map[string]string{}

func init() {
	for _, p := range ingest.Papers {
		titles[p.ID] = p.Title
	}
	titles["2507.07955v2"] = "H-Net: " + titles["2507.07955v2"] // the title never says the model's name
}

// MetaText is what the Meta vectors, the Meta BM25 index and the re-ranker read for a chunk.
// The model still gets the chunk text alone, with its citation label.
func MetaText(c ingest.Chunk) string {
	return fmt.Sprintf("%s | %s %s\n%s", titles[c.Paper], c.SectionNumber, c.SectionPath, c.Text)
}

func metaPath(dir string, size int) string { return fmt.Sprintf("%s/vectors_meta_%d.f32", dir, size) }

// BuildMeta embeds every chunk's MetaText and writes vectors_meta_<size>.f32 (cmd/ingest calls it).
// A chunk whose MetaText would pass the embedder's 512 tokens keeps its plain vector.
func BuildMeta(ctx context.Context, e *embed.Embedder, dir string, size int) error {
	ix, err := Load(dir, size)
	if err != nil {
		return err
	}
	texts := make([]string, len(ix.Chunks))
	for i, c := range ix.Chunks {
		texts[i] = MetaText(c)
	}
	flat := make([]float32, 0, len(texts)*embed.Dim)
	for i, n := range e.CountTokens(texts) {
		v := ix.Vecs[i]
		if n <= embed.MaxTokens {
			vs, err := e.Embed(ctx, texts[i:i+1])
			if err != nil {
				return err
			}
			v = vs[0]
		}
		flat = append(flat, v...)
	}
	buf, err := binary.Append(nil, binary.LittleEndian, flat)
	if err != nil {
		return err
	}
	return os.WriteFile(metaPath(dir, size), buf, 0o644)
}

// Query is one search phrasing: the person's question, the model's rewrite, or a missing part.
type Query struct {
	Text string
	Vec  []float32
}

// SearchFused runs setting R's steps; each is switched by its Setting field. Several phrasings
// share one candidate pool and one re-rank against rerankQ, because the re-rank is the slow step
// (~70 ms per passage on a laptop CPU). scopeText is where a paper name is looked for.
// Hit.Score stays the best plain BGE cosine, so the off-topic threshold means what it meant for A.
func (ix *Index) SearchFused(ctx context.Context, qs []Query, rerankQ, scopeText string, s Setting, rr Scorer) ([]Hit, error) {
	vecs, bm := ix.Vecs, ix.bm25
	if s.Meta {
		if ix.MetaVecs == nil {
			return nil, fmt.Errorf("retrieve: setting %q needs vectors_meta_%d.f32 (run cmd/ingest -meta-only)", s.Name, s.ChunkSize)
		}
		vecs, bm = ix.MetaVecs, ix.bm25meta
	}
	if bm == nil { // an Index built by hand (tests), not by Load
		texts := make([]string, len(ix.Chunks))
		for i, c := range ix.Chunks {
			texts[i] = c.Text
		}
		bm = NewBM25(texts)
	}
	paper := ""
	if s.Scope {
		paper = PaperOf(scopeText)
	}
	var lists [][]int
	for _, q := range qs {
		var sem []int
		cos := make([]float32, len(ix.Chunks))
		for i, v := range vecs {
			if paper == "" || ix.Chunks[i].Paper == paper {
				cos[i] = embed.Cosine(q.Vec, v)
				sem = append(sem, i)
			}
		}
		sort.SliceStable(sem, func(a, b int) bool { return cos[sem[a]] > cos[sem[b]] })
		lists = append(lists, sem[:min(s.Candidates, len(sem))])
		if !s.Hybrid {
			continue
		}
		var lex []int
		for _, r := range bm.Search(q.Text, len(ix.Chunks)) {
			if len(lex) == s.Candidates {
				break
			}
			if paper == "" || ix.Chunks[r.doc].Paper == paper {
				lex = append(lex, r.doc)
			}
		}
		lists = append(lists, lex)
	}
	cands := RRF(lists...)
	if s.Rerank && len(cands) > 0 {
		if rr == nil {
			return nil, fmt.Errorf("retrieve: setting %q needs the re-ranker (run cmd/getmodel)", s.Name)
		}
		cands = cands[:min(2*s.Candidates, len(cands))] // bounds the re-rank time however many phrasings there are
		docs := make([]string, len(cands))
		for i, d := range cands {
			docs[i] = MetaText(ix.Chunks[d])
		}
		sc, err := rr.Scores(ctx, rerankQ, docs)
		if err != nil {
			return nil, err
		}
		order := make([]int, len(cands))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return sc[order[a]] > sc[order[b]] })
		ranked := make([]int, len(cands))
		for i, o := range order {
			ranked[i] = cands[o]
		}
		cands = ranked
	}
	var hits []Hit
	for _, d := range cands[:min(s.K, len(cands))] {
		h := Hit{Chunk: ix.Chunks[d], Score: -1}
		for _, q := range qs {
			h.Score = max(h.Score, embed.Cosine(q.Vec, ix.Vecs[d]))
		}
		hits = append(hits, h)
	}
	return hits, nil
}

// LoadReranker loads the cross-encoder that settings with Rerank need.
func (r *Retriever) LoadReranker() error {
	rr, err := r.Embedder.NewReranker(embed.RerankModelDir)
	if err == nil {
		r.Reranker = rr
	}
	return err
}

// Named are the settings a command line can ask for.
var Named = map[string]Setting{"A": A, "B": B, "C": C, "H10": H10, "H15": H15, "R": R}

// Fused reports whether the setting goes through SearchFused.
func (s Setting) Fused() bool { return s.Meta || s.Rerank || s.Scope }

// RetrieveQueries searches with several phrasings at once (setting R): one pool, one re-rank
// against rerankQ. For other settings it searches each phrasing and keeps the best by cosine.
func (r *Retriever) RetrieveQueries(ctx context.Context, rerankQ, scopeText string, texts []string, s Setting) (Result, error) {
	ix := r.Indexes[s.ChunkSize]
	if ix == nil {
		return Result{}, fmt.Errorf("retrieve: no index loaded for chunk size %d", s.ChunkSize)
	}
	t0 := time.Now()
	var qs []Query
	for _, t := range texts {
		v, err := r.EmbedQuery(ctx, t)
		if err != nil {
			return Result{}, err
		}
		qs = append(qs, Query{t, v})
	}
	t1 := time.Now()
	hits, err := ix.SearchFused(ctx, qs, rerankQ, scopeText, s, r.Reranker)
	return Result{Hits: hits, EmbedMs: float64(t1.Sub(t0).Microseconds()) / 1000, SearchMs: float64(time.Since(t1).Microseconds()) / 1000}, err
}
