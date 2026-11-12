package retrieve

// Experimental hybrid retrieval: BGE semantic ranking + BM25 lexical ranking, fused with
// Reciprocal Rank Fusion. Not the default (A is) until the benchmark shows a measurable gain.
//
// Why try it: the benchmark's persistent misses hinge on exact tokens a dense vector blurs:
// "H100" (q12), "FineWeb-Edu" / "100B" (q17), "h = 8" in a formula-heavy chunk (q01).

import (
	"math"
	"sort"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/textnorm"
)

var (
	H10 = Setting{Name: "H10: BGE top-10 + BM25 top-10, RRF -> k=4", ChunkSize: 300, K: 4, Hybrid: true, Candidates: 10}
	H15 = Setting{Name: "H15: BGE top-15 + BM25 top-15, RRF -> k=4", ChunkSize: 300, K: 4, Hybrid: true, Candidates: 15}
)

const (
	bm25K1 = 1.2 // standard Okapi values; not tuned on the benchmark
	bm25B  = 0.75
	rrfK   = 60 // RRF constant from Cormack et al. 2009; also not tuned
)

// BM25 is an in-memory Okapi BM25 index over chunk text (376 chunks: built in ~ms at load).
type BM25 struct {
	tf    []map[string]int
	len   []int
	df    map[string]int
	avgdl float64
}

func NewBM25(texts []string) *BM25 {
	b := &BM25{df: map[string]int{}}
	total := 0
	for _, t := range texts {
		tf := map[string]int{}
		toks := textnorm.Tokens(t)
		for _, w := range toks {
			tf[w]++
		}
		for w := range tf {
			b.df[w]++
		}
		b.tf = append(b.tf, tf)
		b.len = append(b.len, len(toks))
		total += len(toks)
	}
	if len(texts) > 0 {
		b.avgdl = float64(total) / float64(len(texts))
	}
	return b
}

type ranked struct {
	doc   int
	score float64
}

// Search returns up to k documents by BM25 score, best first; documents scoring 0 are left out.
func (b *BM25) Search(query string, k int) []ranked {
	n := float64(len(b.tf))
	q := map[string]bool{}
	for _, w := range textnorm.Tokens(query) {
		q[w] = true
	}
	var out []ranked
	for d, tf := range b.tf {
		s := 0.0
		for w := range q {
			f := float64(tf[w])
			if f == 0 {
				continue
			}
			df := float64(b.df[w])
			idf := math.Log(1 + (n-df+0.5)/(df+0.5))
			s += idf * f * (bm25K1 + 1) / (f + bm25K1*(1-bm25B+bm25B*float64(b.len[d])/b.avgdl))
		}
		if s > 0 {
			out = append(out, ranked{d, s})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].score > out[j].score })
	return out[:min(k, len(out))]
}

// RRF fuses ranked lists: score(d) = Σ 1/(rrfK + rank), rank from 1. Ties keep the first list's order.
func RRF(lists ...[]int) []int {
	score := map[int]float64{}
	first := map[int]int{}
	order := 0
	for _, l := range lists {
		for r, d := range l {
			score[d] += 1 / float64(rrfK+r+1)
			if _, ok := first[d]; !ok {
				first[d] = order
				order++
			}
		}
	}
	docs := make([]int, 0, len(score))
	for d := range score {
		docs = append(docs, d)
	}
	sort.Slice(docs, func(i, j int) bool {
		if score[docs[i]] != score[docs[j]] {
			return score[docs[i]] > score[docs[j]]
		}
		return first[docs[i]] < first[docs[j]]
	})
	return docs
}

// SearchHybrid takes the top Candidates from BGE and from BM25, fuses them with RRF, and returns
// the top K. Hit.Score stays the BGE cosine, so the pipeline's off-topic threshold means the same.
func (ix *Index) SearchHybrid(qv []float32, question string, s Setting) []Hit {
	if ix.bm25 == nil { // an Index built by hand (tests), not by Load
		texts := make([]string, len(ix.Chunks))
		for i, c := range ix.Chunks {
			texts[i] = c.Text
		}
		ix.bm25 = NewBM25(texts)
	}
	cos := make([]float32, len(ix.Chunks))
	sem := make([]int, len(ix.Chunks))
	for i, v := range ix.Vecs {
		cos[i], sem[i] = embed.Cosine(qv, v), i
	}
	sort.SliceStable(sem, func(a, b int) bool { return cos[sem[a]] > cos[sem[b]] })
	sem = sem[:min(s.Candidates, len(sem))]
	var lex []int
	for _, r := range ix.bm25.Search(question, s.Candidates) {
		lex = append(lex, r.doc)
	}
	var hits []Hit
	for _, d := range RRF(sem, lex) {
		if len(hits) == s.K {
			break
		}
		hits = append(hits, Hit{Chunk: ix.Chunks[d], Score: cos[d]})
	}
	return hits
}

// Doc and Score expose a BM25 result (for tests and diagnostics).
func (r ranked) Doc() int       { return r.doc }
func (r ranked) Score() float64 { return r.score }
