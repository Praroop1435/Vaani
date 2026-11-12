// Package retrieve is the pipeline's search tool: BGE cosine top-k over the chunk index.
// Brute force: ~400 vectors x 768 dims is well under a millisecond, so no vector DB.
package retrieve

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/ingest"
)

// Setting is one retrieval configuration compared in Table 3 of the write-up.
type Setting struct {
	Name      string
	ChunkSize int // which index: data/index/chunks_<size>.json
	K         int
	Windows   bool // small-to-big: search sentence windows, return their parent chunks
	// Hybrid (experimental, hybrid.go): fuse the BGE top-Candidates and BM25 top-Candidates with RRF.
	Hybrid     bool
	Candidates int
	// Setting R (fused.go): each step can be switched on its own for the ablation.
	Meta   bool // search vectors and BM25 built from "paper | section" + text
	Rerank bool // cross-encoder re-rank of the fused candidates
	Scope  bool // a question naming one paper searches only that paper
}

var (
	A = Setting{Name: "A: 300 tok, k=4", ChunkSize: 300, K: 4}
	B = Setting{Name: "B: 480 tok, k=3", ChunkSize: 480, K: 3}
	C = Setting{Name: "C: 2-sentence windows -> 300 tok parents, k=4", ChunkSize: 300, K: 4, Windows: true}
)

type Index struct {
	Chunks []ingest.Chunk
	Vecs   [][]float32 // unit length, same order as Chunks
	// Optional small-to-big layer: windows_<size>.json/.f32. WinParent[i] indexes Chunks.
	WinParent []int
	WinVecs   [][]float32
	Hash      string      // sha256 of the index files read by Load; changes when cmd/ingest re-runs
	bm25      *BM25       // lexical index for hybrid settings
	MetaVecs  [][]float32 // optional vectors_meta_<size>.f32, same order as Chunks
	bm25meta  *BM25
}

// Load reads chunks_<size>.json and vectors_<size>.f32 written by cmd/ingest.
func Load(dir string, size int) (*Index, error) {
	js, err := os.ReadFile(fmt.Sprintf("%s/chunks_%d.json", dir, size))
	if err != nil {
		return nil, err
	}
	var chunks []ingest.Chunk
	if err := json.Unmarshal(js, &chunks); err != nil {
		return nil, err
	}
	vecs, err := readVectors(fmt.Sprintf("%s/vectors_%d.f32", dir, size), len(chunks))
	if err != nil {
		return nil, err
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	ix := &Index{Chunks: chunks, Vecs: vecs, bm25: NewBM25(texts)} // built here, not lazily: concurrent searches
	h := sha256.New()
	h.Write(js)
	for _, f := range []string{"vectors_%d.f32", "windows_%d.json", "windows_%d.f32", "vectors_meta_%d.f32"} {
		if b, err := os.ReadFile(fmt.Sprintf("%s/"+f, dir, size)); err == nil {
			h.Write(b)
		}
	}
	if mv, err := readVectors(metaPath(dir, size), len(chunks)); err == nil {
		meta := make([]string, len(chunks))
		for i, c := range chunks {
			meta[i] = MetaText(c)
		}
		ix.MetaVecs, ix.bm25meta = mv, NewBM25(meta)
	}
	ix.Hash = hex.EncodeToString(h.Sum(nil))
	if wjs, err := os.ReadFile(fmt.Sprintf("%s/windows_%d.json", dir, size)); err == nil {
		var ws []ingest.Window
		if err := json.Unmarshal(wjs, &ws); err != nil {
			return nil, err
		}
		if ix.WinVecs, err = readVectors(fmt.Sprintf("%s/windows_%d.f32", dir, size), len(ws)); err != nil {
			return nil, err
		}
		for _, w := range ws {
			if w.Parent < 0 || w.Parent >= len(chunks) {
				return nil, fmt.Errorf("retrieve: window parent %d out of range: re-run cmd/ingest", w.Parent)
			}
			ix.WinParent = append(ix.WinParent, w.Parent)
		}
	}
	return ix, nil
}

func readVectors(path string, n int) ([][]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if want := n * embed.Dim * 4; len(raw) != want {
		return nil, fmt.Errorf("retrieve: %s has %d bytes, want %d for %d rows: re-run cmd/ingest", path, len(raw), want, n)
	}
	flat := make([]float32, n*embed.Dim)
	if _, err := binary.Decode(raw, binary.LittleEndian, flat); err != nil {
		return nil, err
	}
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = flat[i*embed.Dim : (i+1)*embed.Dim]
	}
	return vecs, nil
}

type Hit struct {
	Chunk ingest.Chunk
	Score float32 // cosine similarity
}

// Search returns the k chunks most similar to q, best first.
func (ix *Index) Search(q []float32, k int) []Hit {
	hits := make([]Hit, len(ix.Chunks))
	for i, v := range ix.Vecs {
		hits[i] = Hit{ix.Chunks[i], embed.Cosine(q, v)}
	}
	sort.SliceStable(hits, func(a, b int) bool { return hits[a].Score > hits[b].Score })
	return hits[:min(k, len(hits))]
}

// SearchWindows ranks windows, keeps each parent chunk's best window score, and returns the
// top k parents. The model still sees whole chunks, so tokens per answer match setting A.
func (ix *Index) SearchWindows(q []float32, k int) []Hit {
	best := map[int]float32{}
	for i, v := range ix.WinVecs {
		p, s := ix.WinParent[i], embed.Cosine(q, v)
		if cur, ok := best[p]; !ok || s > cur {
			best[p] = s
		}
	}
	hits := make([]Hit, 0, len(best))
	for p, s := range best {
		hits = append(hits, Hit{ix.Chunks[p], s})
	}
	sort.SliceStable(hits, func(a, b int) bool {
		if hits[a].Score != hits[b].Score {
			return hits[a].Score > hits[b].Score
		}
		return hits[a].Chunk.ID < hits[b].Chunk.ID // map order is random: keep ties stable
	})
	return hits[:min(k, len(hits))]
}

// Result is what the pipeline gets back from the retrieve tool, with timings for the trace.
type Result struct {
	Hits     []Hit
	EmbedMs  float64
	SearchMs float64
}

type Retriever struct {
	Embedder *embed.Embedder
	Indexes  map[int]*Index // by chunk size
	Reranker Scorer         // needed by settings with Rerank
}

// Retrieve embeds the question (with BGE's query prefix) and searches the setting's index.
func (r *Retriever) Retrieve(ctx context.Context, question string, s Setting) (Result, error) {
	if r.Indexes[s.ChunkSize] == nil {
		return Result{}, fmt.Errorf("retrieve: no index loaded for chunk size %d", s.ChunkSize)
	}
	t0 := time.Now()
	qv, err := r.EmbedQuery(ctx, question)
	if err != nil {
		return Result{}, err
	}
	t1 := time.Now()
	var hits []Hit
	if s.Fused() {
		hits, err = r.Indexes[s.ChunkSize].SearchFused(ctx, []Query{{question, qv}}, question, question, s, r.Reranker)
	} else if s.Hybrid {
		hits = r.Indexes[s.ChunkSize].SearchHybrid(qv, question, s) // BM25 needs the question text
	} else {
		hits, err = r.Search(qv, s)
	}
	return Result{
		Hits:     hits,
		EmbedMs:  float64(t1.Sub(t0).Microseconds()) / 1000,
		SearchMs: float64(time.Since(t1).Microseconds()) / 1000,
	}, err
}

// EmbedQuery is the query vector Retrieve uses; the answer cache shares it so a miss embeds once.
func (r *Retriever) EmbedQuery(ctx context.Context, question string) ([]float32, error) {
	qv, err := r.Embedder.Embed(ctx, []string{embed.QueryPrefix + question})
	if err != nil {
		return nil, err
	}
	return qv[0], nil
}

// Search ranks the setting's index against an already embedded query.
func (r *Retriever) Search(qv []float32, s Setting) ([]Hit, error) {
	ix := r.Indexes[s.ChunkSize]
	if ix == nil {
		return nil, fmt.Errorf("retrieve: no index loaded for chunk size %d", s.ChunkSize)
	}
	if s.Hybrid {
		return nil, fmt.Errorf("retrieve: hybrid setting %q needs the question text: use Retrieve", s.Name)
	}
	if s.Windows {
		if len(ix.WinVecs) == 0 {
			return nil, fmt.Errorf("retrieve: setting %q needs windows_%d.* (run cmd/ingest -windows-only)", s.Name, s.ChunkSize)
		}
		return ix.SearchWindows(qv, s.K), nil
	}
	return ix.Search(qv, s.K), nil
}
