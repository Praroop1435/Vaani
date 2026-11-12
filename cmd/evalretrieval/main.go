// Command evalretrieval runs every benchmark question through retrieval settings A, B and C,
// writes one raw row per (question, setting) to results/retrieval_eval.csv, and prints a summary.
// No Groq calls: this measures retrieval only.
//
//	go run -tags ORT ./cmd/evalretrieval            A, B, C -> results/retrieval_eval.csv
//	go run -tags ORT ./cmd/evalretrieval -variants  experiments -> results/retrieval_variants.csv
//	                                                 (A+meta: paper/section prefix embedded with each chunk;
//	                                                  C+rows: one window per table row; ~8 min of embedding)
package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

func main() {
	variants := flag.Bool("variants", false, "run the A+meta and C+rows experiments instead of A/B/C")
	hybrid := flag.Bool("hybrid", false, "run A against the experimental hybrid settings H10 and H15")
	bench := flag.String("bench", "data/eval/benchmark.json", "question file (data/eval/dev.json = the tuning set)")
	fix := flag.Bool("fix", false, "ablation of setting R: A, A with k=6, then each of R's steps")
	rerank := flag.String("reranker", embed.RerankModelDir, "cross-encoder model directory for -fix")
	reps := flag.Int("reps", 1, "repeat each query this many times for the latency percentiles (scoring uses the first)")
	flag.Parse()
	ctx := context.Background()
	b, err := eval.Load(*bench)
	if err != nil {
		log.Fatal(err)
	}
	e, err := embed.New(ctx, embed.DefaultModelDir)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	settings := []retrieve.Setting{retrieve.A, retrieve.B, retrieve.C}
	r := &retrieve.Retriever{Embedder: e, Indexes: map[int]*retrieve.Index{}}
	for _, s := range settings {
		if r.Indexes[s.ChunkSize] == nil {
			if r.Indexes[s.ChunkSize], err = retrieve.Load("data/index", s.ChunkSize); err != nil {
				log.Fatal(err)
			}
		}
	}
	if _, err := r.Retrieve(ctx, "warm-up", retrieve.A); err != nil { // first call loads ONNX kernels
		log.Fatal(err)
	}

	out := "results/retrieval_eval.csv"
	if *variants {
		base := r.Indexes[300]
		r.Indexes = map[int]*retrieve.Index{1300: withMetadata(ctx, e, base), 2300: withTableRows(ctx, e, base)}
		settings = []retrieve.Setting{
			{Name: "A+meta: 300 tok, k=4, paper/section prefix", ChunkSize: 1300, K: 4},
			{Name: "C+rows: windows + 1 window per table row, k=4", ChunkSize: 2300, K: 4, Windows: true},
		}
		out = "results/retrieval_variants.csv"
	}
	if *hybrid {
		settings = []retrieve.Setting{retrieve.A, retrieve.H10, retrieve.H15}
		out = "results/retrieval_hybrid.csv"
	}
	if *fix {
		if r.Reranker, err = e.NewReranker(*rerank); err != nil {
			log.Fatal(err)
		}
		hyb, noScope, noMeta := retrieve.R, retrieve.R, retrieve.R
		hyb.Name, hyb.Rerank = "R without re-rank (RRF order)", false
		noScope.Name, noScope.Scope = "R without paper scope", false
		noMeta.Name, noMeta.Meta = "R without paper/section prefix", false
		settings = []retrieve.Setting{retrieve.A, hyb, noScope, noMeta}
		for _, c := range []int{12, 20} {
			rc := retrieve.R
			rc.Name, rc.Candidates = fmt.Sprintf("R with top-%d from each", c), c
			settings = append(settings, rc)
		}
		out = "results/retrieval_fix_" + strings.TrimSuffix(filepath.Base(*bench), ".json") + ".csv"
	}
	if err := os.MkdirAll("results", 0o755); err != nil {
		log.Fatal(err)
	}
	f, err := os.Create(out)
	if err != nil {
		log.Fatal(err)
	}
	w := csv.NewWriter(f)
	w.Write([]string{"run_at", "qid", "type", "answerable", "setting", "items_total", "items_found", "all_found", "page_ok",
		"first_rank", "top_score", "context_tokens", "embed_ms", "search_ms", "hit_ids"})
	runAt := time.Now().UTC().Format(time.RFC3339)
	all := map[string][]eval.RetrievalScore{}
	searchMs, totalMs := map[string][]float64{}, map[string][]float64{}
	for _, s := range settings {
		for _, q := range b.Questions {
			res, err := r.Retrieve(ctx, q.Question, s)
			if err != nil {
				log.Fatal(err)
			}
			searchMs[s.Name] = append(searchMs[s.Name], res.SearchMs)
			totalMs[s.Name] = append(totalMs[s.Name], res.EmbedMs+res.SearchMs)
			for i := 1; i < *reps; i++ {
				again, err := r.Retrieve(ctx, q.Question, s)
				if err != nil {
					log.Fatal(err)
				}
				searchMs[s.Name] = append(searchMs[s.Name], again.SearchMs)
				totalMs[s.Name] = append(totalMs[s.Name], again.EmbedMs+again.SearchMs)
			}
			sc := eval.ScoreRetrieval(q, s.Name, res)
			all[s.Name] = append(all[s.Name], sc)
			w.Write([]string{runAt, sc.QID, sc.Type, strconv.FormatBool(sc.Answerable), s.Name,
				strconv.Itoa(sc.ItemsTotal), strconv.Itoa(sc.ItemsFound), strconv.FormatBool(sc.AllFound), strconv.FormatBool(sc.PageOK),
				strconv.Itoa(sc.FirstRank), fmt.Sprintf("%.4f", sc.TopScore), strconv.Itoa(sc.ContextTokens),
				fmt.Sprintf("%.2f", sc.EmbedMs), fmt.Sprintf("%.3f", sc.SearchMs), strings.Join(sc.HitIDs, " ")})
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatal(err)
	}
	f.Close()

	fmt.Println("setting | answerable: all evidence found | evidence items found | page ok | tokens/answer | retrieval ms p50/p95 | top score answerable vs unanswerable")
	for _, s := range settings {
		scs := all[s.Name]
		var ans, allFound, items, itemsTot, pageOK, tokens int
		var lat []float64
		var topA, topU []float64
		var missed []string
		for _, sc := range scs {
			tokens += sc.ContextTokens
			lat = append(lat, sc.EmbedMs+sc.SearchMs)
			if !sc.Answerable {
				topU = append(topU, float64(sc.TopScore))
				continue
			}
			ans++
			topA = append(topA, float64(sc.TopScore))
			items += sc.ItemsFound
			itemsTot += sc.ItemsTotal
			if sc.AllFound {
				allFound++
			} else {
				missed = append(missed, fmt.Sprintf("%s(%d/%d)", sc.QID, sc.ItemsFound, sc.ItemsTotal))
			}
			if sc.PageOK {
				pageOK++
			}
		}
		slices.Sort(lat)
		sm, tm := searchMs[s.Name], totalMs[s.Name]
		slices.Sort(sm)
		slices.Sort(tm)
		fmt.Printf("   latency over %d queries: search-only p50 %.3f / p95 %.3f ms; embed+search p50 %.1f / p95 %.1f ms\n",
			len(sm), sm[len(sm)/2], sm[len(sm)*95/100], tm[len(tm)/2], tm[len(tm)*95/100])
		fmt.Printf("%s | %d/%d (%.1f%%) | %d/%d | %d/%d | %d | %.1f / %.1f | %.3f vs %.3f\n   missed: %s\n",
			s.Name, allFound, ans, 100*float64(allFound)/float64(ans), items, itemsTot, pageOK, ans,
			tokens/len(scs), lat[len(lat)/2], lat[len(lat)*95/100], mean(topA), mean(topU), strings.Join(missed, " "))
	}
	fmt.Println("raw rows:", out)
}

var titles = map[string]string{}

func init() {
	for _, p := range ingest.Papers {
		titles[p.ID] = p.Title
	}
}

func embedAll(ctx context.Context, e *embed.Embedder, texts []string) [][]float32 {
	var vecs [][]float32
	for i := 0; i < len(texts); i += 32 {
		v, err := e.Embed(ctx, texts[i:min(i+32, len(texts))])
		if err != nil {
			log.Fatal(err)
		}
		vecs = append(vecs, v...)
	}
	return vecs
}

// withMetadata re-embeds every chunk with "paper title | section" in front (the chunk text the
// model sees is unchanged). Chunks where the prefix would pass 512 tokens keep their old vector.
func withMetadata(ctx context.Context, e *embed.Embedder, base *retrieve.Index) *retrieve.Index {
	texts := make([]string, len(base.Chunks))
	for i, c := range base.Chunks {
		texts[i] = fmt.Sprintf("%s | %s %s\n%s", titles[c.Paper], c.SectionNumber, c.SectionPath, c.Text)
	}
	ix := &retrieve.Index{Chunks: base.Chunks, Vecs: make([][]float32, len(texts))}
	var idx []int
	var fit []string
	for i, n := range e.CountTokens(texts) {
		if n <= embed.MaxTokens {
			idx, fit = append(idx, i), append(fit, texts[i])
		} else {
			ix.Vecs[i] = base.Vecs[i]
		}
	}
	for j, v := range embedAll(ctx, e, fit) {
		ix.Vecs[idx[j]] = v
	}
	fmt.Printf("A+meta: %d/%d chunks re-embedded with prefix\n", len(fit), len(texts))
	return ix
}

// withTableRows keeps C's text windows and replaces each table window with one window per row:
// "Table N: caption | header=cell; header=cell ...", so a single cell can match a question.
func withTableRows(ctx context.Context, e *embed.Embedder, base *retrieve.Index) *retrieve.Index {
	ix := &retrieve.Index{Chunks: base.Chunks, Vecs: base.Vecs}
	var rowTexts []string
	var rowParent []int
	for i, p := range base.WinParent {
		c := base.Chunks[p]
		if c.Kind != "table" {
			ix.WinParent = append(ix.WinParent, p)
			ix.WinVecs = append(ix.WinVecs, base.WinVecs[i])
			continue
		}
		head, body, _ := strings.Cut(c.Text, "\n\n")
		lines := strings.Split(strings.TrimSpace(body), "\n")
		if len(lines) < 3 {
			continue
		}
		header := splitRow(lines[0])
		for _, l := range lines[2:] {
			var cells []string
			for k, v := range splitRow(l) {
				if v != "" && k < len(header) {
					cells = append(cells, strings.TrimSpace(header[k]+" = "+v))
				}
			}
			rowTexts = append(rowTexts, head+" | "+strings.Join(cells, "; "))
			rowParent = append(rowParent, p)
		}
	}
	ix.WinParent = append(ix.WinParent, rowParent...)
	ix.WinVecs = append(ix.WinVecs, embedAll(ctx, e, rowTexts)...)
	fmt.Printf("C+rows: %d table-row windows\n", len(rowTexts))
	return ix
}

func splitRow(line string) []string {
	cells := strings.Split(strings.Trim(line, "| "), " | ")
	for i := range cells {
		cells[i] = strings.TrimSpace(cells[i])
	}
	return cells
}

func mean(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}
