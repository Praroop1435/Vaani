// Command evalcache measures the semantic answer cache's matching on data/eval/cache_pairs.json:
// for each threshold, how many 'same' pairs hit and how many 'different' pairs would wrongly hit,
// with and without the key-term guard. Also times a lookup. No Groq calls.
//
//	go run -tags ORT ./cmd/evalcache    -> results/cache_eval.csv + summary on stdout
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

type pair struct {
	Cached, Asked, Label, Kind string
}

func main() {
	ctx := context.Background()
	js, err := os.ReadFile("data/eval/cache_pairs.json")
	if err != nil {
		log.Fatal(err)
	}
	var f struct{ Pairs []pair }
	if err := json.Unmarshal(js, &f); err != nil {
		log.Fatal(err)
	}
	e, err := embed.New(ctx, embed.DefaultModelDir)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	r := &retrieve.Retriever{Embedder: e}
	vec := map[string][]float32{}
	for _, p := range f.Pairs {
		for _, q := range []string{p.Cached, p.Asked} {
			if vec[q] == nil {
				if vec[q], err = r.EmbedQuery(ctx, q); err != nil {
					log.Fatal(err)
				}
			}
		}
	}

	out, err := os.Create("results/cache_eval.csv")
	if err != nil {
		log.Fatal(err)
	}
	w := csv.NewWriter(out)
	w.Write([]string{"label", "kind", "sim", "terms_equal", "cached_terms", "asked_terms", "cached", "asked"})
	type row struct {
		p     pair
		sim   float32
		terms bool
	}
	var rows []row
	for _, p := range f.Pairs {
		ct, at := pipeline.KeyTerms(p.Cached), pipeline.KeyTerms(p.Asked)
		rw := row{p, embed.Cosine(vec[p.Cached], vec[p.Asked]), slices.Equal(ct, at)}
		rows = append(rows, rw)
		w.Write([]string{p.Label, p.Kind, fmt.Sprintf("%.4f", rw.sim), fmt.Sprint(rw.terms), strings.Join(ct, " "), strings.Join(at, " "), p.Cached, p.Asked})
	}
	w.Flush()
	out.Close()

	same, diff := 0, 0
	for _, rw := range rows {
		if rw.p.Label == "same" {
			same++
		} else {
			diff++
		}
	}
	fmt.Printf("%d pairs: %d same (exact + paraphrase), %d different\n\n", len(rows), same, diff)
	fmt.Println("| threshold | same hit (guard) | different wrongly hit (guard) | same hit (cosine only) | different wrongly hit (cosine only) |")
	fmt.Println("|---|---|---|---|---|")
	for _, th := range []float32{0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98} {
		var sg, dg, sc, dc int
		for _, rw := range rows {
			over := rw.sim >= th
			if rw.p.Label == "same" {
				sc += b2i(over)
				sg += b2i(over && rw.terms)
			} else {
				dc += b2i(over)
				dg += b2i(over && rw.terms)
			}
		}
		fmt.Printf("| %.2f | %d/%d | %d/%d | %d/%d | %d/%d |\n", th, sg, same, dg, diff, sc, same, dc, diff)
	}

	sort.Slice(rows, func(i, j int) bool { return rows[i].sim > rows[j].sim })
	fmt.Println("\nDifferent pairs, highest cosine first (what the guard has to catch):")
	for _, rw := range rows {
		if rw.p.Label == "different" {
			fmt.Printf("  %.3f terms_equal=%-5v %-9s %q vs %q\n", rw.sim, rw.terms, rw.p.Kind, rw.p.Cached, rw.p.Asked)
		}
	}
	fmt.Println("\nSame pairs the guard blocks (safe misses, cost latency only):")
	for _, rw := range rows {
		if rw.p.Label == "same" && !rw.terms {
			fmt.Printf("  %.3f %q vs %q: %v vs %v\n", rw.sim, rw.p.Cached, rw.p.Asked, pipeline.KeyTerms(rw.p.Cached), pipeline.KeyTerms(rw.p.Asked))
		}
	}

	// Lookup cost: embed one question + scan a cache of n entries.
	c := pipeline.NewCache(pipeline.DefaultCacheThreshold)
	v0 := vec[f.Pairs[0].Cached]
	for i := 0; i < 1000; i++ {
		c.Store(v0, "fp", pipeline.Trace{Question: fmt.Sprint("q", i), Status: "ok", Answer: "a", Citations: []pipeline.Citation{{Tag: "S1"}}})
	}
	const n = 20
	var embedMs, scanMs float64
	for i := 0; i < n; i++ {
		t0 := time.Now()
		qv, _ := r.EmbedQuery(ctx, f.Pairs[i%len(f.Pairs)].Asked)
		t1 := time.Now()
		c.Lookup(qv, "x", "fp")
		embedMs += float64(t1.Sub(t0).Microseconds()) / 1000
		scanMs += float64(time.Since(t1).Microseconds()) / 1000
	}
	fmt.Printf("\nLookup cost (mean of %d): embed %.1f ms + scan of 1000 entries %.2f ms\n", n, embedMs/n, scanMs/n)
	fmt.Println("wrote results/cache_eval.csv")
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
