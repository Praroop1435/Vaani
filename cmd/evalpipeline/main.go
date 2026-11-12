// Command evalpipeline runs the whole benchmark through the full pipeline against the offline
// oracle model (internal/fakellm): no Groq tokens. The oracle reads perfectly and never guesses,
// so this measures retrieval + completeness + targeted retrieval + citations + grading with the
// model held perfect. Model mistakes (e.g. substituting a related figure) need real Groq runs.
//
//	go run -tags ORT ./cmd/evalpipeline            settings A, B, C
//
// Writes results/pipeline_offline_<ts>.csv (results.csv columns plus extras) and _traces.jsonl.
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/fakellm"
	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

func main() {
	style := flag.String("style", "", `oracle draft format: "" ([P1] as asked), "colon", "mixed" (what gpt-oss wrote live)`)
	hybrid := flag.Bool("hybrid", false, "compare A with the experimental hybrid settings H10, H15 instead of A, B, C")
	bench := flag.String("bench", "data/eval/benchmark.json", "question file (data/eval/dev.json = the tuning set)")
	fix := flag.Bool("fix", false, "compare A with setting R")
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
	srv := httptest.NewServer(&fakellm.Oracle{Bench: b, Style: *style})
	defer srv.Close()

	ts := time.Now().Format("20060102-150405")
	os.MkdirAll("results", 0o755)
	name := "results/pipeline_offline_" + ts
	if *style != "" {
		name += "_" + *style
	}
	header := []string{"request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
		"tokens_in", "tokens_out", "retrieval_setting", "answer_grade", "qid", "type", "citation_ok", "model_calls",
		"targeted_chunks", "missing_parts", "dropped", "notes"}

	fmt.Println("setting | correct | wrong | refused ok | refused wrongly | citation ok | targeted rounds | model calls/q | answer")
	settings := []retrieve.Setting{retrieve.A, retrieve.B, retrieve.C}
	if *hybrid {
		settings = []retrieve.Setting{retrieve.A, retrieve.H10, retrieve.H15}
		name += "_hybrid"
	}
	if *fix {
		settings = []retrieve.Setting{retrieve.A, retrieve.R}
		name += "_fix_" + strings.TrimSuffix(filepath.Base(*bench), ".json")
	}
	rr, err := e.NewReranker(embed.RerankModelDir)
	if err != nil {
		log.Fatal(err)
	}
	cf, _ := os.Create(name + ".csv")
	jf, _ := os.Create(name + "_traces.jsonl")
	defer cf.Close()
	defer jf.Close()
	w := csv.NewWriter(cf)
	w.Write(header)
	for _, s := range settings {
		ix, err := retrieve.Load("data/index", s.ChunkSize)
		if err != nil {
			log.Fatal(err)
		}
		byID := map[string]ingest.Chunk{}
		for _, c := range ix.Chunks {
			byID[c.ID] = c
		}
		cfg := pipeline.Defaults
		cfg.Setting = s
		p := &pipeline.Pipeline{
			LLM: &llm.Client{BaseURL: srv.URL, Model: "oracle"},
			Gov: governor.New(governor.Limits{RPM: 100000, TPM: 100_000_000, RPD: 1_000_000}),
			Ret: &retrieve.Retriever{Embedder: e, Indexes: map[int]*retrieve.Index{s.ChunkSize: ix}, Reranker: rr},
			Cfg: cfg,
		}
		count := map[string]int{}
		citeOK, targeted, calls, ans := 0, 0, 0, 0
		var wrong []string
		for _, q := range b.Questions {
			id := fmt.Sprintf("%s-%s", strings.SplitN(s.Name, ":", 2)[0], q.ID)
			tr := p.Answer(ctx, id, q.Question, nil)
			var cited []ingest.Chunk
			for _, c := range tr.Citations {
				cited = append(cited, byID[c.ChunkID])
			}
			g := eval.GradeAnswer(q, tr.Answer, tr.Refused, cited)
			count[g.Grade]++
			calls += tr.ModelCalls
			if len(tr.Targeted) > 0 {
				targeted++
			}
			if q.Answerable {
				ans++
				if g.CiteOK {
					citeOK++
				}
			}
			if g.Grade == "wrong" || g.Grade == "refused_wrongly" || (q.Answerable && !g.CiteOK) {
				wrong = append(wrong, fmt.Sprintf("%s:%s(%s)", q.ID, g.Grade, strings.Join(g.Notes, "; ")))
			}
			w.Write([]string{id, "benchmark_offline", "fake-oracle" + map[bool]string{true: "-" + *style}[*style != ""], tr.Start.UTC().Format(time.RFC3339Nano),
				strconv.FormatInt(tr.FirstTokenMs, 10), strconv.FormatInt(tr.TotalMs, 10), tr.Status, strconv.Itoa(tr.Retries),
				strconv.Itoa(tr.TokensIn), strconv.Itoa(tr.TokensOut), s.Name, g.Grade, q.ID, q.Type, strconv.FormatBool(g.CiteOK),
				strconv.Itoa(tr.ModelCalls), strings.Join(tr.Targeted, " "), strings.Join(tr.Missing, " | "), strconv.Itoa(len(tr.Dropped)),
				strings.Join(g.Notes, "; ")})
			js, _ := json.Marshal(map[string]any{"setting": s.Name, "qid": q.ID, "grade": g, "trace": tr})
			jf.Write(append(js, '\n'))
		}
		n := len(b.Questions)
		pct := func(k int) string { return fmt.Sprintf("%d (%.1f%%)", k, 100*float64(k)/float64(n)) }
		fmt.Printf("%s | %s | %s | %s | %s | %d/%d | %d | %.2f\n   problems: %s\n", s.Name, pct(count["correct"]), pct(count["wrong"]),
			pct(count["refused_correctly"]), pct(count["refused_wrongly"]), citeOK, ans, targeted, float64(calls)/float64(n), strings.Join(wrong, " "))
	}
	w.Flush()
	fmt.Println("raw:", cf.Name(), jf.Name())
}
