// Command load runs the pipeline under load shapes against the stand-in model service (never
// Groq): steady, burst, sustained overload, per-minute limit halved, daily budget exhausted.
// Arrivals are open-loop (on a schedule, whether or not earlier requests have finished), and the
// pipeline's 20 s deadline is the person giving up. Every request appends one results.csv row
// (source "stand-in") and one raw trace line. The answer cache is never used.
//
//	go run -tags ORT ./cmd/load                            all five, ~35 min
//	go run -tags ORT ./cmd/load -scenarios burst -time 0.1 -csv /tmp/x.csv -out /tmp   dry run, not for the tables
//
// The stand-in's latency is fitted from the real calibration traces; its limits are Groq's free
// tier (30 RPM, 8,000 TPM, 1,000 RPD) except where a scenario changes them without telling the
// pipeline, the way the brief says real limits change.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/fakellm"
	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/results"
	"github.com/Praroop1435/Vaani/internal/retrieve"
	"github.com/Praroop1435/Vaani/internal/standin"
)

// Groq's free tier for openai/gpt-oss-120b as the pipeline is told it (Table 1).
var advertised = governor.Limits{RPM: 30, TPM: 8000, RPD: 1000}

type scenario struct {
	name     string
	what     string
	arrivals []time.Duration // offsets from the scenario start
	tweak    func(*standin.Config)
}

func every(start, end, gap time.Duration) []time.Duration {
	var a []time.Duration
	for t := start; t < end; t += gap {
		a = append(a, t)
	}
	return a
}

func scenarios(scale float64) []scenario {
	m := func(x float64) time.Duration { return time.Duration(x * scale * float64(time.Minute)) }
	burst := every(0, m(6), m(0.5))
	for i := 0; i < 20; i++ { // 20 people at once, 2 minutes in (spread over 5 s)
		burst = append(burst, m(2)+time.Duration(float64(i)*250*scale)*time.Millisecond)
	}
	slices.Sort(burst)
	return []scenario{
		{name: "steady", what: "2 questions/min for 8 min (under the ~3/min the 8K TPM allows)", arrivals: every(0, m(8), m(0.5))},
		{name: "burst", what: "2/min for 6 min, plus 20 questions within 5 s at minute 2", arrivals: burst},
		{name: "overload", what: "8 questions/min for 8 min (~3x what 8K TPM allows)", arrivals: every(0, m(8), m(0.125))},
		{name: "rpm_halved", what: "2/min for 8 min; the stand-in halves RPM and TPM (15, 4,000) without telling the pipeline",
			arrivals: every(0, m(8), m(0.5)), tweak: func(c *standin.Config) { c.RPM, c.TPM = 15, 4000 }},
		{name: "daily_exhausted", what: "4/min for 4 min with 3 requests left in the day's 1,000",
			arrivals: every(0, m(4), m(0.25)), tweak: func(c *standin.Config) { c.RPDUsed = c.RPD - 3 }},
	}
}

func main() {
	which := flag.String("scenarios", "steady,burst,overload,rpm_halved,daily_exhausted", "comma-separated")
	csvPath := flag.String("csv", "results/results.csv", "results.csv to append to")
	calib := flag.String("calib", "results/calibration_qwen/*_traces.jsonl", "real traces the stand-in latency is fitted from")
	// measured: 78 prompt tokens for 30 chars, 5,278 for 23,430 chars (results/calibration/*_probe.jsonl)
	overhead := flag.Float64("prompt-overhead", 71, "provider prompt tokens = overhead + ratio * chars/4")
	ratio := flag.Float64("prompt-ratio", 0.889, "see -prompt-overhead")
	scale := flag.Float64("time", 1, "time scale for dry runs (0.1 = 10x faster); measured runs use 1")
	seed := flag.Uint64("seed", 1, "random seed (question order, stand-in faults)")
	outDir := flag.String("out", "results/load", "directory for raw traces and summaries")
	table3 := flag.Bool("retrieval", false, "instead of load shapes: the frozen 18 once per retrieval setting A, H10, H15, R, one at a time (Table 3)")
	flag.Parse()
	ctx := context.Background()

	paths, _ := filepath.Glob(*calib)
	calls, err := standin.Calls(paths...)
	if err != nil || len(calls) < 5 {
		log.Fatalf("need >= 5 real calls to calibrate from %s: %d, %v", *calib, len(calls), err)
	}
	qs, _, err := eval.LoadFrozen("data/eval/benchmark.json", "data/eval/frozen.json")
	if err != nil {
		log.Fatal(err)
	}
	b, err := eval.Load("data/eval/benchmark.json")
	if err != nil {
		log.Fatal(err)
	}
	oracle := &fakellm.Oracle{Bench: b}
	base := standin.Fit(standin.Config{
		RPM: advertised.RPM, TPM: advertised.TPM, RPD: advertised.RPD, PromptOverhead: *overhead, PromptRatio: *ratio,
		// assumptions, not measured (a free tier cannot be loaded to measure them): Docs/STANDIN.md
		Capacity: 3, SlowPerExtra: 0.3, FailRate: 0.01, StallRate: 0.02, StallMs: 8000, DropRate: 0.005,
		Reply: oracle.Reply,
	}, calls)
	fmt.Printf("stand-in fitted from %d real calls: first token = %.0f ms + %.0f ms per 1K prompt tokens, x lognormal(sigma %.2f); %.1f ms per content token; %d hidden reasoning tokens\n",
		len(calls), base.FirstBaseMs, base.FirstPerKTokMs, base.FirstSigma, base.ContentTokMs, base.ReasoningTok)

	e, err := embed.New(ctx, embed.DefaultModelDir)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	ix, err := retrieve.Load("data/index", 300)
	if err != nil {
		log.Fatal(err)
	}
	ret := &retrieve.Retriever{Embedder: e, Indexes: map[int]*retrieve.Index{300: ix}}
	if err := ret.LoadReranker(); err != nil {
		log.Fatal(err)
	}
	byID := map[string]ingest.Chunk{}
	for _, c := range ix.Chunks {
		byID[c.ID] = c
	}
	rw, err := results.Open(*csvPath)
	if err != nil {
		log.Fatal(err)
	}
	defer rw.Close()
	os.MkdirAll(*outDir, 0o755)
	fmt.Println("answer cache: off (measured run)")

	if *table3 {
		cfg := base
		// Table 3 compares retrieval settings, not reliability: no limits, no injected faults
		cfg.RPM, cfg.TPM, cfg.RPD, cfg.FailRate, cfg.StallRate, cfg.DropRate = 0, 1<<30, 1<<30, 0, 0, 0
		retrievalRun(ctx, cfg, qs, ret, byID, rw, *outDir)
		return
	}
	for i, sc := range scenarios(*scale) {
		if !slices.Contains(strings.Split(*which, ","), sc.name) {
			continue
		}
		cfg := base
		// one random stream per scenario: with one shared seed every scenario replayed the same
		// first draws, and none of them happened to inject a fault (runs of 2026-11-11 21:49-22:29)
		cfg.Seed = *seed*1000 + uint64(i)
		if sc.tweak != nil {
			sc.tweak(&cfg)
		}
		run(ctx, sc, cfg, qs, ret, byID, rw, *scale, *seed, *outDir)
	}
}

type line struct {
	Trace    pipeline.Trace `json:"trace"`
	QID      string         `json:"qid"`
	Grade    string         `json:"answer_grade"`
	CiteOK   bool           `json:"citation_ok"`
	OffsetMs int64          `json:"arrival_offset_ms"`
}

func run(ctx context.Context, sc scenario, cfg standin.Config, qs []eval.Question, ret *retrieve.Retriever,
	byID map[string]ingest.Chunk, rw *results.Writer, scale float64, seed uint64, outDir string) {
	sv := standin.New(cfg)
	srv := httptest.NewServer(sv)
	defer srv.Close()
	gov := governor.New(advertised)
	p := &pipeline.Pipeline{LLM: &llm.Client{BaseURL: srv.URL, Model: "qwen/qwen3.8-27b"}, Gov: gov, Ret: ret, Cfg: pipeline.Defaults}
	p.LLM.Warm(ctx)

	ts := time.Now().Format("20060102-150405")
	scenarioName := sc.name
	if scale != 1 {
		scenarioName += "_dryrun"
	}
	tf, err := os.Create(fmt.Sprintf("%s/%s_%s_traces.jsonl", outDir, ts, scenarioName))
	if err != nil {
		log.Fatal(err)
	}
	defer tf.Close()
	order := rand.New(rand.NewPCG(seed, 7)).Perm(len(qs))
	fmt.Printf("\n=== %s: %s (%d requests, limiter %s)\n", scenarioName, sc.what, len(sc.arrivals), governor.Version)

	var mu sync.Mutex
	var wg sync.WaitGroup
	count := map[string]int{}
	t0 := time.Now()
	for n, at := range sc.arrivals {
		time.Sleep(time.Until(t0.Add(at)))
		q := qs[order[n%len(qs)]]
		wg.Add(1)
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("load-%s-%s-%03d-%s", scenarioName, ts, n, q.ID)
			tr := p.Answer(ctx, id, q.Question, nil)
			var cited []ingest.Chunk
			for _, c := range tr.Citations {
				cited = append(cited, byID[c.ChunkID])
			}
			g := eval.GradeAnswer(q, tr.Answer, tr.Refused, cited)
			js, _ := json.Marshal(line{Trace: tr, QID: q.ID, Grade: g.Grade, CiteOK: g.CiteOK, OffsetMs: at.Milliseconds()})
			mu.Lock()
			defer mu.Unlock()
			tf.Write(append(js, '\n'))
			count[tr.Status]++
			if err := rw.Write(results.FromTrace(tr, scenarioName, "stand-in", results.SettingID(p.Cfg.Setting.Name), g.Grade)); err != nil {
				log.Fatal(err)
			}
		}()
	}
	wg.Wait()
	st := sv.Stats()
	sum := map[string]any{"scenario": scenarioName, "what": sc.what, "requests": len(sc.arrivals), "wall_s": time.Since(t0).Seconds(),
		"status": count, "governor_peak_waiting": gov.PeakWaiting(), "standin": st,
		"standin_config": map[string]any{"rpm": cfg.RPM, "tpm": cfg.TPM, "rpd": cfg.RPD, "rpd_used_at_start": cfg.RPDUsed,
			"first_base_ms": cfg.FirstBaseMs, "first_per_1k_prompt_ms": cfg.FirstPerKTokMs, "first_sigma": cfg.FirstSigma,
			"content_tok_ms": cfg.ContentTokMs, "reasoning_tok": cfg.ReasoningTok, "prompt_overhead": cfg.PromptOverhead, "prompt_ratio": cfg.PromptRatio,
			"capacity": cfg.Capacity, "slow_per_extra": cfg.SlowPerExtra, "fail_rate": cfg.FailRate,
			"stall_rate": cfg.StallRate, "stall_ms": cfg.StallMs, "drop_rate": cfg.DropRate, "seed": cfg.Seed},
		"pipeline_told": advertised, "limiter": governor.Version}
	js, _ := json.MarshalIndent(sum, "", "  ")
	os.WriteFile(fmt.Sprintf("%s/%s_%s_summary.json", outDir, ts, scenarioName), js, 0o644)
	fmt.Printf("status %v | governor peak queue %d | stand-in: %d requests, %d ok, %d limited, %d 500s, %d stalled, %d cut, peak in flight %d\n",
		count, gov.PeakWaiting(), st.Requests, st.OK, st.Limited429, st.Failed500, st.Stalled, st.Dropped, st.PeakInFlight)
}

// retrievalRun asks each frozen question once per setting, one request at a time, on the stand-in.
func retrievalRun(ctx context.Context, cfg standin.Config, qs []eval.Question, ret *retrieve.Retriever,
	byID map[string]ingest.Chunk, rw *results.Writer, outDir string) {
	srv := httptest.NewServer(standin.New(cfg))
	defer srv.Close()
	ts := time.Now().Format("20060102-150405")
	tf, err := os.Create(fmt.Sprintf("%s/%s_retrieval_standin_traces.jsonl", outDir, ts))
	if err != nil {
		log.Fatal(err)
	}
	defer tf.Close()
	for _, s := range []retrieve.Setting{retrieve.A, retrieve.H10, retrieve.H15, retrieve.R} {
		p := &pipeline.Pipeline{LLM: &llm.Client{BaseURL: srv.URL, Model: "qwen/qwen3.8-27b"},
			Gov: governor.New(governor.Limits{TPM: 1 << 30, RPD: 1 << 30}), Ret: ret, Cfg: pipeline.Defaults}
		p.Cfg.Setting = s
		count := map[string]int{}
		for _, q := range qs {
			tr := p.Answer(ctx, fmt.Sprintf("retrieval_standin-%s-%s-%s", ts, results.SettingID(s.Name), q.ID), q.Question, nil)
			var cited []ingest.Chunk
			for _, c := range tr.Citations {
				cited = append(cited, byID[c.ChunkID])
			}
			g := eval.GradeAnswer(q, tr.Answer, tr.Refused, cited)
			js, _ := json.Marshal(line{Trace: tr, QID: q.ID, Grade: g.Grade, CiteOK: g.CiteOK})
			tf.Write(append(js, '\n'))
			if err := rw.Write(results.FromTrace(tr, "retrieval_standin", "stand-in", results.SettingID(s.Name), g.Grade)); err != nil {
				log.Fatal(err)
			}
			count[g.Grade]++
		}
		fmt.Printf("%s: %v\n", s.Name, count)
	}
}
