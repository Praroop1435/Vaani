// Command smoke runs benchmark questions through the full pipeline against real Groq, one at a
// time and paced, under a hard token ceiling. It keeps every raw trace and appends one
// results.csv row per request (source "real"). The answer cache is never used: Answer, not AnswerCached.
//
//	go run -tags ORT ./cmd/smoke -ids q05,q14,q19 -max-tokens 15000
//	go run -tags ORT ./cmd/smoke -set quality -pace 30s -max-tokens 60000   frozen 15, paced
//
// Before each question it checks that the worst case for that question (understand + draft +
// one re-draft at their max sizes) still fits under the ceiling; otherwise it stops.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/results"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

type graded struct {
	Trace     pipeline.Trace `json:"trace"`
	QID       string         `json:"qid"`
	Type      string         `json:"type"`
	Expected  string         `json:"expected"`
	Grade     string         `json:"answer_grade"` // correct, wrong, refused_correctly, refused_wrongly
	CiteOK    bool           `json:"citation_ok"`  // every evidence item is in a cited chunk on the right page
	CiteNotes []string       `json:"citation_notes"`
}

func main() {
	ids := flag.String("ids", "q05,q14,q19", "benchmark question ids (ignored with -set)")
	bench := flag.String("bench", "data/eval/benchmark.json", "question file for -ids (data/eval/dev.json = the tuning set; \"all\" as -ids takes every question)")
	setting := flag.String("setting", "R", "retrieval setting: A or R")
	model := flag.String("model", "qwen/qwen3.8-27b", "Groq model id")
	set := flag.String("set", "", `"quality" = the frozen 15, "unanswerable" = the frozen 3, "frozen" = all 18`)
	ceiling := flag.Int("max-tokens", 15000, "hard ceiling on Groq tokens (in + out) for the whole run")
	pace := flag.Duration("pace", 0, "wait between questions (30s keeps ~2 questions/min under the 8K tokens/min)")
	scenario := flag.String("scenario", "smoke", "results.csv scenario")
	csvPath := flag.String("csv", "results/results.csv", "results.csv to append to (empty: none)")
	outDir := flag.String("out", "results/smoke", "directory for raw traces")
	doProbe := flag.Bool("probe", false, "after the questions, wait 65 s and run the 3-call limit probe (see probe)")
	pad := flag.Int("probe-pad", 0, "pad each of the probe's two simultaneous prompts with ~N tokens of filler")
	flag.Parse()
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		key = envFile(".env", "GROQ_API_KEY")
	}
	if key == "" {
		log.Fatal("GROQ_API_KEY not set and not in .env")
	}
	var questions []eval.Question
	if *set != "" {
		qs, fz, err := eval.LoadFrozen("data/eval/benchmark.json", "data/eval/frozen.json")
		if err != nil {
			log.Fatal(err)
		}
		switch *set {
		case "quality":
			qs = qs[:len(fz.Quality)]
		case "unanswerable":
			qs = qs[len(fz.Quality):]
		case "frozen":
		default:
			log.Fatalf("unknown -set %q", *set)
		}
		questions = qs
	} else if *ids != "" {
		b, err := eval.Load(*bench)
		if err != nil {
			log.Fatal(err)
		}
		if *ids == "all" {
			questions = b.Questions
			*ids = ""
		}
		for _, id := range strings.Fields(strings.ReplaceAll(*ids, ",", " ")) {
			i := slices.IndexFunc(b.Questions, func(q eval.Question) bool { return q.ID == id })
			if i < 0 {
				log.Fatalf("no question %s", id)
			}
			questions = append(questions, b.Questions[i])
		}
	}
	ctx := context.Background()
	e, err := embed.New(ctx, embed.DefaultModelDir)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	ix, err := retrieve.Load("data/index", 300)
	if err != nil {
		log.Fatal(err)
	}
	byID := map[string]ingest.Chunk{}
	for _, c := range ix.Chunks {
		byID[c.ID] = c
	}
	cfg := pipeline.Defaults
	cfg.Setting = retrieve.Named[*setting]
	ret := &retrieve.Retriever{Embedder: e, Indexes: map[int]*retrieve.Index{300: ix}}
	if err := ret.LoadReranker(); err != nil {
		log.Fatal(err)
	}
	p := &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: llm.GroqBaseURL, APIKey: key, Model: *model},
		Gov: governor.New(governor.Limits{RPM: 30, TPM: 8000, RPD: 1000}),
		Ret: ret,
		Cfg: cfg,
	}
	// worst case per question: understand (≈200 in + max out) + draft (4 passages ≈1600 in + max out)
	// + targeted draft (up to 8 passages ≈2800 in + max out)
	worst := 200 + cfg.UnderstandMax + 1600 + cfg.DraftMaxTokens + 2800 + cfg.DraftMaxTokens

	if code, d, err := p.LLM.Warm(ctx); err != nil {
		fmt.Println("warm-up failed (continuing cold):", err)
	} else {
		fmt.Printf("warm-up: HTTP %d in %s (unauthenticated, no quota)\n", code, d.Round(time.Millisecond))
	}
	ts := time.Now().Format("20060102-150405")
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatal(err)
	}
	out, err := os.Create(*outDir + "/" + ts + "_traces.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	var rw *results.Writer
	if *csvPath != "" {
		if rw, err = results.Open(*csvPath); err != nil {
			log.Fatal(err)
		}
		defer rw.Close()
	}
	fmt.Println("answer cache: off (measured run)")

	used := 0
	for n, q := range questions {
		if used+worst > *ceiling {
			fmt.Printf("STOP before %s: %d used + %d worst case > %d ceiling\n", q.ID, used, worst, *ceiling)
			break
		}
		if n > 0 && *pace > 0 {
			time.Sleep(*pace)
		}
		fmt.Printf("\n=== %s (%s): %s\n> ", q.ID, q.Type, q.Question)
		tr := p.Answer(ctx, fmt.Sprintf("%s-%s-%s", *scenario, ts, q.ID), q.Question, func(s string) { fmt.Print(s) })
		fmt.Println()
		used += tr.TokensIn + tr.TokensOut
		g := grade(q, tr, byID)
		js, _ := json.Marshal(g)
		out.Write(append(js, '\n'))
		if rw != nil {
			if err := rw.Write(results.FromTrace(tr, *scenario, "real", results.SettingID(cfg.Setting.Name), g.Grade)); err != nil {
				log.Fatal(err)
			}
		}
		fmt.Printf("    status=%s grade=%s citation_ok=%v ttft=%dms total=%dms tokens=%d in / %d out retries=%d dropped=%d degraded=%v\n",
			tr.Status, g.Grade, g.CiteOK, tr.FirstTokenMs, tr.TotalMs, tr.TokensIn, tr.TokensOut, tr.Retries, len(tr.Dropped), tr.Degraded)
		for _, s := range tr.Steps {
			fmt.Printf("    step %-10s wait=%6.0fms work=%6.0fms retry=%5.0fms retries=%d first=%dms tok=%d/%d %s\n",
				s.Name, s.WaitMs, s.WorkMs, s.RetryMs, s.Retries, s.FirstMs, s.TokensIn, s.TokensOut, s.Note)
		}
		for _, d := range tr.Dropped {
			fmt.Println("    DROPPED:", d)
		}
		for _, n := range g.CiteNotes {
			fmt.Println("    cite:", n)
		}
	}
	fmt.Printf("\ntokens used: %d of %d ceiling\nraw traces: %s\n", used, *ceiling, out.Name())
	if *doProbe {
		time.Sleep(65 * time.Second) // keep <= 3 calls in any minute
		probe(ctx, p.LLM, *outDir+"/"+ts+"_probe.jsonl", rw, *scenario, ts, *pad)
	}
}

type probeCall struct {
	Name         string            `json:"name"`
	Start        time.Time         `json:"start_time"`
	MaxTokens    int               `json:"max_completion_tokens"`
	Status       int               `json:"http_status"`
	FirstTokenMs int64             `json:"first_token_ms"`
	TotalMs      int64             `json:"total_ms"`
	TokensIn     int               `json:"tokens_in"`
	TokensOut    int               `json:"tokens_out"`
	RetryAfter   string            `json:"retry_after,omitempty"`
	Body         string            `json:"body,omitempty"`
	Headers      map[string]string `json:"headers"`
}

// probe learns what Groq does at its per-minute token limit with 3 cheap calls (<= 3 in the minute):
// two at once, each reserving 4,500 completion tokens (2 x ~4,530 > the 8,000 TPM, while the reply
// itself is one word), then, once retry-after has passed, one small call to time the recovery.
//
// Run 1 (pad 0) was admitted twice with ~149 tokens charged each, so Groq does not reserve a large
// max_completion_tokens in full. Run 2 pads each prompt (pad 5200: 2 x ~5,200 > 8,000) because
// prompt tokens are certainly charged.
func probe(ctx context.Context, c *llm.Client, path string, rw *results.Writer, scenario, ts string, pad int) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	msgs := []llm.Message{{Role: "user", Content: strings.Repeat("The quick brown fox jumps over the lazy dog. ", pad/10) + "Reply with the single word OK."}}
	one := func(name string, maxTok int) probeCall {
		pc := probeCall{Name: name, Start: time.Now(), MaxTokens: maxTok, Status: 200, Headers: map[string]string{}}
		r, err := c.ChatStream(ctx, msgs, llm.Options{MaxTokens: maxTok})
		pc.FirstTokenMs, pc.TotalMs, pc.TokensIn, pc.TokensOut = r.FirstTokenMs, r.TotalMs, r.TokensIn, r.TokensOut
		var ae *llm.APIError
		if errors.As(err, &ae) {
			pc.Status, pc.Body, pc.RetryAfter = ae.Status, ae.Body, ae.RetryAfter.String()
		} else if err != nil {
			pc.Status, pc.Body = 0, err.Error()
		}
		for k, v := range r.Header {
			if k = strings.ToLower(k); strings.HasPrefix(k, "x-ratelimit-") || k == "retry-after" || k == "x-groq-region" {
				pc.Headers[k] = strings.Join(v, ",")
			}
		}
		return pc
	}
	var wg sync.WaitGroup
	pair := make([]probeCall, 2)
	for i := range pair {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pair[i] = one(fmt.Sprintf("tpm_pair_%d", i+1), map[bool]int{true: 4500, false: 50}[pad == 0])
		}()
	}
	wg.Wait()
	calls := pair
	wait := 10 * time.Second
	for _, pc := range pair {
		if d, err := time.ParseDuration(pc.RetryAfter); err == nil && pc.Status == 429 {
			wait = max(d, time.Second) + 500*time.Millisecond
		}
	}
	time.Sleep(wait)
	msgs = []llm.Message{{Role: "user", Content: "Reply with the single word OK."}} // recovery is a small call
	rec := one("recovery", 50)
	rec.Name = fmt.Sprintf("recovery_after_%s", wait)
	calls = append(calls, rec)
	for _, pc := range calls {
		js, _ := json.Marshal(pc)
		f.Write(append(js, '\n'))
		fmt.Printf("probe %-22s HTTP %d first=%dms total=%dms tok=%d/%d retry-after=%s %s\n", pc.Name, pc.Status,
			pc.FirstTokenMs, pc.TotalMs, pc.TokensIn, pc.TokensOut, pc.RetryAfter, pc.Body)
		if rw != nil {
			st := "ok"
			switch {
			case pc.Status == 429:
				st = "rate_limited"
			case pc.Status != 200:
				st = "error"
			}
			rw.Write(results.Row{RequestID: fmt.Sprintf("%s-%s-%s", scenario, ts, pc.Name), Scenario: scenario + "_probe", Source: "real",
				Start: pc.Start, FirstTokenMs: pc.FirstTokenMs, TotalMs: pc.TotalMs, Status: st,
				TokensIn: pc.TokensIn, TokensOut: pc.TokensOut, Setting: "none"})
		}
	}
	fmt.Println("probe raw:", path)
}

func grade(q eval.Question, tr pipeline.Trace, byID map[string]ingest.Chunk) graded {
	var cited []ingest.Chunk
	for _, c := range tr.Citations {
		cited = append(cited, byID[c.ChunkID])
	}
	ag := eval.GradeAnswer(q, tr.Answer, tr.Refused, cited)
	return graded{Trace: tr, QID: q.ID, Type: q.Type, Expected: q.Answer, Grade: ag.Grade, CiteOK: ag.CiteOK, CiteNotes: ag.Notes}
}

func envFile(path, name string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if v, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), name+"="); ok {
			return strings.Trim(v, `"'`)
		}
	}
	return ""
}
