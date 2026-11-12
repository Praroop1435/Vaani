// Command standin serves the stand-in model service on its own port (OpenAI-style
// POST /chat/completions, streaming), fitted from the real calibration traces. cmd/load runs the
// same server in-process; this is for poking at it by hand, e.g. with cmd/ask -base-url.
//
//	go run ./cmd/standin -addr :8099 -tpm 4000 -fail 0.05
package main

import (
	"flag"
	"log"
	"net/http"
	"path/filepath"

	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/fakellm"
	"github.com/Praroop1435/Vaani/internal/standin"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8099", "listen address")
	c := standin.Config{PromptOverhead: 71, PromptRatio: 0.889, StallMs: 8000}
	flag.IntVar(&c.RPM, "rpm", 30, "requests per minute")
	flag.IntVar(&c.TPM, "tpm", 8000, "tokens per minute (token bucket)")
	flag.IntVar(&c.RPD, "rpd", 1000, "requests per day")
	flag.IntVar(&c.RPDUsed, "rpd-used", 0, "daily requests already used at start")
	flag.IntVar(&c.Capacity, "capacity", 3, "requests in flight before it slows down")
	flag.Float64Var(&c.SlowPerExtra, "slow", 0.3, "slowdown per request in flight over capacity")
	flag.Float64Var(&c.FailRate, "fail", 0.01, "share of calls answered HTTP 500")
	flag.Float64Var(&c.StallRate, "stall", 0.02, "share of calls whose first token takes -stall-ms")
	flag.Float64Var(&c.DropRate, "drop", 0.005, "share of streams cut half way")
	calib := flag.String("calib", "results/calibration/*_traces.jsonl", "real traces to fit latency from")
	flag.Parse()
	paths, _ := filepath.Glob(*calib)
	calls, err := standin.Calls(paths...)
	if err != nil {
		log.Fatal(err)
	}
	c = standin.Fit(c, calls)
	if b, err := eval.Load("data/eval/benchmark.json"); err == nil {
		c.Reply = (&fakellm.Oracle{Bench: b}).Reply
	}
	log.Printf("stand-in on http://%s (fitted from %d real calls: %+v)", *addr, len(calls), c)
	log.Fatal(http.ListenAndServe(*addr, standin.New(c)))
}
