// Command tables recomputes the brief's Tables 1, 2a, 2b and 3 from results.csv and prints them
// as Markdown (also written to results/tables.md). Reviewers can rerun it:
//
//	go run ./cmd/tables
//
// Rules (also printed under the tables):
//   - Percentiles are nearest-rank: the value at position ceil(p/100 * n) of the sorted list.
//   - First token and total time are over requests with status ok (a real answer or refusal);
//     rate-limited, timed-out and errored requests are counted in Table 2b, not in the latencies.
//   - Tokens per answer = all tokens of all requests in the scenario / number of ok requests, so
//     tokens spent on requests that failed are charged to the answers that succeeded.
//   - Cost per 1,000 answers uses Groq's paid price for openai/gpt-oss-120b ($0.15 / 1M input,
//     $0.60 / 1M output, console.groq.com, 2026-11-11). On the free tier we paid $0.
//   - Peak queue length = the most requests in flight in our system at once, from start_time and
//     total_ms.
//
// Table 1 is the exception: the limits themselves come from response headers, which are not
// results.csv columns. They are read from the raw calibration logs submitted with it
// (results/calibration/*_probe.jsonl and *_traces.jsonl); the 429, the recovery time and the
// requests in flight come from results.csv's calibration_probe rows.
package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
)

const priceIn, priceOut = 0.15, 0.60 // $ per 1M tokens

type row struct {
	id, scenario, source string
	start                time.Time
	first, total         int64
	status               string
	retries, in, out     int
	setting, grade       string
}

// The load shapes in Table 2, in order, with the name the table shows.
var loadShapes = [][2]string{
	{"steady", "steady"}, {"burst", "burst"}, {"overload", "sustained overload"},
	{"rpm_halved", "per-minute limit halved"}, {"daily_exhausted", "daily budget exhausted"},
}

// Limiter version per scenario (governor.Version at run time; see results/load/*_summary.json).
var limiter = map[string]string{"steady": "v2", "burst": "v2", "overload": "v2", "rpm_halved": "v2", "daily_exhausted": "v2"}

// Table 3 sources: the real runs (A, then R, the default since 2026-11-12) and the stand-in comparison.
var table3 = []struct{ scenario, label string }{
	{"quality", "real Groq, gpt-oss-120b"}, {"quality_qwen", "real Groq, qwen3.8-27b"}, {"retrieval_standin", "stand-in (oracle model)"},
}

var settingName = map[string]string{
	"A":   "A: 300-token pieces, 4 fetched, no re-rank",
	"H10": "H10: BGE top-10 + BM25 top-10 fused (RRF re-rank), 4 kept",
	"H15": "H15: BGE top-15 + BM25 top-15 fused (RRF re-rank), 4 kept",
	"R":   "R: 300-token pieces; paper scope, BGE top-20 + BM25 top-20, cross-encoder re-rank, 6 kept",
}

func main() {
	path := flag.String("csv", "results/results.csv", "results.csv")
	out := flag.String("out", "results/tables.md", "Markdown output (empty: stdout only)")
	flag.Parse()
	rows, err := load(*path)
	if err != nil {
		log.Fatal(err)
	}
	var b strings.Builder
	table1(&b, rows)
	table2(&b, rows)
	table3s(&b, rows)
	fmt.Fprintf(&b, "\nComputed by `go run ./cmd/tables` from `%s` (%d rows). Percentiles nearest-rank; latencies over ok requests; tokens per answer = all tokens / ok requests; cost at $%.2f / $%.2f per 1M input / output tokens (Groq paid price; free tier cost $0).\n",
		*path, len(rows), priceIn, priceOut)
	fmt.Print(b.String())
	if *out != "" {
		if err := os.WriteFile(*out, []byte(b.String()), 0o644); err != nil {
			log.Fatal(err)
		}
	}
}

func load(path string) ([]row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	var rows []row
	for i, r := range recs[1:] {
		if len(r) != 12 {
			return nil, fmt.Errorf("row %d: %d columns", i+2, len(r))
		}
		t, err := time.Parse(time.RFC3339Nano, r[3])
		if err != nil {
			return nil, fmt.Errorf("row %d: %v", i+2, err)
		}
		n := func(s string) int64 { v, _ := strconv.ParseInt(s, 10, 64); return v }
		rows = append(rows, row{r[0], r[1], r[2], t, n(r[4]), n(r[5]), r[6], int(n(r[7])), int(n(r[8])), int(n(r[9])), r[10], r[11]})
	}
	return rows, nil
}

func pick(rows []row, f func(row) bool) []row {
	var out []row
	for _, r := range rows {
		if f(r) {
			out = append(out, r)
		}
	}
	return out
}

func pct(v []int64, p float64) int64 {
	if len(v) == 0 {
		return -1
	}
	s := slices.Clone(v)
	slices.Sort(s)
	return s[max(int(math.Ceil(p/100*float64(len(s))))-1, 0)]
}

func ms(rows []row, first bool, ps ...float64) string {
	var v []int64
	for _, r := range rows {
		if r.status != "ok" {
			continue
		}
		if first {
			v = append(v, r.first)
		} else {
			v = append(v, r.total)
		}
	}
	if len(v) == 0 {
		return "not measured (no ok requests)"
	}
	var parts []string
	for _, p := range ps {
		parts = append(parts, strconv.FormatInt(pct(v, p), 10))
	}
	return strings.Join(parts, " / ")
}

func share(rows []row, f func(row) bool) string {
	return fmt.Sprintf("%.1f", 100*float64(len(pick(rows, f)))/float64(len(rows)))
}

func sources(rows []row) string {
	seen := map[string]bool{}
	var s []string
	for _, r := range rows {
		if !seen[r.source] {
			seen[r.source] = true
			s = append(s, r.source)
		}
	}
	return strings.Join(s, " + ")
}

func tokensPerAnswer(rows []row) (float64, float64) {
	in, out, ok := 0, 0, 0
	for _, r := range rows {
		in, out = in+r.in, out+r.out
		if r.status == "ok" {
			ok++
		}
	}
	if ok == 0 {
		return math.NaN(), math.NaN()
	}
	return float64(in+out) / float64(ok), 1000 * (float64(in)*priceIn + float64(out)*priceOut) / 1e6 / float64(ok)
}

func peakInFlight(rows []row) int {
	type ev struct {
		t time.Time
		d int
	}
	var evs []ev
	for _, r := range rows {
		evs = append(evs, ev{r.start, 1}, ev{r.start.Add(time.Duration(r.total) * time.Millisecond), -1})
	}
	sort.Slice(evs, func(i, j int) bool {
		if evs[i].t.Equal(evs[j].t) {
			return evs[i].d < evs[j].d // an end before a start at the same instant
		}
		return evs[i].t.Before(evs[j].t)
	})
	cur, peak := 0, 0
	for _, e := range evs {
		cur += e.d
		peak = max(peak, cur)
	}
	return peak
}

func table2(b *strings.Builder, rows []row) {
	b.WriteString("\n### Table 2a: Latency, one row per load scenario\n\n| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |\n|---|---|---|---|---|---|---|\n")
	var b2 strings.Builder
	b2.WriteString("\n### Table 2b: Outcomes, same rows\n\n| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |\n|---|---|---|---|---|---|---|---|\n")
	for _, sh := range loadShapes {
		rs := pick(rows, func(r row) bool { return r.scenario == sh[0] })
		if len(rs) == 0 {
			fmt.Fprintf(b, "| %s | not measured | | | | | |\n", sh[1])
			fmt.Fprintf(&b2, "| %s | not measured | | | | | | |\n", sh[1])
			continue
		}
		first, last, end := rs[0].start, rs[0].start, rs[0].start
		for _, r := range rs {
			if r.start.Before(first) {
				first = r.start
			}
			if r.start.After(last) {
				last = r.start
			}
			if e := r.start.Add(time.Duration(r.total) * time.Millisecond); e.After(end) {
				end = e
			}
		}
		rate := "n/a"
		if span := last.Sub(first).Minutes(); span > 0 {
			rate = fmt.Sprintf("%.1f", float64(len(rs)-1)/span)
		}
		ok := len(pick(rs, func(r row) bool { return r.status == "ok" }))
		fmt.Fprintf(b, "| %s (limiter %s) | %s | %d | %.0f | %s | %s (n=%d) | %s (n=%d) |\n", sh[1], limiter[sh[0]], sources(rs), len(rs),
			end.Sub(first).Seconds(), rate, ms(rs, true, 50, 95, 99), ok, ms(rs, false, 50, 95, 99), ok)
		retries := 0
		for _, r := range rs {
			retries += r.retries
		}
		tpa, cost := tokensPerAnswer(rs)
		errs := ""
		if e := len(pick(rs, func(r row) bool { return r.status == "error" })); e > 0 {
			errs = fmt.Sprintf(" (+%s%% error)", share(rs, func(r row) bool { return r.status == "error" }))
		}
		fmt.Fprintf(&b2, "| %s | %s | %s | %s%s | %.2f | %d | %.0f | $%.2f (free tier: $0) |\n", sh[1],
			share(rs, func(r row) bool { return r.status == "ok" }), share(rs, func(r row) bool { return r.status == "rate_limited" }),
			share(rs, func(r row) bool { return r.status == "timeout" }), errs, float64(retries)/float64(len(rs)), peakInFlight(rs), tpa, cost)
	}
	b.WriteString(b2.String())
}

func table3s(b *strings.Builder, rows []row) {
	b.WriteString("\n### Table 3: Retrieval settings\n\n| Setting (piece size, pieces fetched, re-rank) | Model | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |\n|---|---|---|---|---|---|---|---|---|\n")
	for _, src := range table3 {
		for _, set := range []string{"A", "H10", "H15", "R"} {
			rs := pick(rows, func(r row) bool { return r.scenario == src.scenario && r.setting == set })
			if len(rs) == 0 {
				if src.scenario == "quality_qwen" || (src.scenario == "quality" && set != "A" && set != "R") {
					continue // the real runs are A and R only
				}
				fmt.Fprintf(b, "| %s | %s | not measured | | | | | | |\n", settingName[set], src.label)
				continue
			}
			tpa, _ := tokensPerAnswer(rs)
			fmt.Fprintf(b, "| %s | %s | %d | %s | %s | %s | %.0f | %s | %s |\n", settingName[set], src.label, len(rs),
				share(rs, func(r row) bool { return r.grade == "correct" }), share(rs, func(r row) bool { return r.grade == "wrong" }),
				share(rs, func(r row) bool { return r.grade == "refused_correctly" }), tpa, ms(rs, true, 50, 95), ms(rs, false, 50, 95))
		}
	}
	b.WriteString("\n\"Correct and cited\": answer_grade correct. Every sentence the pipeline shows carries a citation to a passage that contains its numbers (the check refuses any other), so a correct answer is always cited. Percentages are of questions asked, so the four grades (with refused wrongly) sum to 100%.\n")
}

// ---- Table 1: headers from the raw calibration logs, events from results.csv ----

type hdrs = map[string]string

func rawHeaders(dir string) ([]hdrs, []map[string]any) {
	var hs []hdrs
	var probes []map[string]any
	files, _ := filepath.Glob(dir + "/*_traces.jsonl")
	for _, f := range files {
		eachLine(f, func(l []byte) {
			var t struct {
				Trace struct {
					Steps []struct {
						RateHdr hdrs `json:"rate_headers"`
					} `json:"steps"`
				} `json:"trace"`
			}
			if json.Unmarshal(l, &t) == nil {
				for _, s := range t.Trace.Steps {
					if s.RateHdr != nil {
						hs = append(hs, s.RateHdr)
					}
				}
			}
		})
	}
	files, _ = filepath.Glob(dir + "/*_probe.jsonl")
	for _, f := range files {
		eachLine(f, func(l []byte) {
			var p map[string]any
			if json.Unmarshal(l, &p) == nil {
				probes = append(probes, p)
				if h, ok := p["headers"].(map[string]any); ok {
					m := hdrs{}
					for k, v := range h {
						m[k] = fmt.Sprint(v)
					}
					hs = append(hs, m)
				}
			}
		})
	}
	return hs, probes
}

func eachLine(path string, f func([]byte)) {
	fh, err := os.Open(path)
	if err != nil {
		return
	}
	defer fh.Close()
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for sc.Scan() {
		f(sc.Bytes())
	}
}

func table1(b *strings.Builder, rows []row) {
	hs, probes := rawHeaders("results/calibration")
	qh, _ := rawHeaders("results/calibration_qwen")
	var qRates []float64
	for _, h := range qh {
		lim, _ := strconv.Atoi(h["x-ratelimit-limit-tokens"])
		left, err1 := strconv.Atoi(h["x-ratelimit-remaining-tokens"])
		reset, err2 := time.ParseDuration(h["x-ratelimit-reset-tokens"])
		if err1 == nil && err2 == nil && reset > 0 {
			qRates = append(qRates, float64(lim-left)/reset.Seconds())
		}
	}
	slices.Sort(qRates)
	var rates, perReq []float64
	tpm, rpd := "", ""
	var prevReset float64 = -1
	prevLeft := -1
	for _, h := range hs {
		tpm, rpd = h["x-ratelimit-limit-tokens"], h["x-ratelimit-limit-requests"]
		lim, _ := strconv.Atoi(tpm)
		left, err1 := strconv.Atoi(h["x-ratelimit-remaining-tokens"])
		reset, err2 := time.ParseDuration(h["x-ratelimit-reset-tokens"])
		if err1 == nil && err2 == nil && reset > 0 {
			rates = append(rates, float64(lim-left)/reset.Seconds())
		}
		rq, _ := strconv.Atoi(h["x-ratelimit-remaining-requests"])
		rr, err := time.ParseDuration(h["x-ratelimit-reset-requests"])
		if err == nil && prevLeft-rq == 1 && prevReset >= 0 {
			perReq = append(perReq, rr.Seconds()-prevReset)
		}
		prevLeft, prevReset = rq, rr.Seconds()
	}
	slices.Sort(rates)
	slices.Sort(perReq)
	med := func(x []float64) float64 {
		if len(x) == 0 {
			return math.NaN()
		}
		return x[len(x)/2]
	}
	msg, retryAfter := "not observed", ""
	for _, p := range probes {
		if fmt.Sprint(p["http_status"]) == "429" {
			msg = fmt.Sprint(p["body"])
			if h, ok := p["headers"].(map[string]any); ok {
				retryAfter = fmt.Sprint(h["retry-after"])
			}
		}
	}
	if m := regexp.MustCompile(`on tokens per minute \(TPM\): (Limit \d+, Used \d+, Requested \d+\. Please try again in [\d.]+s)`).FindStringSubmatch(msg); m != nil {
		msg = "HTTP 429 `rate_limit_exceeded`, `retry-after: " + retryAfter + "`, \"" + m[1] + "\"; the refused call is not charged"
	}
	pr := pick(rows, func(r row) bool { return strings.HasSuffix(r.scenario, "_probe") })
	recovery, inFlight := "not measured", "not measured"
	for i, r := range pr {
		if r.status == "rate_limited" {
			for _, q := range pr[i+1:] {
				if q.status == "ok" && q.start.After(r.start) {
					recovery = fmt.Sprintf("%d ms after the 429 a call succeeded (retry-after said %s s)", q.start.Sub(r.start).Milliseconds(), retryAfter)
					break
				}
			}
		}
	}
	same := map[time.Time]int{}
	for _, r := range pr {
		if r.status == "ok" {
			same[r.start.Truncate(10*time.Millisecond)]++
		}
	}
	most := 0
	for _, n := range same {
		most = max(most, n)
	}
	if most > 0 {
		inFlight = fmt.Sprintf(">= %d (sent together, all admitted); more not tested", most)
	}
	b.WriteString("### Table 1: Limits found in the real service (Groq free tier; probed on openai/gpt-oss-120b, headers re-read on qwen/qwen3.8-27b, the model used)\n\n| Limit | Value found | How found | Confidence |\n|---|---|---|---|\n")
	fmt.Fprintf(b, "| Requests per minute | 30 | Groq rate-limits docs; no header reports it; never reached | medium |\n")
	fmt.Fprintf(b, "| Tokens per minute | %s, a token bucket refilling at %.1f tokens/s | `x-ratelimit-limit-tokens` on %d responses; (limit - remaining) / reset-tokens, median of %d | high |\n", tpm, med(rates), len(hs), len(rates))
	fmt.Fprintf(b, "| Requests per day | %s, refilling one every %.1f s | `x-ratelimit-limit-requests`; reset-requests grows by that much per request (median of %d) | high |\n", rpd, med(perReq), len(perReq))
	fmt.Fprintf(b, "| Tokens per day | 200,000 | Groq rate-limits docs; no header reports it; never reached | low |\n")
	fmt.Fprintf(b, "| Requests in flight at once | %s | results.csv calibration_probe rows | low |\n", inFlight)
	fmt.Fprintf(b, "| What it returns when a limit is hit | %s | results.csv calibration_probe row + raw probe log | high (n=1) |\n", msg)
	fmt.Fprintf(b, "| Time until it recovers | %s | results.csv calibration_probe rows | high (n=1) |\n", recovery)
	if len(qh) > 0 {
		last := qh[len(qh)-1]
		fmt.Fprintf(b, "| Same limits on qwen/qwen3.8-27b | %s tokens/min refilling at %.1f tokens/s; %s requests/day | headers on %d responses of the scored run (results/calibration_qwen/); no 429 probe on this model | high for the limits, 429 behaviour assumed the same |\n",
			last["x-ratelimit-limit-tokens"], med(qRates), last["x-ratelimit-limit-requests"], len(qh))
	}
}
