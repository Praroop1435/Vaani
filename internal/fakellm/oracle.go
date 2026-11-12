// Package fakellm is an OpenAI-style chat server for offline runs: no Groq tokens.
//
// Oracle plays a model that reads perfectly and never guesses. For the understand prompt it returns
// the benchmark's own parts for the question. For a draft it answers each requested part only if a
// passage in the prompt really contains that part's verified evidence (it quotes that evidence and
// cites the passage), and otherwise writes "[P#] MISSING". Unanswerable questions get
// "NOT IN PAPERS". So an offline benchmark run measures retrieval, completeness, targeted retrieval,
// citations and grading together, with the model held perfect. It cannot show model mistakes such
// as substituting a related figure: that needs the real model.
package fakellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/ingest"
)

type Oracle struct {
	Bench      *eval.Benchmark
	FirstDelay time.Duration // before the first token
	TokenDelay time.Duration // between streamed words
	// GroqHeaders sends Groq's free-tier limits (1000 requests/day, 8000 tokens/min) in
	// x-ratelimit-* headers; the governor then enforces them. Off for quality runs.
	GroqHeaders bool
	// Style is how the draft is formatted. "" = the format the prompt asks for ("[P1] … [S2].").
	// "colon" and "mixed" are what gpt-oss-120b actually wrote in live smoke test 2:
	//   colon: "P1: … [S1][S2]."
	//   mixed: "P1: … [S1]. P2: [P2] MISSING." and, for a lone missing part, "P1: [S1] MISSING"
	Style string

	mu       sync.Mutex
	requests int
}

var paperByShort = map[string]string{
	"Attention Is All You Need": "1706.03762v7", "Moshi": "2410.00037v1", "H-Net (Dynamic Chunking)": "2507.07955v2",
}

var (
	passageHead = regexp.MustCompile(`(?m)^\[S(\d+)\] \((.+?), page ([\d-]+), section (.*)\)$`)
	partLine    = regexp.MustCompile(`(?m)^\[?P(\d+)\]?:? (.+)$`)
	tableIn     = regexp.MustCompile(`, (Table \w+)$`)
	authorYear  = regexp.MustCompile(`\s*\([A-Z][^()]{0,80}?\d{4}[a-z]?\)`)
)

type passage struct {
	tag   string
	chunk ingest.Chunk
}

func (o *Oracle) question(text string) *eval.Question {
	for i, q := range o.Bench.Questions {
		if q.Question == strings.TrimSpace(text) {
			return &o.Bench.Questions[i]
		}
	}
	return nil
}

func (o *Oracle) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Messages []struct{ Role, Content string } `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) < 2 {
		http.Error(w, `{"error":{"message":"bad request"}}`, http.StatusBadRequest)
		return
	}
	reply := o.Reply(req.Messages[0].Content, req.Messages[len(req.Messages)-1].Content)
	o.mu.Lock()
	o.requests++
	left := 1000 - o.requests
	o.mu.Unlock()
	w.Header().Set("Content-Type", "text/event-stream")
	if o.GroqHeaders {
		w.Header().Set("x-ratelimit-limit-requests", "1000")
		w.Header().Set("x-ratelimit-remaining-requests", strconv.Itoa(left))
		w.Header().Set("x-ratelimit-limit-tokens", "8000")
		w.Header().Set("x-ratelimit-remaining-tokens", "8000")
		w.Header().Set("x-ratelimit-reset-requests", (time.Duration(1000-left) * 86400 * time.Millisecond).String())
		w.Header().Set("x-ratelimit-reset-tokens", "1s")
	}
	time.Sleep(o.FirstDelay)
	fl, _ := w.(http.Flusher)
	for _, word := range Pieces(reply, 3) {
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", word)
		if fl != nil {
			fl.Flush()
		}
		time.Sleep(o.TokenDelay)
	}
	prompt := 0
	for _, m := range req.Messages {
		prompt += len(m.Content) / 4
	}
	fmt.Fprintf(w, "data: {\"choices\":[],\"x_groq\":{\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}}\n\ndata: [DONE]\n\n", prompt, len(reply)/4+1)
}

// Reply is what the oracle answers to one system + user prompt (the stand-in reuses it).
func (o *Oracle) Reply(system, user string) string {
	switch {
	case strings.Contains(system, "You prepare questions"):
		return o.understand(user)
	case strings.Contains(system, "You answer engineers' questions"):
		return o.draft(user)
	}
	return "NOT IN PAPERS"
}

func (o *Oracle) tag(k int) string {
	if o.Style == "colon" || o.Style == "mixed" {
		return fmt.Sprintf("P%d:", k)
	}
	return fmt.Sprintf("[P%d]", k)
}

func (o *Oracle) understand(question string) string {
	parts := []string{question}
	if q := o.question(question); q != nil && len(q.Parts) > 0 {
		parts = q.Parts
	}
	js, _ := json.Marshal(map[string]any{"query": question, "parts": parts})
	return string(js)
}

func (o *Oracle) draft(user string) string {
	qText, _, _ := strings.Cut(user[strings.LastIndex(user, "Question: ")+len("Question: "):], "\n")
	q := o.question(qText)
	if q == nil || !q.Answerable {
		return "NOT IN PAPERS"
	}
	var ps []passage
	heads := passageHead.FindAllStringSubmatchIndex(user, -1)
	stop := strings.LastIndex(user, "Question: ")
	for i, h := range heads {
		end := stop
		if i+1 < len(heads) {
			end = heads[i+1][0]
		}
		sec := user[h[8]:h[9]]
		c := ingest.Chunk{Paper: paperByShort[user[h[4]:h[5]]], Kind: "text", Text: strings.TrimSpace(user[h[1]:end])}
		if m := tableIn.FindStringSubmatch(sec); m != nil {
			c.Kind, c.Table = "table", m[1]
		}
		ps = append(ps, passage{"S" + user[h[2]:h[3]], c})
	}
	var out []string
	for _, m := range partLine.FindAllStringSubmatch(user[stop:], -1) {
		k, _ := strconv.Atoi(m[1])
		var ev []eval.Evidence
		switch {
		case len(q.Parts) > 0 && k <= len(q.Evidence):
			ev = []eval.Evidence{q.Evidence[k-1]}
		case k == 1:
			ev = q.Evidence // single-part question: every evidence item answers it
		}
		sentence := ""
		for _, e := range ev {
			for _, src := range e.AnyOf {
				for _, p := range ps {
					if sentence == "" && src.Covers(p.chunk) {
						stmt := authorYear.ReplaceAllString(src.Quote, "")
						if src.Table != "" {
							stmt = fmt.Sprintf("%s %s: %s", src.Row, src.Column, src.Value)
						}
						sentence = fmt.Sprintf("%s %s [%s].", o.tag(k), stmt, p.tag)
					}
				}
			}
		}
		if sentence == "" {
			switch {
			case o.Style == "mixed" && len(q.Parts) <= 1 && len(ps) > 0:
				sentence = fmt.Sprintf("P%d: [%s] MISSING", k, ps[0].tag)
			case o.Style == "mixed":
				sentence = fmt.Sprintf("P%d: [P%d] MISSING.", k, k)
			default:
				sentence = fmt.Sprintf("%s MISSING", o.tag(k))
			}
		}
		out = append(out, sentence)
	}
	if len(out) == 0 {
		return "NOT IN PAPERS"
	}
	return strings.Join(out, " ")
}

// Pieces splits s into chunks of n runes. Groq streams pieces this small, which split "[S1][S2]"
// mid-tag; streaming whole words hid that (live smoke test 3).
func Pieces(s string, n int) []string {
	var out []string
	r := []rune(s)
	for i := 0; i < len(r); i += n {
		out = append(out, string(r[i:min(i+n, len(r))]))
	}
	return out
}
