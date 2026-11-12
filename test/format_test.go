package test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/Praroop1435/Vaani/internal/fakellm"
	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

// Regressions from live smoke test 2 (results/smoke/20261111-185506_traces.jsonl). Each reply below
// is what gpt-oss-120b actually wrote; only the [S#] numbers are chosen to point at a passage that
// holds the fact, because passage order depends on retrieval.

type scripted struct {
	mu      sync.Mutex
	bodies  []string
	replies []func(body string) string
}

func (sc *scripted) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	sc.mu.Lock()
	sc.bodies = append(sc.bodies, string(b))
	reply := "NOT IN PAPERS"
	if len(sc.replies) > 0 {
		reply = sc.replies[0](string(b))
		sc.replies = sc.replies[1:]
	}
	sc.mu.Unlock()
	streamReply(reply, 0)(w)
}

// tagWith returns the [S#] of the first passage in the prompt whose text contains s ("S?" if none).
func tagWith(body, s string) string {
	body = strings.ReplaceAll(strings.ReplaceAll(body, `\n`, "\n"), `\"`, `"`)
	heads := regexp.MustCompile(`\[S(\d+)\] \(`).FindAllStringSubmatchIndex(body, -1)
	for i, h := range heads {
		end := len(body)
		if i+1 < len(heads) {
			end = heads[i+1][0]
		}
		if strings.Contains(body[h[1]:end], s) {
			return "S" + body[h[2]:h[3]]
		}
	}
	return "S?"
}

func fixed(s string) func(string) string { return func(string) string { return s } }

func scriptedPipeline(t *testing.T, sc *scripted) *pipeline.Pipeline {
	t.Helper()
	ix, err := retrieve.Load("../data/index", 300)
	if err != nil {
		t.Skip(err)
	}
	srv := httptest.NewServer(sc)
	t.Cleanup(srv.Close)
	return &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: srv.URL, Model: "scripted"},
		Gov: governor.New(governor.Limits{RPM: 1000, TPM: 1_000_000, RPD: 100_000}),
		Ret: &retrieve.Retriever{Embedder: embedder(t), Indexes: map[int]*retrieve.Index{300: ix}},
		Cfg: pipeline.Defaults,
	}
}

func TestLiveFormatColonPartTagIsNotShown(t *testing.T) {
	sc := &scripted{replies: []func(string) string{
		fixed(`{"query": "Transformer big BLEU WMT 2014 English German", "parts": ["BLEU score of the Transformer (big) model on WMT 2014 English-to-German"]}`),
		func(b string) string {
			return fmt.Sprintf("P1: The Transformer (big) model achieves a BLEU score of 28.4 on WMT 2014 English‑to‑German [%s].", tagWith(b, "28.4"))
		},
	}}
	p := scriptedPipeline(t, sc)
	tr := p.Answer(context.Background(), "f1", "What BLEU score does the Transformer (big) model reach on WMT 2014 English-to-German?", nil)
	if strings.Contains(tr.Answer, "P1") || !strings.HasPrefix(tr.Answer, "The Transformer (big) model achieves a BLEU score of 28.4") {
		t.Errorf("part tag leaked or sentence lost: %q", tr.Answer)
	}
	if tr.Refused || len(tr.Missing) != 0 || len(sc.bodies) != 2 {
		t.Errorf("refused %v missing %q calls %d", tr.Refused, tr.Missing, len(sc.bodies))
	}
}

func TestLiveFormatMixedMissingRetargetsTheRightPart(t *testing.T) {
	sc := &scripted{replies: []func(string) string{
		fixed(`{"query": "Helium parameters tokens pretrained dataset full-duplex Moshi", "parts": ["number of parameters of Helium", "number of tokens Helium was pretrained on", "dataset that gives Moshi its full-duplex ability"]}`),
		func(b string) string {
			return fmt.Sprintf("P1: Helium has 7 B parameters [%s]. P2: [P2] MISSING. P3: Moshi gains its full‑duplex ability from fine‑tuning on the Fisher dataset [%s].",
				tagWith(b, "7B-parameter"), tagWith(b, "Fisher"))
		},
		func(b string) string {
			return fmt.Sprintf("[P2] Helium was trained on 2.1T tokens [%s].", tagWith(b, "2.1T tokens"))
		},
	}}
	p := scriptedPipeline(t, sc)
	tr := p.Answer(context.Background(), "f2", "How many parameters does Helium have, how many tokens was it pretrained on, and which dataset gives Moshi its full-duplex ability?", nil)
	if len(sc.bodies) != 3 {
		t.Fatalf("%d model calls, want 3", len(sc.bodies))
	}
	targeted := sc.bodies[2]
	if !strings.Contains(targeted, "[P2] number of tokens Helium was pretrained on") || strings.Contains(targeted, "[P1] number of parameters") || strings.Contains(targeted, "[P3] dataset") {
		t.Errorf("targeted round must ask for P2 only")
	}
	if strings.Count(targeted, "] (Moshi, page") > 4 {
		t.Errorf("targeted prompt must carry only the new passages (≤4), not resend the first ones")
	}
	for _, bad := range []string{"P1:", "P2:", "MISSING", "P3:"} {
		if strings.Contains(tr.Answer, bad) {
			t.Errorf("%q reached the person: %q", bad, tr.Answer)
		}
	}
	if !strings.Contains(tr.Answer, "7 B parameters") || !strings.Contains(tr.Answer, "Fisher") || !strings.Contains(tr.Answer, "2.1T") || len(tr.Missing) != 0 {
		t.Errorf("answer %q missing %q (want all three parts)", tr.Answer, tr.Missing)
	}
}

func TestLiveFormatTagThenMissingIsARefusalNotAnAnswer(t *testing.T) {
	sc := &scripted{replies: []func(string) string{
		fixed(`{"query": "Transformer BLEU English to Spanish", "parts": ["BLEU score of the Transformer on English-to-Spanish translation"]}`),
		fixed("P1: [S1] MISSING"), // verbatim, round 1
		fixed("P1: [S5] MISSING"), // same shape, targeted round
	}}
	p := scriptedPipeline(t, sc)
	tr := p.Answer(context.Background(), "f3", "What BLEU score does the Transformer reach on English-to-Spanish translation?", nil)
	if !tr.Refused || !strings.HasPrefix(tr.Answer, "The papers do not say.") {
		t.Errorf("an unanswerable question got %q (refused=%v)", tr.Answer, tr.Refused)
	}
}

func TestContentFreeAndP100(t *testing.T) {
	// a bare tag is dropped; a terse but real answer ("8 heads", a table cell) is kept
	sc := &scripted{replies: []func(string) string{
		fixed(`{"query": "Transformer training hardware", "parts": ["hardware the Transformer models were trained on"]}`),
		func(b string) string {
			return fmt.Sprintf("[P1] [%s]. [P1] 8 P100 GPUs [%s].", tagWith(b, "P100"), tagWith(b, "P100"))
		},
	}}
	p := scriptedPipeline(t, sc)
	tr := p.Answer(context.Background(), "f4", "What hardware were the Transformer models trained on?", nil)
	if !strings.HasPrefix(tr.Answer, "8 P100 GPUs [") {
		t.Errorf("answer %q", tr.Answer)
	}
	if len(tr.Dropped) != 1 || !strings.HasPrefix(tr.Dropped[0], "no content") {
		t.Errorf("the bare-tag sentence must be dropped as content-free: %q", tr.Dropped)
	}
}

func TestWarmSendsNoAPIKey(t *testing.T) {
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization") + r.Method + r.URL.Path
		w.WriteHeader(401)
	}))
	defer srv.Close()
	code, _, err := (&llm.Client{BaseURL: srv.URL, APIKey: "gsk_secret"}).Warm(context.Background())
	if err != nil || code != 401 || auth != "GET/models" {
		t.Errorf("code %d err %v request %q (must be an unauthenticated GET /models)", code, err, auth)
	}
}

// tagsWith returns the [S#] tags of every passage in the prompt whose text contains s.
func tagsWith(body, s string) []string {
	body = strings.ReplaceAll(strings.ReplaceAll(body, `\n`, "\n"), `\"`, `"`)
	heads := regexp.MustCompile(`\[S(\d+)\] \(`).FindAllStringSubmatchIndex(body, -1)
	var tags []string
	for i, h := range heads {
		end := len(body)
		if i+1 < len(heads) {
			end = heads[i+1][0]
		}
		if strings.Contains(body[h[1]:end], s) {
			tags = append(tags, "S"+body[h[2]:h[3]])
		}
	}
	return tags
}

// Smoke test 3, q05: "[P1] 28.4 BLEU [S1][S2]" arrived in small pieces; the unit ended after
// [S1] and the lone [S2] was dropped as content-free, losing the Table 2 citation.
func TestLiveSplitTagGroupKeepsBothCitations(t *testing.T) {
	sc := &scripted{replies: []func(string) string{
		fixed(`{"query": "Transformer big BLEU WMT 2014 English German", "parts": ["BLEU score of the Transformer (big) model on WMT 2014 English-to-German"]}`),
		func(b string) string {
			tags := tagsWith(b, "28.4")
			if len(tags) < 2 {
				return "need two passages with 28.4"
			}
			return fmt.Sprintf("[P1] 28.4 BLEU [%s][%s]", tags[0], tags[1]) // no final full stop, as sent
		},
	}}
	p := scriptedPipeline(t, sc)
	tr := p.Answer(context.Background(), "f5", "What BLEU score does the Transformer (big) model reach on WMT 2014 English-to-German?", nil)
	if strings.Contains(tr.Answer, "need two passages") {
		t.Skip("retrieval no longer returns two passages with 28.4")
	}
	if len(tr.Dropped) != 0 || len(tr.Citations) != 2 || !strings.Contains(tr.Answer, "; ") {
		t.Errorf("both tags must stay in one citation: answer %q dropped %q citations %d", tr.Answer, tr.Dropped, len(tr.Citations))
	}
}

// Smoke test 3, q19: round 1 said "[P1] MISSING" (right), the targeted round could not fit the
// per-minute tokens before the deadline, and the person got "can't answer right now" instead of
// a refusal. The targeted round is optional: skip it and finish.
func TestTargetedRoundSkippedWhenTokensRunOut(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "text/event-stream")
		reply := `{"query": "Transformer BLEU English to Spanish", "parts": ["BLEU score of the Transformer on English-to-Spanish translation"]}`
		if calls == 2 {
			reply = "[P1] MISSING"
			// The bucket is empty and the provider now allows 1,000 tokens/min (16.7/s): the targeted
			// draft (~1,300 tokens) would fit only after ~78 s, far past the 20 s deadline. (With
			// 8,000 TPM it would fit after ~10 s, and the governor rightly waits for it.)
			w.Header().Set("x-ratelimit-limit-tokens", "1000")
			w.Header().Set("x-ratelimit-remaining-tokens", "0")
			w.Header().Set("x-ratelimit-reset-tokens", "60s")
		}
		for _, piece := range fakellm.Pieces(reply, 3) {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", piece)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	ix, err := retrieve.Load("../data/index", 300)
	if err != nil {
		t.Skip(err)
	}
	p := &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: srv.URL, Model: "scripted"},
		Gov: governor.New(governor.Limits{RPM: 30, TPM: 8000, RPD: 1000}),
		Ret: &retrieve.Retriever{Embedder: embedder(t), Indexes: map[int]*retrieve.Index{300: ix}},
		Cfg: pipeline.Defaults,
	}
	tr := p.Answer(context.Background(), "f6", "What BLEU score does the Transformer reach on English-to-Spanish translation?", nil)
	if tr.Status != "ok" || !tr.Refused || !strings.HasPrefix(tr.Answer, "The papers do not say.") {
		t.Errorf("status %s refused %v answer %q", tr.Status, tr.Refused, tr.Answer)
	}
	if calls != 2 || !slices.ContainsFunc(tr.Degraded, func(d string) bool { return strings.HasPrefix(d, "targeted_skipped") }) {
		t.Errorf("calls %d degraded %q: the targeted draft must be skipped, not attempted", calls, tr.Degraded)
	}
}
