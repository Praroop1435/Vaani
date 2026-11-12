package test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/fakellm"
	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

// fakeModel replays scripted replies, one per request, and counts requests. No Groq budget.
type fakeModel struct {
	mu      sync.Mutex
	replies []func(w http.ResponseWriter)
	calls   atomic.Int32
}

func streamReply(text string, delay time.Duration) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("x-ratelimit-remaining-requests", "900")
		w.Header().Set("x-ratelimit-remaining-tokens", "7000")
		time.Sleep(delay)
		for _, word := range fakellm.Pieces(text, 3) { // Groq-sized pieces: tags get split
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", word)
			w.(http.Flusher).Flush()
		}
		fmt.Fprint(w, "data: {\"choices\":[],\"x_groq\":{\"usage\":{\"prompt_tokens\":1500,\"completion_tokens\":60}}}\n\ndata: [DONE]\n\n")
	}
}

func statusReply(code int, retryAfter, body string) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		if retryAfter != "" {
			w.Header().Set("retry-after", retryAfter)
		}
		w.WriteHeader(code)
		fmt.Fprint(w, body)
	}
}

func (f *fakeModel) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.calls.Add(1)
	f.mu.Lock()
	if len(f.replies) == 0 {
		f.mu.Unlock()
		statusReply(500, "", "no scripted reply")(w)
		return
	}
	next := f.replies[0]
	f.replies = f.replies[1:]
	f.mu.Unlock()
	next(w)
}

func newPipeline(t *testing.T, fm *fakeModel, cfg pipeline.Config) *pipeline.Pipeline {
	t.Helper()
	ix, err := retrieve.Load("../data/index", 300)
	if err != nil {
		t.Skipf("index not built: %v", err)
	}
	srv := httptest.NewServer(fm)
	t.Cleanup(srv.Close)
	return &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: srv.URL, Model: "fake"},
		Gov: governor.New(governor.Limits{RPM: 30, TPM: 8000, RPD: 1000}),
		Ret: &retrieve.Retriever{Embedder: embedder(t), Indexes: map[int]*retrieve.Index{300: ix}},
		Cfg: cfg,
	}
}

func cfgNoUnderstand() pipeline.Config {
	c := pipeline.Defaults
	c.Understand = false
	c.Policy.BaseBackoff = 50 * time.Millisecond
	return c
}

func TestPipelineSendsOnlyVerifiedSentences(t *testing.T) {
	fm := &fakeModel{replies: []func(http.ResponseWriter){
		streamReply("The Transformer uses multi-head attention [S1]. It has 999 heads [S1]. It cites a passage that does not exist [S9].", 0),
	}}
	p := newPipeline(t, fm, cfgNoUnderstand())
	var shown []string
	tr := p.Answer(context.Background(), "t1", "How many parallel attention heads does the base Transformer use?", func(s string) { shown = append(shown, s) })
	if tr.Status != "ok" || tr.Refused {
		t.Fatalf("status %s refused %v err %s", tr.Status, tr.Refused, tr.Err)
	}
	if !strings.HasPrefix(tr.Answer, "The Transformer uses multi-head attention [Attention Is All You Need, p.") {
		t.Errorf("verified sentence not rendered with its citation: %q", tr.Answer)
	}
	if strings.Contains(tr.Answer, "999") || strings.Contains(tr.Answer, "S9") {
		t.Errorf("unsupported sentence reached the person: %q", tr.Answer)
	}
	if len(tr.Dropped) != 2 {
		t.Errorf("want 2 dropped sentences (made-up number, missing passage), got %q", tr.Dropped)
	}
	if tr.FirstTokenMs < 0 || len(shown) == 0 || len(tr.Citations) != 1 {
		t.Errorf("first token %d, shown %d pieces, citations %+v", tr.FirstTokenMs, len(shown), tr.Citations)
	}
	if tr.TotalMs <= 0 {
		t.Error("total time not recorded")
	}
}

func TestPipelineRefusalFromModel(t *testing.T) {
	// round 1 refuses, the targeted round adds passages, round 2 refuses again
	fm := &fakeModel{replies: []func(http.ResponseWriter){streamReply("NOT IN PAPERS", 0), streamReply("NOT IN PAPERS", 0)}}
	p := newPipeline(t, fm, cfgNoUnderstand())
	tr := p.Answer(context.Background(), "t2", "What BLEU score does the Transformer reach on English-to-Spanish translation?", nil)
	if tr.Status != "ok" || !tr.Refused || !strings.HasPrefix(tr.Answer, "The papers do not say.") {
		t.Errorf("status %s refused %v answer %q", tr.Status, tr.Refused, tr.Answer)
	}
}

func TestPipelineRefusesOffTopicWithoutModelCall(t *testing.T) {
	fm := &fakeModel{}
	p := newPipeline(t, fm, cfgNoUnderstand())
	tr := p.Answer(context.Background(), "t3", "What is the capital of France?", nil)
	if !tr.Refused || fm.calls.Load() != 0 {
		t.Errorf("refused %v after %d model calls: %q", tr.Refused, fm.calls.Load(), tr.Answer)
	}
}

func TestPipelineRetriesPerMinuteLimit(t *testing.T) {
	fm := &fakeModel{replies: []func(http.ResponseWriter){
		statusReply(429, "0.3", `{"error":{"message":"Rate limit reached on tokens per minute (TPM)"}}`),
		streamReply("The Transformer uses multi-head attention [S1].", 0),
	}}
	p := newPipeline(t, fm, cfgNoUnderstand())
	tr := p.Answer(context.Background(), "t4", "How many parallel attention heads does the base Transformer use?", nil)
	draft := tr.Steps[len(tr.Steps)-1]
	if tr.Status != "ok" || tr.Retries != 1 || draft.RetryMs < 300 {
		t.Errorf("status %s retries %d retry wait %.0fms (want >= 300, the retry-after)", tr.Status, tr.Retries, draft.RetryMs)
	}
}

func TestPipelineDailyBudgetGivesPassagesNotGuesses(t *testing.T) {
	daily := statusReply(429, "3600", `{"error":{"message":"Rate limit reached for model on requests per day (RPD): Limit 1000"}}`)
	fm := &fakeModel{replies: []func(http.ResponseWriter){daily, daily}}
	cfg := cfgNoUnderstand()
	cfg.Understand = true
	p := newPipeline(t, fm, cfg)
	tr := p.Answer(context.Background(), "t5", "What frame rate does the Mimi codec run at?", nil)
	if tr.Status != "rate_limited" || !strings.Contains(tr.Answer, "model budget (1000 requests/day) is used up") ||
		!strings.Contains(tr.Answer, "next request can run") || !strings.Contains(tr.Answer, "full daily quota is back") || !strings.Contains(tr.Answer, "Moshi, p.") {
		t.Errorf("status %s answer %q", tr.Status, tr.Answer)
	}
	if fm.calls.Load() != 2 { // understand + draft, each once: a daily 429 is never retried
		t.Errorf("model called %d times, want 2", fm.calls.Load())
	}
}

func TestPipelineGivesUpOnSlowFirstToken(t *testing.T) {
	slow := streamReply("late [S1].", 3*time.Second)
	fm := &fakeModel{replies: []func(http.ResponseWriter){slow, slow, slow}}
	cfg := cfgNoUnderstand()
	cfg.Deadline, cfg.FirstTokenTimeout = 2*time.Second, 400*time.Millisecond
	cfg.Policy.MinUseful = 500 * time.Millisecond
	p := newPipeline(t, fm, cfg)
	start := time.Now()
	tr := p.Answer(context.Background(), "t6", "How many parallel attention heads does the base Transformer use?", nil)
	if tr.Status != "timeout" || time.Since(start) > 2500*time.Millisecond {
		t.Errorf("status %s after %s, answer %q", tr.Status, time.Since(start), tr.Answer)
	}
	if tr.Retries < 1 {
		t.Errorf("a slow first token should be retried while the deadline allows, retries=%d", tr.Retries)
	}
}

func TestGovernorWaitsAndFailsFast(t *testing.T) {
	g := governor.New(governor.Limits{RPM: 2, TPM: 1000, RPD: 3})
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		if _, err := g.Reserve(ctx, 100); err != nil {
			t.Fatal(err)
		}
	}
	short, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
	defer cancel()
	if _, err := g.Reserve(short, 100); !errors.Is(err, governor.ErrDeadlineWait) {
		t.Errorf("3rd call in a minute with RPM=2 and 200ms left: got %v, want ErrDeadlineWait", err)
	}
	if _, err := g.Reserve(ctx, 5000); !errors.Is(err, governor.ErrTooLargeAlone) {
		t.Errorf("call larger than TPM: got %v", err)
	}
	h := http.Header{}
	h.Set("x-ratelimit-remaining-requests", "0")
	h.Set("x-ratelimit-reset-requests", "2h0m0s")
	g.Observe(h)
	if _, err := g.Reserve(ctx, 10); !errors.Is(err, governor.ErrDailyBudget) {
		t.Errorf("provider says 0 requests left today: got %v", err)
	}
	// next request ≠ full reset: with RPD=3 one request refills every 8 h; the header's 2 h is "full again"
	if d := g.Daily(); d.Left != 0 || d.NextIn != 8*time.Hour || d.FullAt.Before(time.Now().Add(119*time.Minute)) || d.FullAt.After(time.Now().Add(121*time.Minute)) {
		t.Errorf("daily status from headers: %+v", d)
	}
}

// Groq refills the daily budget continuously (measured: reset-requests grew 86.4 s per request at
// 1000/day). After the provider reports 0 left, a request must become possible again after one
// refill interval, not after 24 hours.
func TestGovernorDailyBudgetRefillsGradually(t *testing.T) {
	g := governor.New(governor.Limits{RPM: 30, TPM: 8000, RPD: 1000})
	h := http.Header{}
	h.Set("x-ratelimit-limit-requests", "1000")
	h.Set("x-ratelimit-remaining-requests", "0")
	h.Set("x-ratelimit-reset-requests", "23h58m33.6s")
	g.Observe(h)
	d := g.Daily()
	if d.Left != 0 || d.RefillEach != 86400*time.Millisecond || d.NextIn != 86400*time.Millisecond {
		t.Errorf("right after hitting the limit: %+v (want next request in 1m26.4s)", d)
	}
	if d.FullAt.Sub(time.Now()) < 23*time.Hour {
		t.Errorf("full reset should be ~24 h away, got %v", time.Until(d.FullAt))
	}
}

func TestRetryPolicy(t *testing.T) {
	p := llm.Policy{MaxRetries: 2, BaseBackoff: 10 * time.Millisecond, MaxBackoff: 50 * time.Millisecond, AttemptTimeout: time.Second, MinUseful: 100 * time.Millisecond}
	errs := func(es ...error) func(context.Context) (llm.Result, error) {
		i := 0
		return func(context.Context) (llm.Result, error) {
			e := es[min(i, len(es)-1)]
			i++
			return llm.Result{}, e
		}
	}
	cases := []struct {
		name     string
		call     func(context.Context) (llm.Result, error)
		attempts int
	}{
		{"5xx twice then ok", errs(&llm.APIError{Status: 503}, &llm.APIError{Status: 502}, nil), 3},
		{"400 is never retried", errs(&llm.APIError{Status: 400}), 1},
		{"daily 429 is never retried", errs(&llm.APIError{Status: 429, Body: "requests per day (RPD)"}), 1},
		{"gives up after MaxRetries", errs(&llm.APIError{Status: 500}), 3},
	}
	for _, c := range cases {
		_, st, _ := llm.Do(context.Background(), p, c.call)
		if len(st.Attempts) != c.attempts {
			t.Errorf("%s: %d attempts, want %d", c.name, len(st.Attempts), c.attempts)
		}
	}
	// a retry-after longer than the time left is not attempted: it would only arrive after the person gave up
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	_, st, _ := llm.Do(ctx, p, errs(&llm.APIError{Status: 429, RetryAfter: time.Second}))
	if len(st.Attempts) != 1 {
		t.Errorf("retried into a missed deadline: %d attempts", len(st.Attempts))
	}
}
