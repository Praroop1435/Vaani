package test

import (
	"context"
	"errors"
	"math"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/standin"
)

func standinClient(t *testing.T, c standin.Config) (*llm.Client, *standin.Server) {
	s := standin.New(c)
	ts := httptest.NewServer(s)
	t.Cleanup(ts.Close)
	return &llm.Client{BaseURL: ts.URL, Model: "openai/gpt-oss-120b"}, s
}

var hi = []llm.Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "hello there"}}

func TestStandinStreamsInPieces(t *testing.T) {
	c, _ := standinClient(t, standin.Config{TPM: 8000, RPD: 1000, FirstBaseMs: 20, ContentTokMs: 2,
		Reply: func(_, _ string) string { return "[P1] 28.4 BLEU [S1]." }})
	var got []string
	r, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 50, OnToken: func(s string) { got = append(got, s) }})
	if err != nil || r.Text != "[P1] 28.4 BLEU [S1]." || len(got) < 5 {
		t.Fatalf("text %q err %v pieces %d", r.Text, err, len(got))
	}
	if r.FirstTokenMs < 20 || r.TokensOut == 0 || r.Header.Get("x-ratelimit-remaining-requests") != "999" {
		t.Errorf("first %d ms, tokens out %d, remaining %q", r.FirstTokenMs, r.TokensOut, r.Header.Get("x-ratelimit-remaining-requests"))
	}
}

func TestStandinChargesMaxTokensAgainstTPM(t *testing.T) {
	c, s := standinClient(t, standin.Config{TPM: 1000, RPD: 1000})
	if _, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 600}); err != nil {
		t.Fatal(err)
	}
	_, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 600})
	var ae *llm.APIError
	if !errors.As(err, &ae) || ae.Status != 429 || ae.RetryAfter <= 0 || !strings.Contains(ae.Body, "tokens per minute") || llm.IsDaily(err) {
		t.Fatalf("second call must hit TPM (600+600 reserved > 1000) with retry-after: %v", err)
	}
	if _, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 2000}); !errors.As(err, &ae) || ae.Status != 413 {
		t.Errorf("a call bigger than the whole minute must be 413: %v", err)
	}
	if st := s.Stats(); st.OK != 1 {
		t.Errorf("stats %+v", st)
	}
}

func TestStandinRPMLimit(t *testing.T) {
	c, _ := standinClient(t, standin.Config{RPM: 2, TPM: 100000, RPD: 1000})
	for i := 0; i < 2; i++ {
		if _, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10}); err != nil {
			t.Fatal(err)
		}
	}
	_, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10})
	var ae *llm.APIError
	if !errors.As(err, &ae) || ae.Status != 429 || !strings.Contains(ae.Body, "requests per minute") {
		t.Fatalf("third call in a minute must hit RPM 2: %v", err)
	}
}

func TestStandinDailyBudgetRefillsGradually(t *testing.T) {
	// 86,400 requests/day refills one per second
	c, _ := standinClient(t, standin.Config{TPM: 100000, RPD: 86400, RPDUsed: 86400})
	_, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10})
	if !llm.IsDaily(err) {
		t.Fatalf("empty daily budget must be the per-day 429 the pipeline recognises: %v", err)
	}
	time.Sleep(1100 * time.Millisecond)
	if _, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10}); err != nil {
		t.Fatalf("one request must have refilled after 1 s: %v", err)
	}
}

func TestStandinSlowerUnderLoad(t *testing.T) {
	cfg := standin.Config{TPM: 1 << 30, RPD: 1 << 20, FirstBaseMs: 100, Capacity: 1, SlowPerExtra: 1}
	c, s := standinClient(t, cfg)
	alone, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var worst int64
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, _ := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10})
			mu.Lock()
			worst = max(worst, r.FirstTokenMs)
			mu.Unlock()
		}()
	}
	wg.Wait()
	if worst < 2*alone.FirstTokenMs || s.Stats().PeakInFlight < 3 {
		t.Errorf("5 at once with capacity 1 must be much slower: alone %d ms, worst under load %d ms, peak %d",
			alone.FirstTokenMs, worst, s.Stats().PeakInFlight)
	}
}

func TestStandinFaults(t *testing.T) {
	c, _ := standinClient(t, standin.Config{TPM: 1 << 30, RPD: 1 << 20, FailRate: 1})
	var ae *llm.APIError
	if _, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10}); !errors.As(err, &ae) || ae.Status != 500 {
		t.Errorf("FailRate 1 must answer 500: %v", err)
	}
	c, _ = standinClient(t, standin.Config{TPM: 1 << 30, RPD: 1 << 20, StallRate: 1, StallMs: 300})
	if _, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10, FirstTokenTimeout: 100 * time.Millisecond}); !errors.Is(err, llm.ErrFirstTokenTimeout) {
		t.Errorf("a stalled call must trip the first-token timeout: %v", err)
	}
	c, _ = standinClient(t, standin.Config{TPM: 1 << 30, RPD: 1 << 20, DropRate: 1,
		Reply: func(_, _ string) string { return strings.Repeat("word ", 20) }})
	if r, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 10}); err == nil {
		t.Errorf("a stream cut half way must be an error, got text %q", r.Text)
	}
}

func TestFitRecoversLatencyModel(t *testing.T) {
	var calls []standin.Call
	for i, tok := range []int{200, 400, 1200, 1500, 2500, 3000} {
		f := 300 + 0.5*float64(tok) // 300 ms + 500 ms per 1K prompt tokens
		if i%2 == 0 {
			f *= 1.1
		} else {
			f /= 1.1
		}
		calls = append(calls, standin.Call{Step: "draft", TokensIn: tok, TokensOut: 80, Content: 20, FirstMs: int64(f), WorkMs: f + 20*5})
	}
	c := standin.Fit(standin.Config{}, calls)
	if math.Abs(c.FirstBaseMs-300) > 40 || math.Abs(c.FirstPerKTokMs-500) > 40 || c.FirstSigma < 0.05 || c.FirstSigma > 0.15 {
		t.Errorf("fit base %.0f per-1K %.0f sigma %.3f", c.FirstBaseMs, c.FirstPerKTokMs, c.FirstSigma)
	}
	if math.Abs(c.ContentTokMs-5) > 0.5 || c.ReasoningTok != 60 {
		t.Errorf("content %.1f ms/token, reasoning %d", c.ContentTokMs, c.ReasoningTok)
	}
}

// The real probe (results/calibration/20261111-214432_probe.jsonl): two simultaneous calls of
// 5,278 prompt + 50 max tokens. Groq admitted one (remaining 2672), refused the other with
// "Used 5318, Requested 5328 ... try again in 19.845s", retry-after 20, and charged nothing for it.
func TestStandinReplaysRealTPMProbe(t *testing.T) {
	c, _ := standinClient(t, standin.Config{TPM: 8000, RPD: 1000, PromptOverhead: 5274.5, PromptRatio: 1})
	r, err := c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 50}) // hi is 14 chars = 3.5 tokens
	if err != nil || r.Header.Get("x-ratelimit-remaining-tokens") != "2672" {
		t.Fatalf("first call: %v, remaining %q", err, r.Header.Get("x-ratelimit-remaining-tokens"))
	}
	r, err = c.ChatStream(context.Background(), hi, llm.Options{MaxTokens: 50})
	var ae *llm.APIError
	if !errors.As(err, &ae) || ae.Status != 429 || ae.RetryAfter != 20*time.Second || !strings.Contains(ae.Body, "Requested 5328") {
		t.Fatalf("second call must be refused like Groq's (retry-after 20): %v", err)
	}
	if got := r.Header.Get("x-ratelimit-remaining-requests"); got != "999" {
		t.Errorf("a refused call must not use a daily request: remaining %s", got)
	}
	if d, _ := time.ParseDuration(r.Header.Get("x-ratelimit-reset-tokens")); d < 39*time.Second || d > 40*time.Second {
		t.Errorf("reset-tokens is the time until the bucket is full (5328/133.3 = 40 s): %s", d)
	}
}
