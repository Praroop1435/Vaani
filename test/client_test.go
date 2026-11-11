package test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/llm"
)

// Offline tests: a fake server, no Groq budget spent.

func TestStreamsTextAndTimesFirstToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("x-ratelimit-remaining-tokens", "7900")
		f := w.(http.Flusher)
		for _, piece := range []string{"Hel", "lo", " world"} {
			time.Sleep(30 * time.Millisecond)
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", piece)
			f.Flush()
		}
		fmt.Fprint(w, "data: {\"choices\":[],\"x_groq\":{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := &llm.Client{BaseURL: srv.URL, Model: "fake"}
	res, err := c.Chat(context.Background(), []llm.Message{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "Hello world" {
		t.Errorf("text = %q", res.Text)
	}
	if res.TokensIn != 12 || res.TokensOut != 3 {
		t.Errorf("tokens = %d/%d, want 12/3", res.TokensIn, res.TokensOut)
	}
	if res.FirstTokenMs < 25 || res.TotalMs < res.FirstTokenMs+50 {
		t.Errorf("timing first=%dms total=%dms: first token must come before the rest", res.FirstTokenMs, res.TotalMs)
	}
	if res.Header.Get("x-ratelimit-remaining-tokens") != "7900" {
		t.Error("rate-limit headers not kept")
	}
}

func TestRateLimitReturnsRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("retry-after", "2.5")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"message":"Rate limit reached"}}`)
	}))
	defer srv.Close()

	c := &llm.Client{BaseURL: srv.URL, Model: "fake"}
	_, err := c.Chat(context.Background(), []llm.Message{{Role: "user", Content: "hi"}})
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *llm.APIError, got %v", err)
	}
	if apiErr.Status != 429 || apiErr.RetryAfter != 2500*time.Millisecond {
		t.Errorf("status=%d retryAfter=%v", apiErr.Status, apiErr.RetryAfter)
	}
}
