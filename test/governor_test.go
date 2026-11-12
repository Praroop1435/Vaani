package test

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/governor"
)

// fakeClock lets a test replay real timings without waiting for them.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time { return c.t }
func (c *fakeClock) sleep(_ context.Context, d time.Duration) error {
	c.t = c.t.Add(d)
	return nil
}

// Replays the real Groq probe (results/calibration/20261111-214432_probe.jsonl): two calls of
// 5,278 prompt + 50 max = 5,328 tokens sent together. Groq admitted the first, and 75 ms later
// refused the second with "Limit 8000, Used 5318, Requested 5328. Please try again in 19.845s":
// 5,328 - (8,000 - 5,318) = 2,646 tokens short, at 133.3 tokens/s = 19.845 s. The governor must
// hold the second call exactly that long: not less (Groq would 429) and not the ~40 s until the
// bucket is full (limiter v1), which refused calls Groq would have taken.
func TestGovernorReplaysRealTPMProbe(t *testing.T) {
	clk := &fakeClock{t: time.Date(2026, 11, 11, 16, 15, 37, 914_000_000, time.UTC)}
	g := governor.New(governor.Limits{RPM: 30, TPM: 8000, RPD: 1000})
	g.SetClock(clk.now, clk.sleep)
	ctx := context.Background()
	if w, err := g.Reserve(ctx, 5328); err != nil || w != 0 {
		t.Fatalf("first call: waited %v, %v", w, err)
	}
	h := http.Header{} // Groq's headers on the admitted call
	h.Set("x-ratelimit-limit-tokens", "8000")
	h.Set("x-ratelimit-remaining-tokens", "2672")
	h.Set("x-ratelimit-reset-tokens", "39.96s")
	g.Observe(h)
	clk.t = clk.t.Add(75 * time.Millisecond)

	// a deadline 19.8 s away is too short: fail fast instead of a 429
	short, cancel := context.WithDeadline(ctx, clk.t.Add(19800*time.Millisecond))
	defer cancel()
	if _, err := g.Reserve(short, 5328); !errors.Is(err, governor.ErrDeadlineWait) {
		t.Fatalf("19.8 s left is less than Groq's 19.845 s: got %v", err)
	}
	waited, err := g.Reserve(ctx, 5328)
	if err != nil {
		t.Fatal(err)
	}
	if d := waited - 19845*time.Millisecond; d < -10*time.Millisecond || d > 10*time.Millisecond {
		t.Errorf("waited %v, Groq's retry point is 19.845 s", waited)
	}
}

// The call goes through as soon as available >= cost, including exactly equal.
func TestGovernorAdmitsWhenAvailableCoversCost(t *testing.T) {
	clk := &fakeClock{t: time.Unix(0, 0)}
	g := governor.New(governor.Limits{TPM: 6000, RPD: 1000}) // 100 tokens/s
	g.SetClock(clk.now, clk.sleep)
	ctx := context.Background()
	if w, _ := g.Reserve(ctx, 5000); w != 0 {
		t.Fatal("first call must not wait")
	}
	if w, _ := g.Reserve(ctx, 1000); w != 0 {
		t.Errorf("1,000 available covers a 1,000 cost exactly: waited %v", w)
	}
	clk.t = clk.t.Add(3 * time.Second) // 300 tokens refilled
	if w, _ := g.Reserve(ctx, 500); w < 1999*time.Millisecond || w > 2001*time.Millisecond {
		t.Errorf("300 available for 500: must wait (500-300)/100 = 2 s, waited %v", w)
	}
}
