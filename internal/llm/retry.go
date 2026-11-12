package llm

import (
	"context"
	"errors"
	"math/rand/v2"
	"net"
	"strings"
	"time"
)

// Policy is the retry policy for one model call.
//
//   - Retry: 429 per-minute limits (after retry-after), 5xx, a first token that never comes, network errors.
//   - Never retry: 400/401/404/413 (the same request fails again), or the daily limit (it won't clear for hours).
//   - Never start a retry the request's deadline can't fit: a late answer is a given-up answer that also spent budget.
type Policy struct {
	MaxRetries     int           // retries after the first attempt
	BaseBackoff    time.Duration // first backoff, doubled each retry, with jitter
	MaxBackoff     time.Duration
	AttemptTimeout time.Duration // one attempt, start to last token
	MinUseful      time.Duration // a retry needs at least this much time left after waiting
}

var DefaultPolicy = Policy{MaxRetries: 2, BaseBackoff: 400 * time.Millisecond, MaxBackoff: 4 * time.Second,
	AttemptTimeout: 15 * time.Second, MinUseful: 2 * time.Second}

type Attempt struct {
	Status   int     // HTTP status, 0 for transport errors and timeouts
	Err      string  // "" when it succeeded
	BeforeMs float64 // backoff waited before this attempt
	Ms       float64 // attempt duration
}

type Stats struct {
	Attempts    []Attempt
	RetryWaitMs float64 // time spent backing off
	WorkMs      float64 // time spent in attempts, failed ones included
}

func (s Stats) Retries() int { return max(len(s.Attempts)-1, 0) }

// IsDaily reports a 429 caused by the per-day limit (Groq's message names "per day").
func IsDaily(err error) bool {
	var ae *APIError
	return errors.As(err, &ae) && ae.Status == 429 && strings.Contains(strings.ToLower(ae.Body), "per day")
}

// retryWait says whether err is worth retrying and how long to wait first.
func retryWait(err error, attempt int, p Policy) (time.Duration, bool) {
	backoff := min(p.BaseBackoff<<attempt, p.MaxBackoff)
	backoff = backoff/2 + time.Duration(rand.Int64N(int64(backoff/2)+1)) // jitter: spread out retries from a burst
	var ae *APIError
	switch {
	case errors.As(err, &ae):
		switch {
		case IsDaily(err):
			return 0, false
		case ae.Status == 429:
			if ae.RetryAfter > 0 {
				return ae.RetryAfter, true
			}
			return backoff, true
		case ae.Status >= 500 || ae.Status == 408:
			return backoff, true
		}
		return 0, false
	case errors.Is(err, ErrFirstTokenTimeout), errors.Is(err, context.DeadlineExceeded):
		return backoff, true
	}
	var ne net.Error
	if errors.As(err, &ne) || errors.Is(err, net.ErrClosed) {
		return backoff, true
	}
	return 0, false
}

// Do runs call under the policy. ctx carries the whole request's deadline; each attempt also gets
// AttemptTimeout. The returned Stats split time into working (attempts) and retrying (backoff).
func Do(ctx context.Context, p Policy, call func(ctx context.Context) (Result, error)) (Result, Stats, error) {
	var st Stats
	var before time.Duration
	for attempt := 0; ; attempt++ {
		actx, cancel := context.WithTimeout(ctx, p.AttemptTimeout)
		t0 := time.Now()
		res, err := call(actx)
		cancel()
		a := Attempt{BeforeMs: ms(before), Ms: ms(time.Since(t0))}
		st.WorkMs += a.Ms
		if err == nil {
			st.Attempts = append(st.Attempts, a)
			return res, st, nil
		}
		a.Err = err.Error()
		var ae *APIError
		if errors.As(err, &ae) {
			a.Status = ae.Status
		}
		st.Attempts = append(st.Attempts, a)
		if ctx.Err() != nil { // the whole request ran out of time
			return res, st, err
		}
		wait, ok := retryWait(err, attempt, p)
		if !ok || attempt >= p.MaxRetries {
			return res, st, err
		}
		if dl, has := ctx.Deadline(); has && time.Until(dl) < wait+p.MinUseful {
			return res, st, err
		}
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return res, st, ctx.Err()
		}
		st.RetryWaitMs += ms(wait)
		before = wait
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
