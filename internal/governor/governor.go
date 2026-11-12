// Package governor keeps model calls inside the provider's limits: requests per minute (RPM),
// tokens per minute (TPM) and requests per day (RPD), and corrects itself from the provider's
// x-ratelimit-* headers, because the limits "can change without notice".
//
// TPM is a token bucket, as measured on Groq (Docs/STANDIN.md): TPM tokens that refill
// continuously at TPM/60 per second; a call is charged prompt + max_completion_tokens when it is
// admitted, and goes through as soon as the available tokens cover that cost. (v1 counted tokens
// over a 60 s sliding window and, after a header, waited for the whole bucket to refill: more
// cautious than Groq, so it refused calls Groq would have taken.)
package governor

import (
	"context"
	"errors"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"
)

var (
	ErrDailyBudget   = errors.New("governor: daily request budget used up")
	ErrDeadlineWait  = errors.New("governor: waiting for the per-minute limit would miss the request deadline")
	ErrTooLargeAlone = errors.New("governor: request needs more tokens than one minute allows")
)

// Version names the limiter logic, recorded with every load run so results never mix versions.
// v1 (sliding 60 s window, waited for a full bucket) produced the steady and burst runs of
// 2026-11-11 21:49; v2 is the measured token bucket.
const Version = "v2-token-bucket"

type Limits struct {
	RPM, TPM, RPD int
}

type Governor struct {
	mu  sync.Mutex
	lim Limits
	win []time.Time // calls in the last 60 s (RPM: a sliding window, assumed; not measured)

	tokDebt float64   // TPM bucket: tokens charged and not yet refilled
	tokAt   time.Time // when tokDebt was last brought up to date

	dayWin []time.Time // our requests in the last 24 h (sliding, like the provider's refill)

	// last values the provider reported
	srvDayLeft   int // -1 unknown
	srvDayLeftAt time.Time
	srvDayReset  time.Time

	waiting, peakWaiting int

	now   func() time.Time
	sleep func(context.Context, time.Duration) error
}

func New(l Limits) *Governor {
	return &Governor{lim: l, srvDayLeft: -1, now: time.Now, sleep: sleepCtx, tokAt: time.Now()}
}

// SetClock replaces the clock and the sleep, for tests that replay real timings.
func (g *Governor) SetClock(now func() time.Time, sleep func(context.Context, time.Duration) error) {
	g.mu.Lock()
	g.now, g.sleep, g.tokAt = now, sleep, now()
	g.mu.Unlock()
}

func (g *Governor) tokRate() float64 { return float64(g.lim.TPM) / 60 } // tokens per second

// refill brings the bucket up to now.
func (g *Governor) refill(now time.Time) {
	if g.lim.TPM > 0 {
		g.tokDebt = max(0, g.tokDebt-now.Sub(g.tokAt).Seconds()*g.tokRate())
	}
	g.tokAt = now
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	select {
	case <-time.After(d):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (g *Governor) prune(now time.Time) {
	i := 0
	for i < len(g.win) && now.Sub(g.win[i]) >= time.Minute {
		i++
	}
	g.win = g.win[i:]
	j := 0
	for j < len(g.dayWin) && now.Sub(g.dayWin[j]) >= 24*time.Hour {
		j++
	}
	g.dayWin = g.dayWin[j:]
}

// DailyLeft is the smaller of our own count and the provider's last report.
func (g *Governor) DailyLeft() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.dailyLeft()
}

func (g *Governor) dailyLeft() int {
	left := g.lim.RPD - len(g.dayWin)
	if g.srvDayLeft >= 0 && g.lim.RPD > 0 {
		// the provider's count, plus what has refilled since it told us (one per 24h/limit)
		refilled := int(g.now().Sub(g.srvDayLeftAt) / (24 * time.Hour / time.Duration(g.lim.RPD)))
		left = min(left, min(g.srvDayLeft+refilled, g.lim.RPD))
	}
	return left
}

// Daily describes the per-day request budget as Groq runs it. Measured 2026-11-11:
// x-ratelimit-reset-requests grew by 86.4 s per request (1m26.4s after 1, 5m45.6s after 4) with a
// 1000/day limit, so the budget refills continuously, one request every 24h/limit, and the header
// is the time until it is full again, not until the next request.
type Daily struct {
	Limit      int
	Left       int
	RefillEach time.Duration // one request comes back this often
	NextIn     time.Duration // until the next request can be sent (0 if one is available)
	FullAt     time.Time     // whole quota back (zero if the provider has not said)
}

func (g *Governor) Daily() Daily {
	g.mu.Lock()
	defer g.mu.Unlock()
	d := Daily{Limit: g.lim.RPD, Left: max(g.dailyLeft(), 0), FullAt: g.srvDayReset}
	if g.lim.RPD > 0 {
		d.RefillEach = 24 * time.Hour / time.Duration(g.lim.RPD)
	}
	if d.Left == 0 {
		d.NextIn = d.RefillEach
	}
	return d
}

// PeakWaiting is the most calls that were queued behind the per-minute limit at once.
func (g *Governor) PeakWaiting() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.peakWaiting
}

// waitFor says how long until a call of `tokens` fits the per-minute limits.
func (g *Governor) waitFor(now time.Time, tokens int) time.Duration {
	var wait time.Duration
	if g.lim.RPM > 0 && len(g.win) >= g.lim.RPM {
		wait = max(wait, g.win[len(g.win)-g.lim.RPM].Add(time.Minute).Sub(now))
	}
	if g.lim.TPM > 0 {
		if short := float64(tokens) - (float64(g.lim.TPM) - g.tokDebt); short > 0 { // go when available >= cost
			wait = max(wait, time.Duration(math.Ceil(short/g.tokRate()*1e9)))
		}
	}
	return wait
}

// Reserve blocks until a call of about `tokens` fits the limits, then records it.
// It returns how long it waited. It fails fast instead of waiting past ctx's deadline.
func (g *Governor) Reserve(ctx context.Context, tokens int) (time.Duration, error) {
	start := g.now()
	g.mu.Lock()
	if g.lim.TPM > 0 && tokens > g.lim.TPM {
		g.mu.Unlock()
		return 0, ErrTooLargeAlone
	}
	g.waiting++
	g.peakWaiting = max(g.peakWaiting, g.waiting)
	defer func() { g.mu.Lock(); g.waiting--; g.mu.Unlock() }()
	for {
		now := g.now()
		g.prune(now)
		g.refill(now)
		if g.dailyLeft() <= 0 { // fail now: the next slot is ~86 s away, longer than anyone waits
			g.mu.Unlock()
			return now.Sub(start), ErrDailyBudget
		}
		wait := g.waitFor(now, tokens)
		if wait <= 0 {
			g.win = append(g.win, now)
			g.tokDebt += float64(tokens)
			g.dayWin = append(g.dayWin, now)
			if g.srvDayLeft > 0 {
				g.srvDayLeft--
			}
			g.mu.Unlock()
			return now.Sub(start), nil
		}
		if dl, ok := ctx.Deadline(); ok && now.Add(wait).After(dl) {
			g.mu.Unlock()
			return now.Sub(start), ErrDeadlineWait
		}
		g.mu.Unlock()
		if err := g.sleep(ctx, wait); err != nil {
			return g.now().Sub(start), err
		}
		g.mu.Lock()
	}
}

// Observe reads Groq's headers. On Groq, *-requests are per day and *-tokens per minute.
func (g *Governor) Observe(h http.Header) {
	if h == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	now := g.now()
	if v, err := strconv.Atoi(h.Get("x-ratelimit-remaining-requests")); err == nil {
		g.srvDayLeft, g.srvDayLeftAt = v, now
	}
	if d, err := time.ParseDuration(h.Get("x-ratelimit-reset-requests")); err == nil {
		g.srvDayReset = now.Add(d)
	}
	g.refill(now)
	if v, err := strconv.Atoi(h.Get("x-ratelimit-limit-tokens")); err == nil && v > 0 {
		g.lim.TPM = v // the provider's current limit beats our configured one
	}
	// The provider's count of the bucket, which includes its own (exact) prompt token count. Keep
	// the larger debt: ours also holds calls we admitted that the provider has not seen yet.
	if v, err := strconv.Atoi(h.Get("x-ratelimit-remaining-tokens")); err == nil && g.lim.TPM > 0 {
		g.tokDebt = max(g.tokDebt, float64(g.lim.TPM-v))
	}
	if v, err := strconv.Atoi(h.Get("x-ratelimit-limit-requests")); err == nil && v > 0 {
		g.lim.RPD = v
	}
}

// SetLimits replaces the limits, e.g. to halve RPM for a load test.
func (g *Governor) SetLimits(l Limits) {
	g.mu.Lock()
	g.lim = l
	g.mu.Unlock()
}
