// Package standin is the stand-in model service: an OpenAI-style streaming chat endpoint we
// control, for loads Groq's free tier cannot take. It can be slow, fail, refuse at a limit (429),
// get slower under load, and stream its reply a piece at a time.
//
// Limits follow what we measured on Groq (see admit and Docs/STANDIN.md). Latency is fitted to real calls (Fit). The load slowdown and
// the fault rates are assumptions: a free tier cannot be loaded to measure them.
package standin

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/Praroop1435/Vaani/internal/fakellm"
)

type Config struct {
	Model         string
	RPM, TPM, RPD int
	RPDUsed       int // daily requests already used at start (0: a full budget)

	// Latency at no load, fitted from real calls: first content token after
	// FirstBaseMs + FirstPerKTokMs * prompt_tokens/1000, times a lognormal(0, FirstSigma) factor.
	FirstBaseMs, FirstPerKTokMs, FirstSigma float64
	ContentTokMs                            float64 // between content tokens once streaming
	ReasoningTok                            int     // hidden reasoning tokens added to completion usage
	// Provider prompt tokens = PromptOverhead + PromptRatio * chars/4 (measured: 71 + 0.889x).
	PromptOverhead, PromptRatio float64

	// Load (assumed): up to Capacity requests in flight run at full speed; each one beyond it
	// makes every request SlowPerExtra slower (0.3 = +30%).
	Capacity     int
	SlowPerExtra float64

	// Faults (assumed rates, 0..1): FailRate answers HTTP 500 before streaming; StallRate sends the
	// first token only after StallMs ("succeeds slowly"); DropRate cuts the stream half way.
	FailRate, StallRate, DropRate float64
	StallMs                       float64

	Seed  uint64
	Reply func(system, user string) string // nil: "NOT IN PAPERS"
}

// Stats are the stand-in's own counters for one run.
type Stats struct {
	Requests, OK, Limited429, Failed500, Stalled, Dropped int
	PeakInFlight                                          int
}

type Server struct {
	cfg Config

	mu       sync.Mutex
	rng      *rand.Rand
	reqs     []time.Time // accepted calls in the last minute (RPM, sliding window: assumed)
	tokDebt  float64     // TPM token bucket: tokens charged and not yet refilled (measured)
	tokAt    time.Time
	dayLeft  float64
	dayAt    time.Time
	inFlight int
	st       Stats
	now      func() time.Time
}

func New(c Config) *Server {
	if c.PromptRatio == 0 {
		c.PromptRatio = 1
	}
	if c.Model == "" {
		c.Model = "qwen/qwen3.8-27b"
	}
	s := &Server{cfg: c, rng: rand.New(rand.NewPCG(c.Seed, c.Seed^0x9e3779b97f4a7c15)), now: time.Now}
	s.dayLeft, s.dayAt, s.tokAt = float64(c.RPD-c.RPDUsed), s.now(), s.now()
	return s
}

func (s *Server) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st
}

func (s *Server) refillEach() time.Duration { return 24 * time.Hour / time.Duration(s.cfg.RPD) }

// admit applies the limits as Groq does (measured 2026-11-11, results/calibration):
//   - TPM is a token bucket of TPM tokens refilling at TPM/60 per second; a call is charged
//     prompt + max_completion_tokens when admitted; x-ratelimit-reset-tokens = time until full;
//     retry-after = time until this call fits ("Used 5318, Requested 5328 ... try again in 19.845s").
//   - RPD refills one request every 24h/RPD.
//   - A refused call is charged nothing (remaining-requests and -tokens unchanged after a 429).
//
// It returns status 0 and the headers on success, or an HTTP status and a Groq-style message.
func (s *Server) admit(reserve int) (status int, msg string, h http.Header) {
	var retry time.Duration
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	s.st.Requests++
	s.dayLeft = math.Min(float64(s.cfg.RPD), s.dayLeft+float64(now.Sub(s.dayAt))/float64(s.refillEach()))
	s.dayAt = now
	rate := float64(s.cfg.TPM) / 60 // tokens per second
	s.tokDebt = math.Max(0, s.tokDebt-now.Sub(s.tokAt).Seconds()*rate)
	s.tokAt = now
	i := 0
	for i < len(s.reqs) && now.Sub(s.reqs[i]) >= time.Minute {
		i++
	}
	s.reqs = s.reqs[i:]
	limited := func(kind string, limit, used, req int, wait time.Duration) {
		s.st.Limited429++
		status, retry = http.StatusTooManyRequests, wait
		msg = fmt.Sprintf("Rate limit reached for model `%s` in organization `org_standin` service tier `on_demand` on %s: Limit %d, Used %d, Requested %d. Please try again in %s.",
			s.cfg.Model, kind, limit, used, req, wait.Round(time.Millisecond))
	}
	switch {
	case s.cfg.TPM > 0 && reserve > s.cfg.TPM: // not observed on Groq; message from its docs
		s.st.Limited429++
		status = http.StatusRequestEntityTooLarge
		msg = fmt.Sprintf("Request too large for model `%s` on tokens per minute (TPM): Limit %d, Requested %d, please reduce your message size and try again.",
			s.cfg.Model, s.cfg.TPM, reserve)
	case s.cfg.RPD > 0 && s.dayLeft < 1:
		limited("requests per day (RPD)", s.cfg.RPD, s.cfg.RPD, 1, time.Duration((1-s.dayLeft)*float64(s.refillEach())))
	case s.cfg.RPM > 0 && len(s.reqs) >= s.cfg.RPM:
		limited("requests per minute (RPM)", s.cfg.RPM, len(s.reqs), 1, s.reqs[len(s.reqs)-s.cfg.RPM].Add(time.Minute).Sub(now))
	case s.cfg.TPM > 0 && s.tokDebt+float64(reserve) > float64(s.cfg.TPM):
		over := s.tokDebt + float64(reserve) - float64(s.cfg.TPM)
		limited("tokens per minute (TPM)", s.cfg.TPM, int(s.tokDebt), reserve, time.Duration(over/rate*float64(time.Second)))
	default:
		s.reqs = append(s.reqs, now)
		s.tokDebt += float64(reserve)
		s.dayLeft--
	}
	h = http.Header{}
	h.Set("x-ratelimit-limit-requests", strconv.Itoa(s.cfg.RPD))
	h.Set("x-ratelimit-remaining-requests", strconv.Itoa(max(int(s.dayLeft), 0)))
	h.Set("x-ratelimit-reset-requests", (time.Duration((float64(s.cfg.RPD) - s.dayLeft) * float64(s.refillEach()))).Round(10*time.Millisecond).String())
	h.Set("x-ratelimit-limit-tokens", strconv.Itoa(s.cfg.TPM))
	h.Set("x-ratelimit-remaining-tokens", strconv.Itoa(max(s.cfg.TPM-int(math.Ceil(s.tokDebt)), 0)))
	h.Set("x-ratelimit-reset-tokens", time.Duration(s.tokDebt/rate*float64(time.Second)).Round(time.Millisecond).String())
	if status != 0 && retry > 0 {
		h.Set("retry-after", strconv.Itoa(int(math.Ceil(retry.Seconds()))))
	}
	return
}

func (s *Server) roll(p float64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return p > 0 && s.rng.Float64() < p
}

func (s *Server) normal() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rng.NormFloat64()
}

func (s *Server) count(f func(*Stats)) {
	s.mu.Lock()
	f(&s.st)
	s.mu.Unlock()
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet { // /models, used by the client's warm-up
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","data":[]}`)
		return
	}
	var req struct {
		Messages  []struct{ Role, Content string } `json:"messages"`
		MaxTokens int                              `json:"max_completion_tokens"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) == 0 {
		http.Error(w, `{"error":{"message":"bad request"}}`, http.StatusBadRequest)
		return
	}
	chars := 0
	for _, m := range req.Messages {
		chars += len(m.Content)
	}
	prompt := int(math.Round(s.cfg.PromptOverhead + float64(chars)/4*s.cfg.PromptRatio))
	maxTok := req.MaxTokens
	if maxTok == 0 {
		maxTok = 8192 // assumed provider default
	}
	status, msg, h := s.admit(prompt + maxTok)
	for k, v := range h {
		w.Header()[k] = v
	}
	if status != 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		js, _ := json.Marshal(map[string]any{"error": map[string]any{"message": msg, "type": "tokens", "code": "rate_limit_exceeded"}})
		w.Write(js)
		return
	}

	s.mu.Lock()
	s.inFlight++
	s.st.PeakInFlight = max(s.st.PeakInFlight, s.inFlight)
	load := 1 + s.cfg.SlowPerExtra*float64(max(s.inFlight-s.cfg.Capacity, 0))
	s.mu.Unlock()
	defer func() { s.mu.Lock(); s.inFlight--; s.mu.Unlock() }()

	first := (s.cfg.FirstBaseMs + s.cfg.FirstPerKTokMs*float64(prompt)/1000) * math.Exp(s.cfg.FirstSigma*s.normal()) * load
	if s.roll(s.cfg.FailRate) {
		s.count(func(st *Stats) { st.Failed500++ })
		sleep(r, first/2)
		http.Error(w, `{"error":{"message":"Internal Server Error","type":"internal_server_error"}}`, http.StatusInternalServerError)
		return
	}
	if s.roll(s.cfg.StallRate) {
		s.count(func(st *Stats) { st.Stalled++ })
		first = s.cfg.StallMs
	}
	reply := "NOT IN PAPERS"
	if s.cfg.Reply != nil && len(req.Messages) >= 2 {
		reply = s.cfg.Reply(req.Messages[0].Content, req.Messages[len(req.Messages)-1].Content)
	}
	drop := s.roll(s.cfg.DropRate)

	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	if fl != nil {
		fl.Flush()
	}
	if !sleep(r, first) {
		return
	}
	pieces := fakellm.Pieces(reply, 3)
	perPiece := s.cfg.ContentTokMs * 0.75 * load // 3 characters ≈ 0.75 token
	for i, p := range pieces {
		if drop && i == len(pieces)/2 {
			s.count(func(st *Stats) { st.Dropped++ })
			panic(http.ErrAbortHandler) // connection cut mid-reply, no [DONE]
		}
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", p)
		if fl != nil {
			fl.Flush()
		}
		if i < len(pieces)-1 && !sleep(r, perPiece) {
			return
		}
	}
	completion := len(reply)/4 + 1 + s.cfg.ReasoningTok
	fmt.Fprintf(w, "data: {\"choices\":[],\"x_groq\":{\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}}\n\ndata: [DONE]\n\n", prompt, completion)
	s.count(func(st *Stats) { st.OK++ })
}

// sleep waits ms, or returns false if the client went away.
func sleep(r *http.Request, ms float64) bool {
	if ms <= 0 {
		return true
	}
	t := time.NewTimer(time.Duration(ms * float64(time.Millisecond)))
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-r.Context().Done():
		return false
	}
}
