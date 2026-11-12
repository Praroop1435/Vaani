// Package llm talks to any OpenAI-style chat endpoint (Groq or our stand-in) and streams the reply.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptrace"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

const GroqBaseURL = "https://api.groq.com/openai/v1"

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Client struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client // nil means http.DefaultClient
}

type Result struct {
	Text         string
	FirstTokenMs int64 // -1 if no content token arrived
	TotalMs      int64
	TokensIn     int
	TokensOut    int
	Header       http.Header // carries x-ratelimit-* from Groq
	Net          NetTiming
}

// NetTiming splits the wait before the first byte: a cold call pays DNS + TCP + TLS, a call on
// a kept-alive connection does not. Times are from the start of the request.
type NetTiming struct {
	Reused    bool  `json:"reused"`
	DNSMs     int64 `json:"dns_ms"`
	ConnectMs int64 `json:"connect_ms"`
	TLSMs     int64 `json:"tls_ms"`
	TTFBMs    int64 `json:"ttfb_ms"` // response headers arrived
}

// APIError is a non-200 reply; RetryAfter is set when the server sends retry-after (429).
type APIError struct {
	Status     int
	RetryAfter time.Duration
	Body       string
}

func (e *APIError) Error() string { return fmt.Sprintf("llm: HTTP %d: %s", e.Status, e.Body) }

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type chunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *usage `json:"usage"`
	XGroq *struct {
		Usage *usage `json:"usage"`
	} `json:"x_groq"`
}

// Options tune one call. Zero values mean provider defaults.
type Options struct {
	MaxTokens int               // cap on completion tokens (hidden reasoning counts too)
	OnToken   func(text string) // called for every content piece as it streams in
	// FirstTokenTimeout aborts a call whose first content token is this late: some calls
	// "succeed slowly" instead of failing fast, and waiting them out burns the person's patience.
	FirstTokenTimeout time.Duration
}

// ErrFirstTokenTimeout means no content token arrived within Options.FirstTokenTimeout.
var ErrFirstTokenTimeout = errors.New("llm: first token too slow")

// Chat sends one streamed chat request and times the first content token and the whole reply.
func (c *Client) Chat(ctx context.Context, msgs []Message) (Result, error) {
	return c.ChatStream(ctx, msgs, Options{})
}

// ChatStream is Chat with options; OnToken sees the reply a piece at a time.
func (c *Client) ChatStream(ctx context.Context, msgs []Message, opt Options) (Result, error) {
	payload := map[string]any{
		"model":             c.Model,
		"messages":          msgs,
		"stream":            true,
		"reasoning_effort":  "low", // gpt-oss thinks before answering; keep it short for first-token time
		"include_reasoning": false,
		"temperature":       0,
	}
	if opt.MaxTokens > 0 {
		payload["max_completion_tokens"] = opt.MaxTokens
	}
	body, _ := json.Marshal(payload)
	var slow atomic.Bool
	if opt.FirstTokenTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		defer cancel()
		timer := time.AfterFunc(opt.FirstTokenTimeout, func() { slow.Store(true); cancel() })
		defer timer.Stop()
		onTok := opt.OnToken
		opt.OnToken = func(t string) {
			timer.Stop()
			if onTok != nil {
				onTok(t)
			}
		}
	}
	res, err := c.stream(ctx, body, opt)
	if err != nil && slow.Load() && res.FirstTokenMs < 0 {
		return res, ErrFirstTokenTimeout
	}
	return res, err
}

func (c *Client) stream(ctx context.Context, body []byte, opt Options) (Result, error) {
	res := Result{FirstTokenMs: -1}
	t0 := time.Now()
	since := func() int64 { return time.Since(t0).Milliseconds() }
	var dnsAt, connAt, tlsAt time.Time
	ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
		GotConn:              func(i httptrace.GotConnInfo) { res.Net.Reused = i.Reused },
		DNSStart:             func(httptrace.DNSStartInfo) { dnsAt = time.Now() },
		DNSDone:              func(httptrace.DNSDoneInfo) { res.Net.DNSMs = time.Since(dnsAt).Milliseconds() },
		ConnectStart:         func(string, string) { connAt = time.Now() },
		ConnectDone:          func(string, string, error) { res.Net.ConnectMs = time.Since(connAt).Milliseconds() },
		TLSHandshakeStart:    func() { tlsAt = time.Now() },
		TLSHandshakeDone:     func(tls.ConnectionState, error) { res.Net.TLSMs = time.Since(tlsAt).Milliseconds() },
		GotFirstResponseByte: func() { res.Net.TTFBMs = since() },
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return res, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	hc := c.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	start := time.Now()
	resp, err := hc.Do(req)
	if err != nil {
		return res, err
	}
	defer resp.Body.Close()
	res.Header = resp.Header

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		secs, _ := strconv.ParseFloat(resp.Header.Get("retry-after"), 64)
		res.TotalMs = time.Since(start).Milliseconds()
		return res, &APIError{Status: resp.StatusCode, RetryAfter: time.Duration(secs * float64(time.Second)), Body: string(b)}
	}

	var text strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var ch chunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			return res, fmt.Errorf("llm: bad stream chunk: %w", err)
		}
		for _, choice := range ch.Choices {
			if choice.Delta.Content != "" && res.FirstTokenMs < 0 {
				res.FirstTokenMs = time.Since(start).Milliseconds()
			}
			text.WriteString(choice.Delta.Content)
			if opt.OnToken != nil && choice.Delta.Content != "" {
				opt.OnToken(choice.Delta.Content)
			}
		}
		u := ch.Usage
		if u == nil && ch.XGroq != nil {
			u = ch.XGroq.Usage
		}
		if u != nil {
			res.TokensIn, res.TokensOut = u.PromptTokens, u.CompletionTokens
		}
	}
	res.Text = text.String()
	res.TotalMs = time.Since(start).Milliseconds()
	return res, sc.Err()
}

// Warm opens the connection (DNS + TCP + TLS) before the first real call, so the first person's
// question doesn't pay for it (measured on Groq: 1,015 ms of a 1,578 ms first call). It sends one
// unauthenticated GET to <base>/models; no API key is sent, so it cannot count against the quota.
// Returns the HTTP status (401 expected from Groq) and how long it took.
func (c *Client) Warm(ctx context.Context) (int, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/models", nil)
	if err != nil {
		return 0, 0, err
	}
	hc := c.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	t0 := time.Now()
	resp, err := hc.Do(req)
	if err != nil {
		return 0, time.Since(t0), err
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10)) // drain so the connection goes back to the pool
	resp.Body.Close()
	return resp.StatusCode, time.Since(t0), nil
}
