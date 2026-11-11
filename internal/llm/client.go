// Package llm talks to any OpenAI-style chat endpoint (Groq or our stand-in) and streams the reply.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
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

// Chat sends one streamed chat request and times the first content token and the whole reply.
func (c *Client) Chat(ctx context.Context, msgs []Message) (Result, error) {
	res := Result{FirstTokenMs: -1}
	body, _ := json.Marshal(map[string]any{
		"model":             c.Model,
		"messages":          msgs,
		"stream":            true,
		"reasoning_effort":  "low", // gpt-oss thinks before answering; keep it short for first-token time
		"include_reasoning": false,
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
