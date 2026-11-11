package test

import (
	"bufio"
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/llm"
)

// Live test: spends 1 real Groq request. Runs only with GROQ_LIVE=1 so `go test ./...` stays free.
//   GROQ_LIVE=1 go test ./test -run Live -v

func TestLiveGroqGptOss120b(t *testing.T) {
	if os.Getenv("GROQ_LIVE") != "1" {
		t.Skip("set GROQ_LIVE=1 to spend one real Groq request")
	}
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		key = readEnvFile(t, "../.env", "GROQ_API_KEY")
	}
	if key == "" {
		t.Fatal("GROQ_API_KEY not in environment or ../.env")
	}

	c := &llm.Client{BaseURL: llm.GroqBaseURL, APIKey: key, Model: "openai/gpt-oss-120b"}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	res, err := c.Chat(ctx, []llm.Message{{Role: "user", Content: "Reply with exactly one word: pong"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.ToLower(res.Text), "pong") {
		t.Errorf("text = %q, want pong", res.Text)
	}
	if res.FirstTokenMs < 0 || res.TokensIn == 0 || res.TokensOut == 0 {
		t.Errorf("missing measurements: first=%dms in=%d out=%d", res.FirstTokenMs, res.TokensIn, res.TokensOut)
	}
	t.Logf("text=%q first_token=%dms total=%dms tokens_in=%d tokens_out=%d", res.Text, res.FirstTokenMs, res.TotalMs, res.TokensIn, res.TokensOut)
	for _, h := range []string{"x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"} {
		if res.Header.Get(h) == "" {
			t.Errorf("header %s missing", h)
		}
		t.Logf("%s: %s", h, res.Header.Get(h))
	}
}

func readEnvFile(t *testing.T, path, name string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if v, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), name+"="); ok {
			return strings.Trim(v, `"'`)
		}
	}
	return ""
}
