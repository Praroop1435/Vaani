package test

import (
	"context"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
	"github.com/Praroop1435/Vaani/internal/standin"
)

// The brief's "at 3pm the daily budget is gone". The first person's call meets the provider's
// per-day 429; from then on the governor knows and nobody's request reaches the provider. Each
// person gets this exact reply: why, when the next request and the full quota come back, and the
// passages to read meanwhile. Never a guessed answer.
var dailyReply = regexp.MustCompile(`^I can't write an answer right now: today's model budget \(1000 requests/day\) is used up\. ` +
	`It refills gradually, about one request every 1m26s: the next request can run in about 1m26s, ` +
	`and the full daily quota is back tomorrow at \d\d:\d\d\. ` +
	`The passages most likely to answer this are: ((Moshi|Attention Is All You Need|H-Net \(Dynamic Chunking\)), p\.[\d-]+, [^;]+)(; [^;]+)*\.$`)

func TestDailyBudgetGoneReplyIsExactAndCallsNothing(t *testing.T) {
	sv := standin.New(standin.Config{TPM: 8000, RPD: 1000, RPDUsed: 1000, FirstBaseMs: 10})
	srv := httptest.NewServer(sv)
	defer srv.Close()
	ix, err := retrieve.Load("../data/index", 300)
	if err != nil {
		t.Skip(err)
	}
	p := &pipeline.Pipeline{
		LLM: &llm.Client{BaseURL: srv.URL, Model: "openai/gpt-oss-120b"},
		Gov: governor.New(governor.Limits{RPM: 30, TPM: 8000, RPD: 1000}),
		Ret: &retrieve.Retriever{Embedder: embedder(t), Indexes: map[int]*retrieve.Index{300: ix}},
		Cfg: pipeline.Defaults,
	}
	for i, q := range []string{"What frame rate does the Mimi codec run at?", "What dropout rate does the base Transformer model use?"} {
		var shown strings.Builder
		tr := p.Answer(context.Background(), "daily", q, func(s string) { shown.WriteString(s) })
		if tr.Status != "rate_limited" || tr.Refused || !dailyReply.MatchString(shown.String()) || shown.String() != tr.Answer {
			t.Errorf("person %d saw %q (status %s)", i+1, shown.String(), tr.Status)
		}
		if tr.FirstTokenMs < 0 || tr.TotalMs > 2000 {
			t.Errorf("person %d: the reply must come at once, first %d ms total %d ms", i+1, tr.FirstTokenMs, tr.TotalMs)
		}
	}
	if st := sv.Stats(); st.Requests != 1 || st.Limited429 != 1 {
		t.Errorf("only the first person's call may reach the provider (its daily 429): %+v", st)
	}
}
