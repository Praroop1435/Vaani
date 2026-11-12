package test

import (
	"context"
	"net/http"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

// Semantic answer cache. Uses the local embedder and the fake model: no Groq budget.

const headsQ = "How many parallel attention heads does the base Transformer use?"

var goodReply = func() func(http.ResponseWriter) {
	return streamReply("The Transformer uses multi-head attention [S1].", 300*time.Millisecond) // delay so a hit saves time
}

func TestKeyTermsCatchLookAlikes(t *testing.T) {
	eq := func(a, b string) bool { return slices.Equal(pipeline.KeyTerms(a), pipeline.KeyTerms(b)) }
	same := [][2]string{
		{headsQ, "how many parallel attention heads does the base transformer use"},
		{headsQ, "In the base Transformer, how many parallel attention heads are used?"},
		{"What GPUs were the Moshi models trained on?", "Which GPU was Moshi trained on?"},
	}
	diff := [][2]string{
		{headsQ, "How many parallel attention heads does the big Transformer use?"},
		{"What BLEU does the big Transformer reach on English-to-German?", "What BLEU does the big Transformer reach on English-to-French?"},
		{"What BLEU does the big Transformer reach on English-to-German?", "What BLEU does the big Transformer reach on German-to-English?"},
		{"What accuracy does the 2-stage H-Net reach?", "What accuracy does the 1-stage H-Net reach?"},
		{"What latency does Moshi achieve?", "What latency does Mimi achieve?"},
	}
	for _, p := range same {
		if !eq(p[0], p[1]) {
			t.Errorf("same question, different terms: %v vs %v", pipeline.KeyTerms(p[0]), pipeline.KeyTerms(p[1]))
		}
	}
	for _, p := range diff {
		if eq(p[0], p[1]) {
			t.Errorf("different questions, same terms %v: %q vs %q", pipeline.KeyTerms(p[0]), p[0], p[1])
		}
	}
}

func TestCacheExactRepeatAndParaphraseSkipModel(t *testing.T) {
	fm := &fakeModel{replies: []func(http.ResponseWriter){goodReply()}}
	p := newPipeline(t, fm, cfgNoUnderstand())
	c := pipeline.NewCache(pipeline.DefaultCacheThreshold)
	ctx := context.Background()

	first, cr := p.AnswerCached(ctx, c, "c1", headsQ, nil)
	if cr.Verdict != "miss" || !cr.Stored {
		t.Fatalf("first ask: verdict %s stored %v (%s), answer %q", cr.Verdict, cr.Stored, cr.StoreReason, first.Answer)
	}
	for _, q := range []string{headsQ, "In the base Transformer, how many parallel attention heads are used?"} {
		var shown string
		tr, cr := p.AnswerCached(ctx, c, "c2", q, func(s string) { shown += s })
		if cr.Verdict != "hit" || tr.Answer != first.Answer || shown != first.Answer || len(tr.Citations) == 0 {
			t.Errorf("%q: verdict %s sim %.3f, answer %q", q, cr.Verdict, cr.Sim, tr.Answer)
		}
		if tr.TotalMs >= first.TotalMs {
			t.Errorf("hit took %d ms, the original %d ms", tr.TotalMs, first.TotalMs)
		}
	}
	if n := fm.calls.Load(); n != 1 {
		t.Errorf("model called %d times, want 1 (hits must not call it)", n)
	}
	if s := c.Stats(); s.Hits != 2 || s.Misses != 1 || s.Stored != 1 || s.SavedMs <= 0 {
		t.Errorf("stats %+v", s)
	}
}

func TestCacheFalseMatchGoesToModel(t *testing.T) {
	fm := &fakeModel{replies: []func(http.ResponseWriter){goodReply(), goodReply()}}
	p := newPipeline(t, fm, cfgNoUnderstand())
	c := pipeline.NewCache(0.90) // base vs big scores 0.907: cosine alone would serve the wrong answer
	ctx := context.Background()
	p.AnswerCached(ctx, c, "f1", headsQ, nil)
	_, cr := p.AnswerCached(ctx, c, "f2", "How many parallel attention heads does the big Transformer use?", nil)
	if cr.Verdict != "blocked" || cr.Sim < c.Threshold { // cosine says "same", the guard says no
		t.Errorf("verdict %s sim %.3f, want blocked over %.2f", cr.Verdict, cr.Sim, c.Threshold)
	}
	if fm.calls.Load() != 2 {
		t.Errorf("model called %d times, want 2", fm.calls.Load())
	}
}

func TestCacheStoresOnlyVerifiedAnswers(t *testing.T) {
	fm := &fakeModel{replies: []func(http.ResponseWriter){
		streamReply("NOT IN PAPERS", 0), streamReply("NOT IN PAPERS", 0), // refusal (one re-draft allowed)
		streamReply("It has 999 heads [S1].", 0), streamReply("It has 999 heads [S1].", 0), // nothing verifies
	}}
	p := newPipeline(t, fm, cfgNoUnderstand())
	c := pipeline.NewCache(pipeline.DefaultCacheThreshold)
	ctx := context.Background()
	for _, q := range []string{"What BLEU score does the Transformer reach on English-to-Spanish translation?", headsQ} {
		tr, cr := p.AnswerCached(ctx, c, "v", q, nil)
		if cr.Stored {
			t.Errorf("%q stored: refused %v dropped %v answer %q", q, tr.Refused, tr.Dropped, tr.Answer)
		}
	}
	if c.Len() != 0 {
		t.Errorf("%d entries, want 0", c.Len())
	}
	// unit check of the gate for states the fake model cannot easily produce
	vec := make([]float32, 768)
	ok := pipeline.Trace{Status: "ok", Answer: "a", Citations: []pipeline.Citation{{Tag: "S1"}}}
	for name, tr := range map[string]pipeline.Trace{
		"incomplete":   with(ok, func(t *pipeline.Trace) { t.Missing = []string{"pretraining tokens"} }),
		"cut short":    with(ok, func(t *pipeline.Trace) { t.Degraded = []string{"answer_cut_short"} }),
		"rate limited": with(ok, func(t *pipeline.Trace) { t.Status = "rate_limited" }),
		"no citations": with(ok, func(t *pipeline.Trace) { t.Citations = nil }),
	} {
		if stored, _ := c.Store(vec, "fp", tr); stored {
			t.Errorf("%s answer stored", name)
		}
	}
	if stored, why := c.Store(vec, "fp", with(ok, func(t *pipeline.Trace) { t.Degraded = []string{"understand_skipped_low_budget"} })); !stored {
		t.Errorf("verified answer without the rewrite not stored: %s", why)
	}
}

func with(tr pipeline.Trace, f func(*pipeline.Trace)) pipeline.Trace { f(&tr); return tr }

func TestCacheInvalidatesOnIndexOrSettingChange(t *testing.T) {
	fm := &fakeModel{}
	p := newPipeline(t, fm, cfgNoUnderstand())
	ctx := context.Background()
	qv, err := p.Ret.EmbedQuery(ctx, headsQ)
	if err != nil {
		t.Fatal(err)
	}
	ok := pipeline.Trace{Question: headsQ, Status: "ok", Answer: "8 heads", Citations: []pipeline.Citation{{Tag: "S1"}}}
	fresh := func() *pipeline.Cache {
		c := pipeline.NewCache(pipeline.DefaultCacheThreshold)
		c.Store(qv, p.Fingerprint(), ok)
		return c
	}
	hit := func(c *pipeline.Cache) bool { _, _, v := c.Lookup(qv, headsQ, p.Fingerprint()); return v == "hit" }

	if !hit(fresh()) {
		t.Fatal("baseline: stored answer not found")
	}
	// retrieval setting changes
	c := fresh()
	p.Cfg.Setting = retrieve.C
	if hit(c) || c.Stats().Stale != 1 || c.Len() != 0 {
		t.Errorf("setting changed: stats %+v", c.Stats())
	}
	p.Cfg.Setting = retrieve.A
	c = fresh()
	p.Cfg.Setting.K = 6
	if hit(c) {
		t.Error("k changed: still a hit")
	}
	p.Cfg.Setting.K = retrieve.A.K
	// source documents re-ingested: the index files hash differently
	c = fresh()
	ix := p.Ret.Indexes[300]
	saved := ix.Hash
	ix.Hash = "re-ingested"
	if hit(c) {
		t.Error("index changed: still a hit")
	}
	ix.Hash = saved
	// model changes
	c = fresh()
	p.LLM.Model = "other"
	if hit(c) {
		t.Error("model changed: still a hit")
	}
	p.LLM.Model = "fake"
	// persists across runs, and Invalidate empties it
	path := filepath.Join(t.TempDir(), "cache.json")
	if err := fresh().Save(path); err != nil {
		t.Fatal(err)
	}
	c, err = pipeline.LoadCache(path, pipeline.DefaultCacheThreshold)
	if err != nil || !hit(c) {
		t.Errorf("reloaded cache: err %v, hit %v", err, err == nil && hit(c))
	}
	c.Invalidate()
	if hit(c) {
		t.Error("hit after Invalidate")
	}
}

func TestCacheThresholdIsConfigurable(t *testing.T) {
	fm := &fakeModel{}
	p := newPipeline(t, fm, cfgNoUnderstand())
	ctx := context.Background()
	para := "In the base Transformer, how many parallel attention heads are used?"
	qv, _ := p.Ret.EmbedQuery(ctx, headsQ)
	pv, _ := p.Ret.EmbedQuery(ctx, para)
	ok := pipeline.Trace{Question: headsQ, Status: "ok", Answer: "8", Citations: []pipeline.Citation{{Tag: "S1"}}}
	for _, c := range []struct {
		th   float32
		want string
	}{{0.80, "hit"}, {0.9999, "miss"}} {
		ca := pipeline.NewCache(c.th)
		ca.Store(qv, "fp", ok)
		if _, sim, v := ca.Lookup(pv, para, "fp"); v != c.want {
			t.Errorf("threshold %.4f: %s (sim %.3f), want %s", c.th, v, sim, c.want)
		}
	}
}
