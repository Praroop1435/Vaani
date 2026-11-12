package pipeline

// Semantic answer cache: AnswerCached wraps Answer, which stays unchanged. Off unless the caller
// uses AnswerCached with a non-nil Cache.
//
// A cached answer is returned only when all three hold:
//  1. same fingerprint: same index files, retrieval setting, prompt and model (else the entry is stale and dropped);
//  2. cosine of the BGE query embeddings >= Threshold;
//  3. same key terms: numbers, names and modifiers ("base" vs "big", "Mimi" vs "Moshi", "1-stage" vs
//     "2-stage") must match exactly. Embeddings score such pairs as near-duplicates, so the
//     embedding alone would hand one question the other's answer.
// Only verified, complete answers are stored (see cacheable).

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
)

// DefaultCacheThreshold: see Docs/CACHE_EVAL.md for the measurement behind it.
const DefaultCacheThreshold float32 = 0.90

type CacheStats struct {
	Lookups  int     `json:"lookups"`
	Hits     int     `json:"hits"`
	Misses   int     `json:"misses"`
	Blocked  int     `json:"blocked"` // a cached question was similar enough but its key terms differed
	Stale    int     `json:"stale"`   // entries dropped because the fingerprint changed
	Stored   int     `json:"stored"`
	Rejected int     `json:"rejected"` // answers not stored: refused, degraded, dropped sentences, no citations
	SavedMs  float64 `json:"saved_ms"` // sum over hits of (original answer time - hit time)
}

type CacheEntry struct {
	Question    string     `json:"question"`
	Vec         []float32  `json:"vec"`
	Terms       []string   `json:"terms"`
	Fingerprint string     `json:"fingerprint"`
	Answer      string     `json:"answer"`
	Citations   []Citation `json:"citations"`
	OrigMs      int64      `json:"orig_ms"`
	Stored      time.Time  `json:"stored"`
}

// cacheVersion: bump when the prompts or the check logic in pipeline.go change, so old answers go stale.
const cacheVersion = "2026-11-11a"

// CacheResult says what the cache did for one request.
type CacheResult struct {
	Verdict     string  `json:"verdict"` // hit, miss, blocked, off
	Sim         float32 `json:"sim,omitempty"`
	Matched     string  `json:"matched,omitempty"` // the cached question that was close enough
	LookupMs    float64 `json:"lookup_ms"`
	Stored      bool    `json:"stored"`
	StoreReason string  `json:"store_reason,omitempty"` // why a miss was not stored
}

// AnswerCached checks the cache before any retrieval or model call; on a miss it runs Answer
// and stores the result only if it passed the check. With c == nil it is exactly Answer.
// shortcut: a miss embeds the question twice (here and in retrieval), ~tens of ms; pass the
// vector into Answer if that shows in the latency tables.
func (p *Pipeline) AnswerCached(ctx context.Context, c *Cache, reqID, question string, emit func(string)) (Trace, CacheResult) {
	if c == nil {
		return p.Answer(ctx, reqID, question, emit), CacheResult{Verdict: "off"}
	}
	start := time.Now()
	qv, err := p.Ret.EmbedQuery(ctx, question)
	if err != nil {
		return p.Answer(ctx, reqID, question, emit), CacheResult{Verdict: "off", StoreReason: "embed: " + short(err)}
	}
	fp := p.Fingerprint()
	e, sim, verdict := c.Lookup(qv, question, fp)
	cr := CacheResult{Verdict: verdict, Sim: sim, Matched: e.Question, LookupMs: msSince(start)}
	if verdict == "hit" {
		if emit != nil {
			emit(e.Answer)
		}
		ms := time.Since(start).Milliseconds()
		c.saved(float64(e.OrigMs - ms))
		return Trace{RequestID: reqID, Question: question, Start: start, FirstTokenMs: ms, TotalMs: ms, Status: "ok",
			Answer: e.Answer, Citations: e.Citations,
			Steps: []Step{{Name: "cache", WorkMs: cr.LookupMs, Note: fmt.Sprintf("hit sim=%.3f cached=%q", sim, e.Question)}}}, cr
	}
	tr := p.Answer(ctx, reqID, question, emit)
	cr.Stored, cr.StoreReason = c.Store(qv, fp, tr)
	return tr, cr
}

// Fingerprint is everything a cached answer depends on: index files, the whole Config (retrieval
// setting, thresholds, token limits), the model and cacheVersion. Any change makes entries stale.
func (p *Pipeline) Fingerprint() string {
	ixHash := ""
	if ix := p.Ret.Indexes[p.Cfg.Setting.ChunkSize]; ix != nil {
		ixHash = ix.Hash
	}
	h := sha256.Sum256(fmt.Appendf(nil, "%+v|%s|%s|%s", p.Cfg, ixHash, p.LLM.Model, cacheVersion))
	return hex.EncodeToString(h[:])
}

// shortcut: brute-force scan, no size cap or eviction; fine for thousands of entries, add LRU if it grows past that.
type Cache struct {
	Threshold float32
	mu        sync.Mutex
	entries   []CacheEntry
	stats     CacheStats
}

func NewCache(threshold float32) *Cache { return &Cache{Threshold: threshold} }

// Lookup returns the best entry passing all three checks. verdict is "hit", "miss", or "blocked"
// (the closest entry over the threshold had different key terms).
func (c *Cache) Lookup(vec []float32, question, fp string) (e CacheEntry, sim float32, verdict string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stats.Lookups++
	n := len(c.entries)
	c.entries = slices.DeleteFunc(c.entries, func(e CacheEntry) bool { return e.Fingerprint != fp })
	c.stats.Stale += n - len(c.entries)
	terms := KeyTerms(question)
	best, verdict := -1, "miss"
	for i, en := range c.entries {
		s := embed.Cosine(vec, en.Vec)
		if s < c.Threshold {
			continue
		}
		if !slices.Equal(terms, en.Terms) {
			if best < 0 {
				verdict, sim = "blocked", s
			}
			continue
		}
		if best < 0 || s > sim {
			best, sim, verdict = i, s, "hit"
		}
	}
	switch verdict {
	case "hit":
		c.stats.Hits++
		return c.entries[best], sim, verdict
	case "blocked":
		c.stats.Blocked++
	}
	c.stats.Misses++
	return CacheEntry{}, sim, verdict
}

// Store keeps tr's answer if it is cacheable; reason says why not.
func (c *Cache) Store(vec []float32, fp string, tr Trace) (ok bool, reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if reason = cacheable(tr); reason != "" {
		c.stats.Rejected++
		return false, reason
	}
	c.entries = append(c.entries, CacheEntry{
		Question: tr.Question, Vec: vec, Terms: KeyTerms(tr.Question), Fingerprint: fp,
		Answer: tr.Answer, Citations: tr.Citations, OrigMs: tr.TotalMs, Stored: time.Now(),
	})
	c.stats.Stored++
	return true, ""
}

// cacheable is the store gate: only answers the pipeline fully verified.
// Completeness comes from the pipeline's part tracking (Trace.Missing).
func cacheable(tr Trace) string {
	switch {
	case tr.Status != "ok":
		return "status " + tr.Status
	case tr.Refused:
		return "refused"
	case len(tr.Missing) > 0:
		return "incomplete: " + strings.Join(tr.Missing, "; ")
	case len(tr.Dropped) > 0:
		return "dropped sentences" // the model said something unsupported, or was cut off mid-sentence
	case len(tr.Citations) == 0:
		return "no citations"
	}
	for _, d := range tr.Degraded {
		if !strings.HasPrefix(d, "understand_") { // a skipped rewrite still gives a verified answer
			return "degraded: " + d
		}
	}
	return ""
}

func (c *Cache) Invalidate() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stats.Stale += len(c.entries)
	c.entries = nil
}

func (c *Cache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}

func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}

func (c *Cache) saved(ms float64) {
	c.mu.Lock()
	c.stats.SavedMs += ms
	c.mu.Unlock()
}

// Save and LoadCache keep entries across runs of cmd/ask. Stale entries are dropped on the next lookup.
func (c *Cache) Save(path string) error {
	c.mu.Lock()
	js, err := json.Marshal(c.entries)
	c.mu.Unlock()
	if err != nil {
		return err
	}
	return os.WriteFile(path, js, 0o644)
}

func LoadCache(path string, threshold float32) (*Cache, error) {
	c := NewCache(threshold)
	js, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	return c, json.Unmarshal(js, &c.entries)
}

var (
	termRe = regexp.MustCompile(`\d+(?:\.\d+)?|[a-z]+`)
	// phrasing words: changing them does not change what is asked
	stopWords = map[string]bool{}
)

func init() {
	for _, w := range strings.Fields(`a an the of in on at to for from by with and or as is are was were be been being
		do does did done have has had having can could would should will shall may might must
		what which who whom whose how when where why many much number amount value values
		use uses used using employ employs employed get gets got give gives given reach reaches reached
		achieve achieves achieved report reports reported run runs operate operates
		it its this that these those there their they them me you your please tell explain describe
		about mean means meaning paper papers model models s`) {
		stopWords[w] = true
	}
}

// relationWords mark a direction: "English-to-German" is not "German-to-English", though both
// have the same words and a cosine of 0.999.
var relationWords = map[string]bool{"to": true, "into": true, "than": true, "vs": true, "versus": true}

// KeyTerms is the sorted set of words that decide what a question asks: everything except
// phrasing words, lightly stemmed, plus one "a>b" term per direction (key term before and after
// a relation word).
func KeyTerms(q string) []string {
	var out []string
	last, pending := "", ""
	for _, w := range termRe.FindAllString(strings.ToLower(q), -1) {
		if relationWords[w] {
			pending = last
			continue
		}
		if stopWords[w] {
			continue
		}
		switch { // crude stemming: "heads" == "head", "trained" == "train", "matching" == "match"
		case len(w) > 5 && strings.HasSuffix(w, "ing"):
			w = w[:len(w)-3]
		case len(w) > 4 && strings.HasSuffix(w, "ed"):
			w = w[:len(w)-2]
		case len(w) > 3 && strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss"):
			w = w[:len(w)-1]
		}
		out = append(out, w)
		if pending != "" {
			out = append(out, pending+">"+w)
			pending = ""
		}
		last = w
	}
	slices.Sort(out)
	return slices.Compact(out)
}
