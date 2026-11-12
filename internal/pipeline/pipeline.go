// Package pipeline answers one question:
//
//	understand → retrieve → decide → draft → check → (targeted retrieve → draft missing parts → check)
//
// Understand splits the question into the parts it asks for. The draft must answer each part with
// a "[P#]" sentence citing "[S#]" passages, or say "[P#] MISSING". The check step shows a sentence
// only after verifying it (cited passages exist, every number appears in them) and tracks which
// parts are answered. If a part is missing, one targeted retrieval searches for that part alone
// and a second draft answers only the missing parts. Whatever is still missing is said plainly.
//
// The draft streams from the model but the person sees a sentence only once it is verified, so
// time-to-first-token is the first verified sentence: a little slower, never an unsupported claim.
package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Praroop1435/Vaani/internal/governor"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/llm"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

type Config struct {
	Setting           retrieve.Setting
	Deadline          time.Duration // give-up time for the whole request
	Policy            llm.Policy
	FirstTokenTimeout time.Duration // per model call
	Understand        bool          // spend a model call to split the question into parts and a query
	LowBudget         int           // below this many daily requests left: no understand, no targeted round
	OffTopicScore     float32       // best retrieval score under this: refuse without calling the model
	DraftMaxTokens    int
	UnderstandMax     int
	TargetedPerPart   int           // new chunks fetched per missing part
	TargetedMax       int           // new chunks fetched per request
	TargetedMinTime   time.Duration // a targeted round needs at least this much deadline left
}

// Defaults chosen from measurements: 0.50 sits under every answerable benchmark question's best
// score (min 0.626) and over clearly off-library questions (0.38-0.47), see Docs/RETRIEVAL_EVAL.md.
var Defaults = Config{
	Setting: retrieve.R, Deadline: 20 * time.Second, Policy: llm.DefaultPolicy,
	FirstTokenTimeout: 6 * time.Second, Understand: true, LowBudget: 100, OffTopicScore: 0.50,
	DraftMaxTokens: 350, UnderstandMax: 150, // observed max 231 / 101 (smoke runs)
	TargetedPerPart: 2, TargetedMax: 4, TargetedMinTime: 5 * time.Second,
}

type Pipeline struct {
	LLM *llm.Client
	Gov *governor.Governor
	Ret *retrieve.Retriever
	Cfg Config
}

type Step struct {
	Name      string            `json:"name"`
	WaitMs    float64           `json:"wait_ms"`  // queued behind the rate limit
	WorkMs    float64           `json:"work_ms"`  // model or tool time, failed attempts included
	RetryMs   float64           `json:"retry_ms"` // backoff between attempts
	Retries   int               `json:"retries"`
	StartMs   int64             `json:"start_ms"`           // from request start
	FirstMs   int64             `json:"first_ms,omitempty"` // model's first content token, from step start
	TokensIn  int               `json:"tokens_in,omitempty"`
	TokensOut int               `json:"tokens_out,omitempty"`
	Net       *llm.NetTiming    `json:"net,omitempty"`
	Note      string            `json:"note,omitempty"`
	Raw       string            `json:"raw,omitempty"`          // the model's full reply, before the check
	RateHdr   map[string]string `json:"rate_headers,omitempty"` // x-ratelimit-* / retry-after from the last attempt
}

type Citation struct {
	Tag     string `json:"tag"` // S1..Sn: unique per passage within the request
	ChunkID string `json:"chunk_id"`
	Paper   string `json:"paper"`
	Page    string `json:"page"` // "5" or "4-5"
	Section string `json:"section"`
	Label   string `json:"label"` // what the person reads: "Moshi, p.8, Table 1"
}

type Trace struct {
	RequestID    string     `json:"request_id"`
	Question     string     `json:"question"`
	Start        time.Time  `json:"start_time"`
	FirstTokenMs int64      `json:"first_token_ms"` // first verified sentence shown; -1 if none
	TotalMs      int64      `json:"total_ms"`
	Status       string     `json:"status"` // ok, rate_limited, timeout, error
	Refused      bool       `json:"refused"`
	Retries      int        `json:"retries"`
	ModelCalls   int        `json:"model_calls"`
	TokensIn     int        `json:"tokens_in"`
	TokensOut    int        `json:"tokens_out"`
	Parts        []string   `json:"parts"`
	Missing      []string   `json:"missing,omitempty"`  // parts still unanswered at the end
	Targeted     []string   `json:"targeted,omitempty"` // chunk ids added by targeted retrieval
	Answer       string     `json:"answer"`
	Citations    []Citation `json:"citations"`
	Dropped      []string   `json:"dropped,omitempty"`  // sentences the check refused to send
	Degraded     []string   `json:"degraded,omitempty"` // e.g. "understand_skipped_low_budget"
	Steps        []Step     `json:"steps"`
	Err          string     `json:"error,omitempty"`
}

const (
	refusalToken = "NOT IN PAPERS"
	missingToken = "MISSING"
)

var shortName = map[string]string{
	"1706.03762v7": "Attention Is All You Need", "2410.00037v1": "Moshi", "2507.07955v2": "H-Net (Dynamic Chunking)",
}

// run is the state of one request.
type run struct {
	p       *Pipeline
	ctx     context.Context
	tr      *Trace
	emit    func(string)
	hits    []retrieve.Hit      // passages in context, S1..Sn in order
	cites   map[string]Citation // by tag
	used    map[string]bool     // tags cited by a shown sentence
	covered map[int]bool        // part number → answered by a shown sentence
	shown   int
}

func (r *run) say(s string) {
	if r.tr.FirstTokenMs < 0 {
		r.tr.FirstTokenMs = time.Since(r.tr.Start).Milliseconds()
	}
	if r.shown > 0 {
		s = " " + s
	}
	r.shown++
	r.tr.Answer += s
	if r.emit != nil {
		r.emit(s)
	}
}

// Answer runs the whole pipeline. emit receives each verified piece of the reply as soon as it
// passes the check (nil is fine).
func (p *Pipeline) Answer(ctx context.Context, reqID, question string, emit func(string)) (tr Trace) {
	tr = Trace{RequestID: reqID, Question: question, Start: time.Now(), FirstTokenMs: -1, Status: "ok"}
	ctx, cancel := context.WithTimeout(ctx, p.Cfg.Deadline)
	defer cancel()
	r := &run{p: p, ctx: ctx, tr: &tr, emit: emit, cites: map[string]Citation{}, used: map[string]bool{}, covered: map[int]bool{}}
	defer r.finish()

	// 2a. Retrieve with the raw question first (~20 ms): a clearly off-library question is then
	// refused before any model call. Running it alongside understand would save those ~20 ms but
	// spend a Groq request on every off-topic question.
	t0 := time.Now()
	first := p.Cfg.Setting
	first.Rerank = false // the re-rank (~2 s) runs once, in 2b, after the off-topic check
	res, err := p.Ret.Retrieve(ctx, question, first)
	if err != nil {
		tr.Status, tr.Err = "error", err.Error()
		return tr
	}
	hits := res.Hits
	tr.Steps = append(tr.Steps, Step{Name: "retrieve", WorkMs: msSince(t0), Note: "query=question"})

	// 3. Decide: clearly off-library questions are refused without spending a model call.
	if top(hits) < p.Cfg.OffTopicScore {
		tr.Refused = true
		tr.Steps = append(tr.Steps, Step{Name: "decide", Note: fmt.Sprintf("refuse: best score %.3f < %.2f", top(hits), p.Cfg.OffTopicScore)})
		r.say("The three papers in this library (Attention Is All You Need, Moshi, H-Net) do not cover this question.")
		return tr
	}

	// 1. Understand: split the question into parts and a search query.
	tr.Parts = []string{question}
	query := question
	switch {
	case !p.Cfg.Understand:
	case p.Gov.DailyLeft() < p.Cfg.LowBudget:
		tr.Degraded = append(tr.Degraded, "understand_skipped_low_budget")
	default:
		q, parts, st, err := p.understand(ctx, &tr, question)
		tr.Steps = append(tr.Steps, st)
		if err != nil {
			tr.Degraded = append(tr.Degraded, "understand_failed: "+short(err))
		} else {
			query, tr.Parts = q, parts
		}
	}

	// 2b. Retrieve again with the rewrite and keep the best of both.
	if p.Cfg.Setting.Rerank {
		// Setting R: the question and the rewrite share one candidate pool and one re-rank.
		t0 := time.Now()
		texts := []string{question}
		if query != question {
			texts = append(texts, query)
		}
		if res2, err := p.Ret.RetrieveQueries(ctx, question, question, texts, p.Cfg.Setting); err == nil && len(res2.Hits) > 0 {
			hits = res2.Hits
		} else if err != nil {
			tr.Degraded = append(tr.Degraded, "rerank_failed: "+short(err))
		}
		tr.Steps = append(tr.Steps, Step{Name: "retrieve_rerank", StartMs: time.Since(tr.Start).Milliseconds(), WorkMs: msSince(t0), Note: fmt.Sprintf("query=%q", query)})
	} else if query != question {
		t0 := time.Now()
		if res2, err := p.Ret.Retrieve(ctx, query, p.Cfg.Setting); err == nil {
			hits = mergeHits(hits, res2.Hits, p.Cfg.Setting.K)
		}
		tr.Steps = append(tr.Steps, Step{Name: "retrieve_rewrite", StartMs: time.Since(tr.Start).Milliseconds(), WorkMs: msSince(t0), Note: fmt.Sprintf("query=%q", query)})
	}
	tr.Steps = append(tr.Steps, Step{Name: "decide", Note: fmt.Sprintf("draft: best score %.3f, %d part(s)", top(hits), len(tr.Parts))})
	r.addPassages(hits)

	// 4-5. Draft every part, check every sentence.
	all := make([]int, len(tr.Parts))
	for i := range all {
		all[i] = i + 1
	}
	if err := r.draft("draft", question, all, 0); err != nil {
		r.onError(err)
		return tr
	}

	// Loop: one targeted retrieval for the parts still missing, then draft only those.
	if missing := r.missing(); len(missing) > 0 {
		dl, _ := ctx.Deadline()
		switch {
		case time.Until(dl) < p.Cfg.TargetedMinTime:
			tr.Degraded = append(tr.Degraded, "targeted_skipped_deadline")
		case p.Gov.DailyLeft() < p.Cfg.LowBudget:
			tr.Degraded = append(tr.Degraded, "targeted_skipped_low_budget")
		default:
			before := len(r.hits)
			if added := r.targeted(missing); added > 0 {
				// only the new passages: the old ones did not state these parts, and resending them
				// doubled the prompt and hit the 8K tokens/minute limit (smoke test 2, q14)
				shownBefore := r.shown
				if err := r.draft("draft_targeted", question, missing, before); err != nil {
					if r.shown == shownBefore && unaffordable(err) {
						// the targeted round is an extra; if the budget can't fit it in time, finish
						// with what round 1 established instead of failing (live smoke test 3, q19)
						tr.Degraded = append(tr.Degraded, "targeted_skipped: "+short(err))
					} else {
						r.onError(err)
						return tr
					}
				}
			}
		}
	}

	// Say plainly what is still not answered.
	missing := r.missing()
	if len(missing) == len(tr.Parts) {
		tr.Refused = true
		r.say("The papers do not say. The closest passages are " + r.listCites() + ".")
		return tr
	}
	for _, k := range missing {
		tr.Missing = append(tr.Missing, tr.Parts[k-1])
		r.say("I could not find " + tr.Parts[k-1] + " in the papers.")
	}
	return tr
}

// understand asks for {"query": ..., "parts": [...]} as JSON.
func (p *Pipeline) understand(ctx context.Context, tr *Trace, question string) (string, []string, Step, error) {
	msgs := []llm.Message{
		{Role: "system", Content: "You prepare questions for search over three papers: " +
			"'Attention Is All You Need' (the Transformer), 'Moshi' (speech-text dialogue model; its audio codec is Mimi, its text LLM is Helium), " +
			"and 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling' (H-Net).\n" +
			"Reply with JSON only: {\"query\": \"<one short search query>\", \"parts\": [\"<each distinct thing the question asks for>\"]}.\n" +
			"Each part is a short self-contained phrase that names its subject, e.g. \"number of tokens Helium was pretrained on\". " +
			"A question asking one thing has one part; at most 4 parts. Expand abbreviations and name the model or component."},
		{Role: "user", Content: question},
	}
	res, st, err := p.call(ctx, tr, "understand", msgs, p.Cfg.UnderstandMax, nil)
	if err != nil {
		return "", nil, st, err
	}
	var out struct {
		Query string   `json:"query"`
		Parts []string `json:"parts"`
	}
	text := res.Text
	if i, j := strings.Index(text, "{"), strings.LastIndex(text, "}"); i >= 0 && j > i {
		text = text[i : j+1]
	}
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		return "", nil, st, fmt.Errorf("understand: not JSON: %w", err)
	}
	var parts []string
	for _, pt := range out.Parts {
		if pt = strings.TrimSpace(pt); pt != "" && len(parts) < 4 {
			parts = append(parts, pt)
		}
	}
	if len(parts) == 0 {
		parts = []string{question}
	}
	q := strings.TrimSpace(out.Query)
	if q == "" {
		q = question
	}
	return q, parts, st, nil
}

func mergeHits(a, b []retrieve.Hit, k int) []retrieve.Hit {
	best := map[string]retrieve.Hit{}
	for _, h := range append(a, b...) {
		if cur, ok := best[h.Chunk.ID]; !ok || h.Score > cur.Score {
			best[h.Chunk.ID] = h
		}
	}
	var merged []retrieve.Hit
	for _, h := range best {
		merged = append(merged, h)
	}
	slices.SortFunc(merged, func(x, y retrieve.Hit) int {
		if x.Score != y.Score {
			if x.Score > y.Score {
				return -1
			}
			return 1
		}
		return strings.Compare(x.Chunk.ID, y.Chunk.ID)
	})
	return merged[:min(k, len(merged))]
}

// addPassages appends hits as new tags S(n+1)...
func (r *run) addPassages(hits []retrieve.Hit) {
	for _, h := range hits {
		tag := fmt.Sprintf("S%d", len(r.hits)+1)
		r.hits = append(r.hits, h)
		r.cites[tag] = NewCitation(tag, h.Chunk)
	}
}

// NewCitation gives a passage its unique internal tag and the label the person reads. Two
// passages on the same page get distinct labels through their section or table:
// "Attention Is All You Need, p.8, §6.1" vs "Attention Is All You Need, p.8, Table 2".
func NewCitation(tag string, c ingest.Chunk) Citation {
	page := strconv.Itoa(c.PageStart)
	if c.PageEnd != c.PageStart {
		page = fmt.Sprintf("%d-%d", c.PageStart, c.PageEnd)
	}
	where := "§" + c.SectionNumber
	if c.Table != "" {
		where = c.Table
	} else if c.SectionNumber == "" {
		where = c.Section
	}
	return Citation{Tag: tag, ChunkID: c.ID, Paper: c.Paper, Page: page, Section: strings.TrimSpace(c.SectionNumber + " " + c.Section),
		Label: fmt.Sprintf("%s, p.%s, %s", shortName[c.Paper], page, where)}
}

// targeted searches for each missing part on its own and adds chunks not already in context.
func (r *run) targeted(missing []int) int {
	t0 := time.Now()
	in := map[string]bool{}
	for _, h := range r.hits {
		in[h.Chunk.ID] = true
	}
	s := r.p.Cfg.Setting
	s.K += r.p.Cfg.TargetedPerPart // look past the chunks already in context
	var add []retrieve.Hit
	for _, k := range missing {
		part := r.tr.Parts[k-1]
		var res retrieve.Result
		var err error
		if s.Rerank { // a part rarely names its paper: scope by the whole question too
			res, err = r.p.Ret.RetrieveQueries(r.ctx, part, r.tr.Question+" "+part, []string{part}, s)
		} else {
			res, err = r.p.Ret.Retrieve(r.ctx, part, s)
		}
		if err != nil {
			continue
		}
		n := 0
		for _, h := range res.Hits {
			if !in[h.Chunk.ID] && h.Score >= r.p.Cfg.OffTopicScore && n < r.p.Cfg.TargetedPerPart && len(add) < r.p.Cfg.TargetedMax {
				in[h.Chunk.ID] = true
				add = append(add, h)
				r.tr.Targeted = append(r.tr.Targeted, h.Chunk.ID)
				n++
			}
		}
	}
	var parts []string
	for _, k := range missing {
		parts = append(parts, fmt.Sprintf("P%d", k))
	}
	r.tr.Steps = append(r.tr.Steps, Step{Name: "retrieve_targeted", StartMs: time.Since(r.tr.Start).Milliseconds(), WorkMs: msSince(t0),
		Note: fmt.Sprintf("missing %s, added %d chunk(s)", strings.Join(parts, ","), len(add))})
	r.addPassages(add)
	return len(add)
}

func (r *run) missing() []int {
	var m []int
	for k := 1; k <= len(r.tr.Parts); k++ {
		if !r.covered[k] {
			m = append(m, k)
		}
	}
	return m
}

var (
	tagGroup = regexp.MustCompile(`(?:\s*\[S\d+\])+[.!?]?`)
	tagRe    = regexp.MustCompile(`\[(S\d+)\]`)
	// "[P1]" (asked for), "P1:" and "**P1**" (what gpt-oss also writes); 1-2 digits so "P100 GPUs" is text
	partRe    = regexp.MustCompile(`\[P(\d{1,2})\]|\*\*P(\d{1,2})\*\*:?|\bP(\d{1,2})\s*:`)
	missingRe = regexp.MustCompile(`\bMISSING\b`)
	wordRe    = regexp.MustCompile(`\pL{2,}|\d`)
	numRe     = regexp.MustCompile(`\d+(?:[.,]\d+)*`)
)

// tagMayContinue reports whether the text after a tag group could still extend it: nothing yet,
// or the start of another [S#] tag. Groq streams small pieces, so "[S1][S2]" can arrive as
// "[S1][" + "S2]" (live smoke test 3, q05); ending the unit at "[S1]" lost the [S2] citation.
func tagMayContinue(rest string) bool {
	r := strings.TrimLeft(rest, " \t\n")
	return r == "" || r[0] == '[' && (len(r) < 2 || r[1] == 'S')
}

func partNum(u string) int {
	m := partRe.FindStringSubmatch(u)
	if m == nil {
		return 0
	}
	n, _ := strconv.Atoi(m[1] + m[2] + m[3]) // only one group is set
	return n
}

// draft streams a reply for the given parts and checks it one unit at a time. A unit ends at a
// group of [S#] tags, or where the next part tag starts (a "[P2] MISSING" unit has no [S#]).
// The prompt holds the passages from index `from` on.
func (r *run) draft(name, question string, parts []int, from int) error {
	var b strings.Builder
	for i, h := range r.hits {
		if i < from {
			continue
		}
		c := r.cites[fmt.Sprintf("S%d", i+1)]
		table := ""
		if h.Chunk.Table != "" {
			table = ", " + h.Chunk.Table
		}
		fmt.Fprintf(&b, "[S%d] (%s, page %s, section %s%s)\n%s\n\n", i+1, shortName[c.Paper], c.Page, c.Section, table, h.Chunk.Text)
	}
	b.WriteString("Question: " + question + "\nParts to answer:\n")
	for _, k := range parts {
		fmt.Fprintf(&b, "[P%d] %s\n", k, r.tr.Parts[k-1]) // listed the way answers must be tagged: the model copies it
	}
	if from > 0 {
		b.WriteString("These passages were found for the parts above. Answer only these parts.\n")
	}
	msgs := []llm.Message{
		{Role: "system", Content: "You answer engineers' questions using ONLY the numbered passages from three research papers.\n" +
			"Rules:\n1. Use only facts stated in the passages. No outside knowledge.\n" +
			"2. Answer every listed part. Start each sentence with its part tag in square brackets and end it with the tags of the passages it relies on, e.g. \"[P1] Helium has 7B parameters [S2].\"\n" +
			"3. Give exactly the value each part asks for. If no passage states it, write \"[P2] " + missingToken + "\" instead. Never substitute a related figure (for example a batch size or a per-step number when a total is asked for).\n" +
			"4. If the passages do not address the question at all, reply with exactly: " + refusalToken + "\n" +
			"5. At most 2 sentences per part. No preamble, no headings."},
		{Role: "user", Content: b.String()},
	}
	var buf string
	single := len(r.tr.Parts) == 1
	check := func(unit string) {
		u := strings.TrimSpace(unit)
		if u == "" || strings.Contains(u, refusalToken) {
			return // a refusal covers nothing; missing parts are handled after the draft
		}
		k := partNum(u)
		if k == 0 && single {
			k = 1
		}
		body := strings.TrimSpace(partRe.ReplaceAllString(u, ""))
		if body == "" || missingRe.MatchString(body) {
			return // "[P2] MISSING", "P1: [S1] MISSING", or a bare tag: covers nothing, shows nothing
		}
		if !wordRe.MatchString(tagRe.ReplaceAllString(body, "")) { // a claim needs a word or a number
			r.tr.Dropped = append(r.tr.Dropped, "no content: "+u)
			return
		}
		if k != 0 && !slices.Contains(parts, k) {
			r.tr.Dropped = append(r.tr.Dropped, fmt.Sprintf("answers P%d, which was not asked in this round: %s", k, u))
			return
		}
		if ok, why := r.verify(body); ok {
			r.say(r.render(body))
			if k > 0 {
				r.covered[k] = true
			}
		} else {
			r.tr.Dropped = append(r.tr.Dropped, why+": "+u)
		}
	}
	_, st, err := r.p.call(r.ctx, r.tr, name, msgs, r.p.Cfg.DraftMaxTokens, func(s string) {
		buf += s
		for {
			end := -1
			if loc := tagGroup.FindStringIndex(buf); loc != nil && !tagMayContinue(buf[loc[1]:]) {
				end = loc[1]
			}
			if loc := partRe.FindAllStringIndex(buf, 2); len(loc) == 2 && (end < 0 || loc[1][0] < end) {
				end = loc[1][0] // the next part starts: whatever came before is a unit
			}
			if end < 0 {
				return
			}
			check(buf[:end])
			buf = buf[end:]
		}
	})
	r.tr.Steps = append(r.tr.Steps, st)
	if err == nil {
		check(buf)
	}
	return err
}

// verify is the check step: tags must name passages in context, every number must appear in one.
func (r *run) verify(unit string) (bool, string) {
	tags := tagRe.FindAllStringSubmatch(unit, -1)
	if len(tags) == 0 {
		return false, "no citation"
	}
	var cited []string
	for _, t := range tags {
		c, ok := r.cites[t[1]]
		if !ok {
			return false, "cites missing passage " + t[1]
		}
		for _, h := range r.hits {
			if h.Chunk.ID == c.ChunkID { // the passage header (section, page) may be quoted too
				cited = append(cited, strings.ReplaceAll(c.Section+" page "+c.Page+" "+h.Chunk.Text, ",", ""))
			}
		}
	}
	body := tagRe.ReplaceAllString(unit, "")
	for _, n := range numRe.FindAllString(body, -1) {
		n = strings.TrimRight(strings.ReplaceAll(n, ",", ""), ".")
		re := regexp.MustCompile(`(^|[^\d.])` + regexp.QuoteMeta(n) + `([^\d]|$)`)
		if !slices.ContainsFunc(cited, re.MatchString) {
			return false, "number " + n + " not in cited passages"
		}
	}
	for _, t := range tags {
		r.used[t[1]] = true
	}
	return true, ""
}

func (r *run) render(unit string) string { return Render(unit, r.cites) }

// Render turns each group of [S#] tags into one readable bracket, each label once.
func Render(unit string, cites map[string]Citation) string {
	return strings.TrimSpace(tagGroup.ReplaceAllStringFunc(unit, func(g string) string {
		var labels []string
		for _, t := range tagRe.FindAllStringSubmatch(g, -1) {
			if l := cites[t[1]].Label; !slices.Contains(labels, l) {
				labels = append(labels, l)
			}
		}
		punct := ""
		if last := g[len(g)-1]; last == '.' || last == '!' || last == '?' {
			punct = string(last)
		}
		return " [" + strings.Join(labels, "; ") + "]" + punct
	}))
}

func (r *run) listCites() string {
	var out []string
	for i := range r.hits {
		out = append(out, r.cites[fmt.Sprintf("S%d", i+1)].Label)
	}
	return strings.Join(out, "; ")
}

func (r *run) finish() {
	tr := r.tr
	tr.TotalMs = time.Since(tr.Start).Milliseconds()
	tr.Citations = nil
	for i := range r.hits {
		if tag := fmt.Sprintf("S%d", i+1); r.used[tag] {
			tr.Citations = append(tr.Citations, r.cites[tag])
		}
	}
	tr.Retries, tr.TokensIn, tr.TokensOut, tr.ModelCalls = 0, 0, 0, 0
	for _, s := range tr.Steps {
		tr.Retries += s.Retries
		tr.TokensIn += s.TokensIn
		tr.TokensOut += s.TokensOut
		if s.Name == "understand" || strings.HasPrefix(s.Name, "draft") {
			tr.ModelCalls++
		}
	}
}

// call reserves budget, runs the model call under the retry policy, and records the step.
func (p *Pipeline) call(ctx context.Context, tr *Trace, name string, msgs []llm.Message, maxTok int, onTok func(string)) (llm.Result, Step, error) {
	st := Step{Name: name, StartMs: time.Since(tr.Start).Milliseconds()}
	chars := 0
	for _, m := range msgs {
		chars += len(m.Content)
	}
	est := chars/4 + maxTok // Groq charges prompt + max_completion_tokens against TPM at request time (measured)
	t0 := time.Now()
	waited, err := p.Gov.Reserve(ctx, est)
	st.WaitMs = float64(waited.Microseconds()) / 1000
	if err != nil {
		st.Note = "reserve: " + err.Error()
		st.WaitMs = msSince(t0)
		return llm.Result{}, st, err
	}
	var first int64 = -1
	start := time.Now()
	res, stats, err := llm.Do(ctx, p.Cfg.Policy, func(actx context.Context) (llm.Result, error) {
		r, err := p.LLM.ChatStream(actx, msgs, llm.Options{MaxTokens: maxTok, FirstTokenTimeout: p.Cfg.FirstTokenTimeout, OnToken: func(s string) {
			if first < 0 {
				first = time.Since(start).Milliseconds()
			}
			if onTok != nil {
				onTok(s)
			}
		}})
		p.Gov.Observe(r.Header)
		return r, err
	})
	st.WorkMs, st.RetryMs, st.Retries = stats.WorkMs, stats.RetryWaitMs, stats.Retries()
	st.FirstMs, st.TokensIn, st.TokensOut = first, res.TokensIn, res.TokensOut
	st.Raw = res.Text
	if res.Header != nil {
		net := res.Net
		st.Net = &net
	}
	for k, v := range res.Header {
		if k = strings.ToLower(k); strings.HasPrefix(k, "x-ratelimit-") || k == "retry-after" {
			if st.RateHdr == nil {
				st.RateHdr = map[string]string{}
			}
			st.RateHdr[k] = strings.Join(v, ",")
		}
	}
	if err != nil {
		st.Note = short(err)
	}
	return res, st, err
}

// onError: if part of the answer already reached the person, keep it and say it was cut short;
// otherwise answer without the model. Never a guessed answer.
func (r *run) onError(err error) {
	tr := r.tr
	tr.Err = err.Error()
	status, why := "error", "the model service failed"
	switch {
	case errors.Is(err, governor.ErrDailyBudget) || llm.IsDaily(err):
		status = "rate_limited"
		d := r.p.Gov.Daily()
		next := "soon"
		if d.NextIn > 0 {
			next = "in about " + d.NextIn.Round(time.Second).String()
		}
		full := "within 24 hours"
		if !d.FullAt.IsZero() {
			full = clock(d.FullAt, time.Now())
		}
		why = fmt.Sprintf("today's model budget (%d requests/day) is used up. It refills gradually, about one request every %s: the next request can run %s, and the full daily quota is back %s",
			d.Limit, d.RefillEach.Round(time.Second), next, full)
	case errors.Is(err, governor.ErrDeadlineWait):
		status = "rate_limited"
		why = "the assistant is at its per-minute limit and could not answer within " + r.p.Cfg.Deadline.String() + "; please ask again in a minute"
	case errors.Is(err, context.DeadlineExceeded) || errors.Is(err, llm.ErrFirstTokenTimeout):
		status = "timeout"
		why = "the model did not answer in time"
	}
	tr.Status = status
	if r.shown > 0 {
		tr.Degraded = append(tr.Degraded, "answer_cut_short")
		r.say("(The rest of this answer was cut short: " + why + ".)")
		return
	}
	tr.Degraded = append(tr.Degraded, "passages_only")
	r.say("I can't write an answer right now: " + why + ". The passages most likely to answer this are: " + r.listCites() + ".")
}

// clock says when t is, as a person reads it: "at 15:04 today", "tomorrow at 15:04", "on Fri 13 Nov at 15:04".
// (A bare "by 22:25" read as today when the full quota was 24 h away: load run daily_exhausted.)
func clock(t, now time.Time) string {
	t, now = t.Local(), now.Local()
	day := func(x time.Time) string { return x.Format("2006-01-02") }
	switch day(t) {
	case day(now):
		return "at " + t.Format("15:04") + " today"
	case day(now.AddDate(0, 0, 1)):
		return "tomorrow at " + t.Format("15:04")
	}
	return "on " + t.Format("Mon 2 Jan at 15:04")
}

// unaffordable: the call never started because a limit could not make room before the deadline.
func unaffordable(err error) bool {
	return errors.Is(err, governor.ErrDeadlineWait) || errors.Is(err, governor.ErrDailyBudget) ||
		errors.Is(err, governor.ErrTooLargeAlone) || llm.IsDaily(err)
}

// top is the best cosine among the hits (the first hit for A; fused settings order differently).
func top(h []retrieve.Hit) float32 {
	var best float32
	for _, x := range h {
		best = max(best, x.Score)
	}
	return best
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000 }

func short(err error) string {
	s := err.Error()
	if len(s) > 120 {
		s = s[:120]
	}
	return s
}
