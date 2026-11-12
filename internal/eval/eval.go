// Package eval holds the question benchmark and scores what retrieval (and later the full
// pipeline) gets right against evidence checked in the PDFs.
package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

// Source is one place the answer is written: a quote on a page, or a table cell.
type Source struct {
	Paper  string `json:"paper"`
	Page   int    `json:"page"`
	Quote  string `json:"quote,omitempty"`
	Table  string `json:"table,omitempty"` // "Table 2"
	Row    string `json:"row,omitempty"`
	Column string `json:"column,omitempty"`
	Value  string `json:"value,omitempty"`
}

// Evidence is one fact the answer needs; any of its sources is enough.
// A multi-passage question has several Evidence items, all required.
type Evidence struct {
	AnyOf []Source `json:"any_of"`
}

type Question struct {
	ID         string     `json:"id"`
	Type       string     `json:"type"` // numeric, table, multi, definition, unanswerable
	Answerable bool       `json:"answerable"`
	Question   string     `json:"question"`
	Answer     string     `json:"answer"`
	Accept     []string   `json:"accept,omitempty"` // regexps a correct generated answer must all match
	Parts      []string   `json:"parts,omitempty"`  // what a multi-part question asks for, one per Evidence item
	Evidence   []Evidence `json:"evidence,omitempty"`
	Absent     []string   `json:"absent,omitempty"` // regexps that must not occur in any paper
	Notes      string     `json:"notes,omitempty"`
}

type Benchmark struct {
	Version   string     `json:"version"`
	Questions []Question `json:"questions"`
}

func Load(path string) (*Benchmark, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var b Benchmark
	if err := json.Unmarshal(data, &b); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &b, nil
}

var ws = regexp.MustCompile(`\s+`)

// Norm collapses whitespace, the only difference between PDF page text and chunk text.
func Norm(s string) string { return strings.TrimSpace(ws.ReplaceAllString(s, " ")) }

// Covers reports whether a retrieved chunk holds this source.
func (s Source) Covers(c ingest.Chunk) bool {
	if c.Paper != s.Paper {
		return false
	}
	if s.Table != "" {
		return c.Kind == "table" && c.Table == s.Table && strings.Contains(c.Text, "| "+s.Row+" |") && strings.Contains(c.Text, s.Value)
	}
	return strings.Contains(NormEvidence(c.Text), NormEvidence(s.Quote))
}

// RetrievalScore is one (question, setting) row of results/retrieval_eval.csv.
type RetrievalScore struct {
	QID, Type, Setting string
	Answerable         bool
	ItemsTotal         int
	ItemsFound         int     // evidence items covered by some hit
	AllFound           bool    // every evidence item covered
	PageOK             bool    // every covering hit's page range includes the evidence page
	FirstRank          int     // rank of the best hit covering the first evidence item, 0 if none
	TopScore           float32 // cosine of the best hit
	ContextTokens      int     // tokens the hits would put in front of the model
	EmbedMs, SearchMs  float64
	HitIDs             []string
}

func ScoreRetrieval(q Question, setting string, res retrieve.Result) RetrievalScore {
	sc := RetrievalScore{QID: q.ID, Type: q.Type, Setting: setting, Answerable: q.Answerable,
		ItemsTotal: len(q.Evidence), EmbedMs: res.EmbedMs, SearchMs: res.SearchMs, PageOK: true}
	if len(res.Hits) > 0 {
		sc.TopScore = res.Hits[0].Score
	}
	for _, h := range res.Hits {
		sc.ContextTokens += h.Chunk.Tokens
		sc.HitIDs = append(sc.HitIDs, h.Chunk.ID)
	}
	for i, ev := range q.Evidence {
		found := false
		for rank, h := range res.Hits {
			for _, src := range ev.AnyOf {
				if !src.Covers(h.Chunk) {
					continue
				}
				if !found {
					found = true
					if i == 0 {
						sc.FirstRank = rank + 1
					}
					if src.Page < h.Chunk.PageStart || src.Page > h.Chunk.PageEnd {
						sc.PageOK = false
					}
				}
			}
		}
		if found {
			sc.ItemsFound++
		}
	}
	sc.AllFound = q.Answerable && sc.ItemsFound == sc.ItemsTotal
	if sc.ItemsFound == 0 {
		sc.PageOK = false
	}
	return sc
}
