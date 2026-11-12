package test

import (
	"encoding/json"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/ingest"
)

// Verifies the benchmark against the papers themselves, not against our chunks or retrieval:
// every quote must be on its stated PDF page, every table cell must match the parsed table and
// appear on the PDF page, and every "absent" pattern must occur in none of the three papers.
func TestBenchmarkIsVerifiedAgainstThePDFs(t *testing.T) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		t.Skip("pdftotext not installed")
	}
	b, err := eval.Load("../data/eval/benchmark.json")
	if err != nil {
		t.Fatal(err)
	}

	// shape: the brief asks for 15+ questions incl. some the papers do not answer
	types := map[string]int{}
	ids := map[string]bool{}
	for _, q := range b.Questions {
		types[q.Type]++
		if ids[q.ID] {
			t.Errorf("duplicate id %s", q.ID)
		}
		ids[q.ID] = true
		if q.Answerable == (q.Type == "unanswerable") {
			t.Errorf("%s: type %q but answerable=%v", q.ID, q.Type, q.Answerable)
		}
		if q.Answerable && (len(q.Evidence) == 0 || len(q.Accept) == 0) {
			t.Errorf("%s: answerable question needs evidence and accept patterns", q.ID)
		}
		if len(q.Parts) > 0 && len(q.Parts) != len(q.Evidence) {
			t.Errorf("%s: %d parts but %d evidence items; they must line up", q.ID, len(q.Parts), len(q.Evidence))
		}
		for _, a := range q.Accept {
			if !regexp.MustCompile(a).MatchString(eval.NormText(q.Answer)) {
				t.Errorf("%s: accept %q does not match its own normalised reference answer", q.ID, a)
			}
		}
		// every source that answers a single-part question must satisfy all its accept patterns,
		// otherwise a correct answer quoting that source is graded wrong
		if q.Answerable && len(q.Evidence) == 1 {
			for _, s := range q.Evidence[0].AnyOf {
				stmt := s.Quote
				if s.Table != "" {
					stmt = s.Row + " " + s.Column + ": " + s.Value
				}
				for _, a := range q.Accept {
					if !regexp.MustCompile(a).MatchString(eval.NormText(stmt)) {
						t.Errorf("%s: source %q does not satisfy accept %q", q.ID, stmt, a)
					}
				}
			}
		}
		if !q.Answerable && len(q.Absent) == 0 {
			t.Errorf("%s: unanswerable question needs absent patterns", q.ID)
		}
	}
	if len(b.Questions) < 15 || types["unanswerable"] < 3 || types["multi"] < 2 || types["table"] < 3 || types["numeric"] < 3 {
		t.Errorf("benchmark too thin: %d questions, types %v", len(b.Questions), types)
	}
	t.Logf("%d questions: %v", len(b.Questions), types)

	pages := map[string][]string{}
	for _, p := range ingest.Papers {
		raw, _, err := ingest.Extract("../data/papers/" + p.ID + ".pdf")
		if err != nil {
			t.Skip(err)
		}
		for _, pg := range raw {
			pages[p.ID] = append(pages[p.ID], eval.Norm(pg))
		}
	}
	var tables []ingest.Table
	js, err := os.ReadFile("../data/index/tables.json")
	if err != nil {
		t.Skip("tables.json missing: run cmd/ingest")
	}
	if err := json.Unmarshal(js, &tables); err != nil {
		t.Fatal(err)
	}
	noSpace := func(s string) string { return strings.ReplaceAll(s, " ", "") }

	for _, q := range b.Questions {
		for _, ev := range q.Evidence {
			for _, s := range ev.AnyOf {
				pp := pages[s.Paper]
				if s.Page < 1 || s.Page > len(pp) {
					t.Errorf("%s: %s has no page %d", q.ID, s.Paper, s.Page)
					continue
				}
				if s.Quote != "" {
					if !strings.Contains(pp[s.Page-1], eval.Norm(s.Quote)) {
						var on []int
						for i, pg := range pp {
							if strings.Contains(pg, eval.Norm(s.Quote)) {
								on = append(on, i+1)
							}
						}
						t.Errorf("%s: quote %q not on page %d of %s (found on pages %v)", q.ID, s.Quote, s.Page, s.Paper, on)
					}
					continue
				}
				i := slices.IndexFunc(tables, func(x ingest.Table) bool { return x.Paper == s.Paper && "Table "+x.Number == s.Table })
				if i < 0 {
					t.Errorf("%s: %s %s not parsed", q.ID, s.Paper, s.Table)
					continue
				}
				tb := tables[i]
				if tb.Page != s.Page {
					t.Errorf("%s: %s is on page %d, benchmark says %d", q.ID, s.Table, tb.Page, s.Page)
				}
				got := ""
				for _, g := range tb.Parts {
					col := slices.Index(g.Header, s.Column)
					for _, r := range g.Rows {
						if col >= 0 && r[0] == s.Row && got == "" {
							got = r[col]
						}
					}
				}
				if got != s.Value {
					t.Errorf("%s: %s [%s][%s] = %q in tables.json, benchmark says %q", q.ID, s.Table, s.Row, s.Column, got, s.Value)
				}
				if !strings.Contains(noSpace(pp[s.Page-1]), noSpace(s.Value)) {
					t.Errorf("%s: value %q not printed on PDF page %d", q.ID, s.Value, s.Page)
				}
			}
		}
		for _, a := range q.Absent {
			re := regexp.MustCompile("(?i)" + a)
			for paper, pp := range pages {
				for i, pg := range pp {
					if loc := re.FindStringIndex(pg); loc != nil {
						t.Errorf("%s: 'absent' pattern %q found in %s page %d: %q", q.ID, a, paper, i+1, pg[max(0, loc[0]-60):min(len(pg), loc[1]+60)])
					}
				}
			}
		}
	}
}
