package test

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/Praroop1435/Vaani/internal/ingest"
)

// Unit tests use synthetic pages; the last test runs the real PDFs through pdftotext (no model needed).

func wordCount(texts []string) []int { // stand-in for the BGE tokenizer: words + [CLS] + [SEP]
	n := make([]int, len(texts))
	for i, t := range texts {
		n[i] = len(strings.Fields(t)) + 2
	}
	return n
}

func find(segs []ingest.Segment, title string) *ingest.Segment {
	for i := range segs {
		if segs[i].Title == title {
			return &segs[i]
		}
	}
	return nil
}

func text(s *ingest.Segment) string {
	var b []string
	for _, x := range s.Sentences {
		b = append(b, x.Text)
	}
	return strings.Join(b, " ")
}

func TestParseOutlineNesting(t *testing.T) {
	xml := `<pdf2xml><page number="1"></page><outline><item page="2">Model</item><outline>
<item page="3">Attention</item><outline><item page="4">Multi-Head Attention</item></outline>
</outline><item page="9">Conclusion</item></outline></pdf2xml>`
	heads, err := ingest.ParseOutline([]byte(xml))
	if err != nil {
		t.Fatal(err)
	}
	want := []ingest.Heading{{Title: "Model", Page: 2, Depth: 1}, {Title: "Attention", Page: 3, Depth: 2}, {Title: "Multi-Head Attention", Page: 4, Depth: 3}, {Title: "Conclusion", Page: 9, Depth: 1}}
	if len(heads) != len(want) {
		t.Fatalf("got %v", heads)
	}
	for i := range want {
		if heads[i] != want[i] {
			t.Errorf("heading %d = %+v, want %+v", i, heads[i], want[i])
		}
	}
}

// Figure 2 in the Transformer paper prints both sub-section titles as image labels above the real headings.
func TestSegmentsSkipFigureLabelsThatRepeatHeadings(t *testing.T) {
	page := strings.Join([]string{
		"Scaled Dot-Product Attention", "Multi-Head Attention", "Figure 2: (left) Scaled Dot-Product Attention.",
		"3.2.1", "", "Scaled Dot-Product Attention", "", "We call our particular attention Scaled Dot-Product Attention.",
		"3.2.2", "", "Multi-Head Attention", "", "Instead of performing a single attention function we project.",
		"3.4", "Embeddings and Softmax", "Similarly to other models we use learned embeddings.",
		"Positional Encoding", "Since our model contains no recurrence we add positional encodings.",
		"References", "[1] Some cited paper.",
	}, "\n")
	heads := []ingest.Heading{
		{Title: "Scaled Dot-Product Attention", Page: 1, Depth: 1}, {Title: "Multi-Head Attention", Page: 1, Depth: 1},
		{Title: "Embeddings and Softmax", Page: 1, Depth: 1}, {Title: "Positional Encoding", Page: 1, Depth: 1},
	}
	segs, warns := ingest.Segments([]string{page}, heads, []string{"References"})
	if len(warns) > 0 {
		t.Fatalf("warnings: %v", warns)
	}
	sdp := find(segs, "Scaled Dot-Product Attention")
	if sdp == nil || sdp.Number != "3.2.1" || !strings.HasPrefix(text(sdp), "We call our particular attention") {
		t.Errorf("3.2.1 attached to the figure label: %+v", sdp)
	}
	if mha := find(segs, "Multi-Head Attention"); mha == nil || mha.Number != "3.2.2" || !strings.HasPrefix(text(mha), "Instead of performing") {
		t.Errorf("3.2.2 wrong: %+v", mha)
	}
	if pe := find(segs, "Positional Encoding"); pe == nil || pe.Number != "3.5" {
		t.Errorf("unnumbered heading after 3.4 should infer 3.5: %+v", pe)
	}
	if title := find(segs, "Title page"); title == nil || !strings.Contains(text(title), "Figure 2") {
		t.Errorf("text before the first real heading belongs to the title page: %+v", title)
	}

	chunks := ingest.Pack("p", segs, 1000, 512, wordCount)
	for _, c := range chunks {
		if c.Section == "References" {
			t.Error("References section was indexed")
		}
	}
}

func TestPackRespectsLimitsSectionsAndPages(t *testing.T) {
	long := strings.Repeat("The encoder maps an input sequence to continuous representations. ", 60) // 60 sentences x 9 words
	segs := []ingest.Segment{
		{Number: "1", Title: "Intro", Path: "Intro", Sentences: []ingest.Sentence{{Text: "Short intro.", Page: 1}}},
		{Number: "2", Title: "Body", Path: "Body"},
	}
	for i, s := range strings.SplitAfter(strings.TrimSpace(long), ". ") {
		segs[1].Sentences = append(segs[1].Sentences, ingest.Sentence{Text: strings.TrimSpace(s), Page: 2 + i/30})
	}
	segs[1].Sentences = append(segs[1].Sentences, ingest.Sentence{Text: strings.Repeat("word ", 700), Page: 4}) // one huge "sentence"

	chunks := ingest.Pack("p", segs, 100, 120, wordCount)
	if chunks[0].Section != "Intro" || chunks[0].Text != "Short intro." {
		t.Errorf("first chunk should be the intro alone: %+v", chunks[0])
	}
	spans := false
	for i, c := range chunks {
		if c.Tokens > 120 {
			t.Errorf("chunk %s has %d tokens, max 120", c.ID, c.Tokens)
		}
		if c.Order != i || c.ID == "" {
			t.Errorf("chunk %d has order %d id %q", i, c.Order, c.ID)
		}
		if c.PageEnd < c.PageStart {
			t.Errorf("chunk %s page range %d-%d", c.ID, c.PageStart, c.PageEnd)
		}
		spans = spans || c.PageEnd > c.PageStart
	}
	if !spans {
		t.Error("expected a chunk running across a page break to record both pages")
	}
}

func TestRealPapersSectionAttribution(t *testing.T) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		t.Skip("pdftotext not installed (brew install poppler)")
	}
	type want struct{ number, mustContain string }
	expect := map[string]map[string]want{
		"1706.03762v7": {
			"Scaled Dot-Product Attention": {"3.2.1", "We call our particular attention"},
			"Multi-Head Attention":         {"3.2.2", "Instead of performing a single attention function"},
			"Positional Encoding":          {"3.5", "Since our model contains no recurrence"},
			"Abstract":                     {"", "The dominant sequence transduction models"},
		},
		"2507.07955v2": {
			"Ratio Loss": {"2.2.3", ""},
		},
	}
	for _, p := range ingest.Papers {
		path := "../data/papers/" + p.ID + ".pdf"
		if _, err := os.Stat(path); err != nil {
			t.Skipf("%s not downloaded", path)
		}
		pages, outline, err := ingest.Extract(path)
		if err != nil {
			t.Fatal(err)
		}
		heads, err := ingest.ParseOutline(outline)
		if err != nil {
			t.Fatal(err)
		}
		segs, warns := ingest.Segments(pages, heads, p.Extra)
		if len(warns) > 0 {
			t.Errorf("%s warnings: %v", p.ID, warns)
		}
		for _, s := range segs {
			if s.Number == "" && !strings.Contains(strings.Join(p.Extra, "|")+"|Title page", s.Title) {
				t.Errorf("%s: section %q has no number", p.ID, s.Title)
			}
		}
		for title, w := range expect[p.ID] {
			s := find(segs, title)
			if s == nil {
				t.Errorf("%s: section %q missing", p.ID, title)
				continue
			}
			if s.Number != w.number || !strings.Contains(text(s), w.mustContain) {
				t.Errorf("%s %q: number %q (want %q), starts %.80q", p.ID, title, s.Number, w.number, text(s))
			}
		}
	}
}
