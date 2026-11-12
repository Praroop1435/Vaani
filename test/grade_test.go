package test

import (
	"regexp"
	"slices"
	"testing"

	"github.com/Praroop1435/Vaani/internal/eval"
	"github.com/Praroop1435/Vaani/internal/ingest"
)

// Regressions from the 2026-11-11 live smoke test, where the grader marked right answers wrong.
func TestNormTextUnitsUnicodeAndHyphens(t *testing.T) {
	cases := []struct{ text, pattern string }{
		{"Helium is a 7 billion‑parameter model", `7\s*B`}, // what gpt-oss actually wrote
		{"a 7B-parameter text LLM", `7\s*B`},
		{"pretrained on 2.1 trillion tokens", `2\.1\s*T`},
		{"about 4.5 million sentence pairs", `4\.5\s*M`},
		{"on the WMT 2014 English‑to‑German task", `WMT 2014 English-to-German`},
	}
	for _, c := range cases {
		if !regexp.MustCompile(c.pattern).MatchString(eval.NormText(c.text)) {
			t.Errorf("%q normalised to %q does not match %q", c.text, eval.NormText(c.text), c.pattern)
		}
	}
	if got := eval.NormText("12.5 Hz at 80 ms"); got != "12.5 Hz at 80 ms" {
		t.Errorf("non-magnitude units must be left alone, got %q", got)
	}
}

func TestEvidenceIgnoresCitationMarkers(t *testing.T) {
	// The Table 1 caption chunk is built from LaTeX, where \citep{} is dropped.
	caption := ingest.Chunk{Paper: "2410.00037v1", Kind: "table", Text: "Table 1 (page 8): ... Fine-tuning on the Fisher dataset to gain its fully duplex capabilities; ..."}
	src := eval.Source{Paper: "2410.00037v1", Page: 8, Quote: "Fine-tuning on the Fisher dataset (Cieri et al., 2004) to gain its fully duplex capabilities"}
	if !src.Covers(caption) {
		t.Error("author-year citation in the quote blocked a match")
	}
	pdf := ingest.Chunk{Paper: "1706.03762v7", Text: "We used the Adam optimizer [20] with β1 = 0.9"}
	if !(eval.Source{Paper: "1706.03762v7", Quote: "We used the Adam optimizer with β1 = 0.9"}).Covers(pdf) {
		t.Error("numeric reference [20] blocked a match")
	}
	if (eval.Source{Paper: "1706.03762v7", Quote: "Fisher dataset"}).Covers(caption) {
		t.Error("a source must not match a chunk of another paper")
	}
}

func TestGradeSmokeTestQ14Answer(t *testing.T) {
	b, err := eval.Load("../data/eval/benchmark.json")
	if err != nil {
		t.Fatal(err)
	}
	q := b.Questions[slices.IndexFunc(b.Questions, func(q eval.Question) bool { return q.ID == "q14" })]
	// verbatim from results/smoke/20261111-182937_traces.jsonl
	answer := "Helium is a 7 billion‑parameter model [Moshi, p.8]. It was pretrained with a batch size of 4.2 million tokens per step over 500 k steps, i.e., using 4.2 M‑token batches [Moshi, p.20]. Moshi gains its full‑duplex capability by fine‑tuning on the Fisher dataset [Moshi, p.8]."
	caption := ingest.Chunk{Paper: "2410.00037v1", Kind: "table", Table: "Table 1", PageStart: 8, PageEnd: 8,
		Text: "Table 1 (page 8): Models' hyperparameters. Hyperparameters for the architecture and training of our 7B-parameter Helium language model and of Moshi ... Fine-tuning on the Fisher dataset to gain its fully duplex capabilities ..."}
	g := eval.GradeAnswer(q, answer, false, []ingest.Chunk{caption})
	// still wrong, for the right reason only: the 2.1T total is missing
	if g.Grade != "wrong" || !slices.Equal(g.Notes[:1], []string{`missing 2\.1\s*T`}) {
		t.Errorf("grade %s notes %q", g.Grade, g.Notes)
	}
	// citations: 7B and Fisher are cited (caption), the 2.1T item is not
	if g.CiteOK || !slices.Contains(g.Notes, "evidence item 2 not cited") || slices.Contains(g.Notes, "evidence item 1 not cited") || slices.Contains(g.Notes, "evidence item 3 not cited") {
		t.Errorf("citation notes %q", g.Notes)
	}
}
