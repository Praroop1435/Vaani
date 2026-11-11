package test

import (
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"

	"github.com/Praroop1435/Vaani/internal/ingest"
)

// A table written the way these papers write them: macros, \multicolumn header groups under
// \cmidrule, \multirow labels (down and upward), a group-label row, comments and citations.
const sampleTable = `
\newcommand{\dmodel}{d_{\text{model}}}
\newcommand{\best}[1]{\textbf{#1}}
\begin{table}[t]
\caption{\label{tab:demo}The demo model gets better \textbf{BLEU} scores. % a comment
}
\begin{tabular}{lccc}
\toprule
\multirow{2}{*}{Model} & \multicolumn{2}{c}{BLEU} & \multirow{2}{*}{$\dmodel$} \\
\cmidrule{2-3}
& EN-DE & EN-FR & \\
\midrule
\multicolumn{4}{l}{Small models} \\
ByteNet \citep{bytenet} & 23.75 & & \\
Big & \best{28.4} & $41.8\cdot10^{1}$ & \multirow{-2}{*}{1024} \\
Done \ding{51} & 50\% & -- & \\
\bottomrule
\end{tabular}
\end{table}`

func TestParseTableStructure(t *testing.T) {
	tables, warns := ingest.ParseTables("demo", sampleTable)
	if len(tables) != 1 {
		t.Fatalf("got %d tables, warnings %v", len(tables), warns)
	}
	tb := tables[0]
	if tb.Label != "tab:demo" || tb.Caption != "The demo model gets better BLEU scores." {
		t.Errorf("label %q caption %q", tb.Label, tb.Caption)
	}
	g := tb.Parts[0]
	if want := []string{"Model", "BLEU EN-DE", "BLEU EN-FR", "d_model"}; !slices.Equal(g.Header, want) {
		t.Errorf("header = %q, want %q", g.Header, want)
	}
	want := [][]string{
		{"Small models", "", "", ""},     // group label written once
		{"ByteNet", "23.75", "", "1024"}, // upward \multirow{-2} fills the row above
		{"Big", "28.4", "41.8·10^1", "1024"},
		{"Done ✓", "50%", "--", ""},
	}
	if len(g.Rows) != len(want) {
		t.Fatalf("rows = %q", g.Rows)
	}
	for i := range want {
		if !slices.Equal(g.Rows[i], want[i]) {
			t.Errorf("row %d = %q, want %q", i, g.Rows[i], want[i])
		}
	}
}

func TestLocateTablesFromPDFCaption(t *testing.T) {
	tables, _ := ingest.ParseTables("demo", sampleTable)
	pages := []string{"Intro text.", "Results here.\nTable 7: The demo model gets better BLEU\nscores. More text."}
	segs := []ingest.Segment{{Number: "6", Title: "Results", Path: "Results", Sentences: []ingest.Sentence{
		{Text: "Results here.", Page: 2}, {Text: "Table 7: The demo model gets better BLEU scores.", Page: 2},
	}}}
	if w := ingest.LocateTables(tables, pages, segs); len(w) > 0 {
		t.Fatal(w)
	}
	if tables[0].Number != "7" || tables[0].Page != 2 || tables[0].SectionNumber != "6" {
		t.Errorf("located as Table %q page %d section %q", tables[0].Number, tables[0].Page, tables[0].SectionNumber)
	}

	chunks := ingest.TableChunks(tables, 40, wordCount, 100) // small limit forces a row split
	if len(chunks) < 2 {
		t.Fatalf("expected the table split into several chunks, got %d", len(chunks))
	}
	for _, c := range chunks {
		if c.Kind != "table" || c.Table != "Table 7" || c.PageStart != 2 || c.Order < 100 {
			t.Errorf("chunk meta wrong: %+v", c)
		}
		if !strings.HasPrefix(c.Text, "Table 7 (page 2): The demo model") || !strings.Contains(c.Text, "| Model | BLEU EN-DE |") {
			t.Errorf("every part must repeat caption and header:\n%s", c.Text)
		}
	}
}

func TestRealPaperTables(t *testing.T) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		t.Skip("pdftotext not installed")
	}
	// cell values read off the PDFs by hand
	type cellCheck struct {
		paper, table, row, column, value string
		page                             int
	}
	checks := []cellCheck{
		{"1706.03762v7", "2", "Transformer (big)", "BLEU EN-DE", "28.4", 8},
		{"1706.03762v7", "2", "Transformer (base model)", "BLEU EN-FR", "38.1", 8},
		{"1706.03762v7", "3", "big", "d_model", "1024", 9},
		{"2410.00037v1", "1", "Model dimension", "Helium training pre-training", "4096", 8},
		{"2507.07955v2", "4", "H-Net (2-stage)", "Code Val. BPB ↓", "0.3161", 17},
	}
	byPaper := map[string][]ingest.Table{}
	for _, p := range ingest.Papers {
		pdf, src := "../data/papers/"+p.ID+".pdf", "../data/latex/"+p.ID
		if _, err := os.Stat(src); err != nil {
			t.Skip("LaTeX sources not fetched: run `go run -tags ORT ./cmd/ingest -tables-only`")
		}
		pages, outline, err := ingest.Extract(pdf)
		if err != nil {
			t.Fatal(err)
		}
		heads, _ := ingest.ParseOutline(outline)
		segs, _ := ingest.Segments(pages, heads, p.Extra)
		text, err := ingest.LoadSource(src)
		if err != nil {
			t.Fatal(err)
		}
		tables, _ := ingest.ParseTables(p.ID, text)
		ingest.LocateTables(tables, pages, segs)
		byPaper[p.ID] = tables
	}
	for _, c := range checks {
		var tb *ingest.Table
		for i, x := range byPaper[c.paper] {
			if x.Number == c.table {
				tb = &byPaper[c.paper][i]
			}
		}
		if tb == nil {
			t.Errorf("%s Table %s not found", c.paper, c.table)
			continue
		}
		if tb.Page != c.page {
			t.Errorf("%s Table %s on page %d, want %d", c.paper, c.table, tb.Page, c.page)
		}
		g := tb.Parts[0]
		col := slices.Index(g.Header, c.column)
		if col < 0 {
			t.Errorf("%s Table %s: no column %q in %q", c.paper, c.table, c.column, g.Header)
			continue
		}
		got := "<row missing>"
		for _, r := range g.Rows {
			if r[0] == c.row && got == "<row missing>" { // first match: Moshi Table 1 repeats row names per block
				got = r[col]
			}
		}
		if got != c.value {
			t.Errorf("%s Table %s [%s][%s] = %q, want %q", c.paper, c.table, c.row, c.column, got, c.value)
		}
	}
}
