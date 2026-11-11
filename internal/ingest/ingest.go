// Package ingest turns the three paper PDFs into section-aware chunks with page numbers.
//
// Sections come from each PDF's own outline (bookmarks), not from guessing headings in the text:
// tables and page numbers look like headings to a regex. `pdftotext` gives one text block per page.
package ingest

import (
	"bytes"
	"cmp"
	"encoding/xml"
	"fmt"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

type Paper struct {
	ID    string // arXiv id with version, also the PDF file name
	Title string
	// Headings the PDF outline leaves out but the text has on a line of their own.
	Extra []string
}

var Papers = []Paper{
	{ID: "1706.03762v7", Title: "Attention Is All You Need", Extra: []string{"Abstract", "References", "Attention Visualizations"}},
	{ID: "2410.00037v1", Title: "Moshi: a speech-text foundation model for real-time dialogue", Extra: []string{"Abstract", "References"}},
	{ID: "2507.07955v2", Title: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", Extra: []string{"Abstract", "Acknowledgements", "References"}},
}

// SkipSections are not indexed: a bibliography matches almost any query and answers none.
var SkipSections = map[string]bool{"References": true}

type Heading struct {
	Title string
	Page  int // 1-based
	Depth int // 1 = top level
}

type Sentence struct {
	Text string
	Page int
}

type Segment struct {
	Number    string // "3.2.2", "A", or "" when the PDF has none
	Title     string
	Path      string // "Model Architecture > Attention > Multi-Head Attention"
	Sentences []Sentence
}

type Chunk struct {
	ID            string `json:"id"`
	Kind          string `json:"kind"`            // "text" or "table"
	Table         string `json:"table,omitempty"` // "Table 3" for table chunks
	Paper         string `json:"paper"`
	SectionNumber string `json:"section_number"`
	Section       string `json:"section"`
	SectionPath   string `json:"section_path"`
	PageStart     int    `json:"page_start"`
	PageEnd       int    `json:"page_end"`
	Order         int    `json:"order"` // position within the paper, for graph neighbours
	Tokens        int    `json:"tokens"`
	Text          string `json:"text"`
}

// Extract runs poppler: per-page text, and the outline from the XML export.
func Extract(pdfPath string) (pages []string, outlineXML []byte, err error) {
	txt, err := exec.Command("pdftotext", "-enc", "UTF-8", pdfPath, "-").Output()
	if err != nil {
		return nil, nil, fmt.Errorf("pdftotext %s: %w", pdfPath, err)
	}
	pages = strings.Split(string(txt), "\f")
	if len(pages) > 0 && strings.TrimSpace(pages[len(pages)-1]) == "" {
		pages = pages[:len(pages)-1]
	}
	outlineXML, err = exec.Command("pdftohtml", "-xml", "-i", "-q", "-stdout", pdfPath).Output()
	if err != nil {
		return nil, nil, fmt.Errorf("pdftohtml %s: %w", pdfPath, err)
	}
	return pages, outlineXML, nil
}

// ParseOutline reads <outline><item page="N">Title</item><outline>...</outline></outline> from pdftohtml -xml.
func ParseOutline(data []byte) ([]Heading, error) {
	start := bytes.Index(data, []byte("<outline"))
	if start < 0 {
		return nil, fmt.Errorf("ingest: PDF has no outline")
	}
	dec := xml.NewDecoder(bytes.NewReader(data[start:]))
	dec.Strict = false
	var heads []Heading
	depth := 0
	for {
		tok, err := dec.Token()
		if err != nil {
			break // end of outline region or of document
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "outline":
				depth++
			case "item":
				var page int
				for _, a := range t.Attr {
					if a.Name.Local == "page" {
						page, _ = strconv.Atoi(a.Value)
					}
				}
				var title string
				if err := dec.DecodeElement(&title, &t); err != nil {
					return nil, err
				}
				heads = append(heads, Heading{Title: strings.Join(strings.Fields(title), " "), Page: page, Depth: depth})
			}
		case xml.EndElement:
			if t.Name.Local == "outline" {
				depth--
				if depth == 0 {
					return heads, nil
				}
			}
		}
	}
	return heads, nil
}

var (
	numberLine   = regexp.MustCompile(`^([0-9]+(\.[0-9]+)*|[A-Z](\.[0-9]+)*)\.?$`)
	inlineNumber = regexp.MustCompile(`^(?:Appendix\s+)?([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)\.?\s+(.+)$`)
	arxivStamp   = regexp.MustCompile(`^arXiv:\d{4}\.\d{4,5}v\d`)
	ligatures    = strings.NewReplacer("ﬁ", "fi", "ﬂ", "fl", "ﬀ", "ff", "ﬃ", "ffi", "ﬄ", "ffl")
)

type marker struct {
	page, line int // 0-based page index and line index
	skipFrom   int // first line of the heading (the number line when it stands alone)
	head       Heading
	number     string
}

// matchHeading reports whether line is the heading title, bare or with a section number in front.
func matchHeading(line, title string) (number string, ok bool) {
	if line == title {
		return "", true
	}
	if m := inlineNumber.FindStringSubmatch(line); m != nil && m[2] == title {
		return m[1], true
	}
	return "", false
}

// findMarkers places every heading on a (page, line), searching forward from the previous one
// so headings stay in reading order. On a page, a numbered match beats a bare one: figure labels
// repeat section titles ("Scaled Dot-Product Attention" in Figure 2) but carry no section number.
// A heading not found on its page or the next is a warning.
func findMarkers(lines [][]string, heads []Heading, extra []string) ([]marker, []string) {
	var marks []marker
	var warns []string
	pos := [2]int{0, 0}
	for _, h := range heads {
		found := false
		for p := max(h.Page-1, pos[0]); p < len(lines) && p <= h.Page && !found; p++ {
			from := 0
			if p == pos[0] {
				from = pos[1]
			}
			var bare, numbered *marker
			for i := from; i < len(lines[p]) && numbered == nil; i++ {
				num, ok := matchHeading(lines[p][i], h.Title)
				if !ok {
					continue
				}
				skip := i
				if num == "" {
					for j := i - 1; j >= 0; j-- { // number printed on its own line above the title
						if lines[p][j] == "" {
							continue
						}
						if numberLine.MatchString(lines[p][j]) {
							num, skip = strings.TrimSuffix(lines[p][j], "."), j
						}
						break
					}
				}
				m := marker{page: p, line: i, skipFrom: skip, head: h, number: num}
				if num != "" {
					numbered = &m
				} else if bare == nil {
					bare = &m
				}
			}
			if m := cmp.Or(numbered, bare); m != nil {
				marks = append(marks, *m)
				pos = [2]int{p, m.line + 1}
				found = true
			}
		}
		if !found {
			warns = append(warns, fmt.Sprintf("heading %q not found near page %d", h.Title, h.Page))
		}
	}
	for _, title := range extra {
		for p := range lines {
			for i, l := range lines[p] {
				if l == title {
					marks = append(marks, marker{page: p, line: i, skipFrom: i, head: Heading{Title: title, Page: p + 1, Depth: 1}})
				}
			}
		}
	}
	// sort by where the heading starts (insertion sort: a few dozen markers)
	for i := 1; i < len(marks); i++ {
		for j := i; j > 0 && (marks[j].page < marks[j-1].page || marks[j].page == marks[j-1].page && marks[j].skipFrom < marks[j-1].skipFrom); j-- {
			marks[j], marks[j-1] = marks[j-1], marks[j]
		}
	}
	return marks, warns
}

// Segments cuts the pages into sections. Text before the first heading is the "Title page".
func Segments(pages []string, heads []Heading, extra []string) ([]Segment, []string) {
	lines := make([][]string, len(pages))
	for p, pg := range pages {
		for _, l := range strings.Split(ligatures.Replace(pg), "\n") {
			lines[p] = append(lines[p], strings.TrimSpace(l))
		}
	}
	marks, warns := findMarkers(lines, heads, extra)

	segs := []Segment{{Title: "Title page", Path: "Title page"}}
	var stack []string   // titles by depth
	var numbers []string // last section number by depth, for headings the PDF prints without one
	mi := 0
	for p := range lines {
		var buf []string
		flush := func() {
			if text := strings.Join(strings.Fields(strings.Join(buf, " ")), " "); text != "" {
				cur := &segs[len(segs)-1]
				for _, s := range splitSentences(text) {
					cur.Sentences = append(cur.Sentences, Sentence{Text: s, Page: p + 1})
				}
			}
			buf = buf[:0]
		}
		for i := 0; i < len(lines[p]); i++ {
			for mi < len(marks) && (marks[mi].page < p || marks[mi].page == p && marks[mi].line < i) {
				warns = append(warns, fmt.Sprintf("heading %q overlaps the previous one, dropped", marks[mi].head.Title))
				mi++
			}
			if mi < len(marks) && marks[mi].page == p && marks[mi].skipFrom <= i {
				flush()
				m := marks[mi]
				d := m.head.Depth
				stack = append(stack[:min(d-1, len(stack))], m.head.Title)
				if !slices.Contains(extra, m.head.Title) {
					if m.number == "" {
						m.number = inferNumber(numbers, d)
					}
					for len(numbers) < d {
						numbers = append(numbers, "")
					}
					numbers = append(numbers[:d-1], m.number)
				}
				segs = append(segs, Segment{Number: m.number, Title: m.head.Title, Path: strings.Join(stack, " > ")})
				i = m.line // skip the heading lines
				mi++
				continue
			}
			l := lines[p][i]
			if l == strconv.Itoa(p+1) || arxivStamp.MatchString(l) { // page number, arXiv side stamp
				continue
			}
			buf = append(buf, l)
		}
		flush()
	}
	return segs, warns
}

// inferNumber gives the next sibling's number ("3.4" -> "3.5", "E.1" -> "E.2", "A" -> "B"),
// or the first child's ("3.2" -> "3.2.1"). Empty when there is nothing to count from.
func inferNumber(numbers []string, depth int) string {
	if depth <= len(numbers) && numbers[depth-1] != "" {
		prev := numbers[depth-1]
		i := strings.LastIndex(prev, ".")
		last := prev[i+1:]
		if n, err := strconv.Atoi(last); err == nil {
			return prev[:i+1] + strconv.Itoa(n+1)
		}
		if len(last) == 1 && last[0] >= 'A' && last[0] < 'Z' {
			return prev[:i+1] + string(last[0]+1)
		}
		return ""
	}
	if depth >= 2 && depth-1 <= len(numbers) && numbers[depth-2] != "" {
		return numbers[depth-2] + ".1"
	}
	return ""
}

// splitSentences breaks after . ? ! when the next word starts with a capital, digit or bracket.
// Abbreviations ("et al. Smith") can split early; chunk packing joins neighbours back anyway.
func splitSentences(text string) []string {
	var out []string
	start := 0
	for i := 0; i+2 < len(text); i++ {
		c := text[i]
		if (c == '.' || c == '?' || c == '!') && text[i+1] == ' ' {
			n := text[i+2]
			if n >= 'A' && n <= 'Z' || n >= '0' && n <= '9' || n == '(' || n == '[' {
				out = append(out, text[start:i+1])
				start = i + 2
			}
		}
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

// CountFunc returns token counts including the model's 2 special tokens.
type CountFunc func([]string) []int

// Pack fills chunks up to target tokens without crossing a section. No chunk exceeds maxTokens.
func Pack(paper string, segs []Segment, target, maxTokens int, count CountFunc) []Chunk {
	var chunks []Chunk
	for _, seg := range segs {
		if SkipSections[seg.Title] || len(seg.Sentences) == 0 {
			continue
		}
		texts := make([]string, len(seg.Sentences))
		for i, s := range seg.Sentences {
			texts[i] = s.Text
		}
		sizes := count(texts)
		var cur []Sentence
		curTok := 0
		emit := func() {
			if len(cur) == 0 {
				return
			}
			for _, part := range fit(cur, maxTokens, count) {
				chunks = append(chunks, Chunk{
					Kind: "text", Paper: paper, SectionNumber: seg.Number, Section: seg.Title, SectionPath: seg.Path,
					PageStart: part[0].Page, PageEnd: part[len(part)-1].Page, Text: join(part),
				})
			}
			cur, curTok = nil, 0
		}
		for i, s := range seg.Sentences {
			t := sizes[i] - 2
			if curTok > 0 && curTok+t > target {
				emit()
			}
			cur = append(cur, s)
			curTok += t
		}
		emit()
	}
	finals := make([]string, len(chunks))
	for i := range chunks {
		finals[i] = chunks[i].Text
	}
	for i, n := range count(finals) {
		chunks[i].ID = fmt.Sprintf("%s#c%03d", paper, i)
		chunks[i].Order = i
		chunks[i].Tokens = n
	}
	return chunks
}

// fit splits a run of sentences (and, if needed, one long sentence by words) until each part fits.
func fit(ss []Sentence, maxTokens int, count CountFunc) [][]Sentence {
	if count([]string{join(ss)})[0] <= maxTokens {
		return [][]Sentence{ss}
	}
	if len(ss) > 1 {
		mid := len(ss) / 2
		return append(fit(ss[:mid], maxTokens, count), fit(ss[mid:], maxTokens, count)...)
	}
	words := strings.Fields(ss[0].Text)
	if len(words) < 2 {
		return [][]Sentence{ss} // a single unsplittable token run; Embed will reject it loudly
	}
	mid := len(words) / 2
	a := Sentence{Text: strings.Join(words[:mid], " "), Page: ss[0].Page}
	b := Sentence{Text: strings.Join(words[mid:], " "), Page: ss[0].Page}
	return append(fit([]Sentence{a}, maxTokens, count), fit([]Sentence{b}, maxTokens, count)...)
}

func join(ss []Sentence) string {
	parts := make([]string, len(ss))
	for i, s := range ss {
		parts[i] = s.Text
	}
	return strings.Join(parts, " ")
}
