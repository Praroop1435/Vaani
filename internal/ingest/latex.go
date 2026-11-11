package ingest

// Tables come from each paper's arXiv LaTeX source (same version as the PDF): pdftotext flattens
// tables into number soup, while the source holds every row and cell. Table number and page come
// from matching the caption against the PDF, so citations still point at real PDF pages.

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Grid is one tabular: merged header row plus data rows, all the same width.
type Grid struct {
	Header []string   `json:"header"`
	Rows   [][]string `json:"rows"`
}

type Table struct {
	Paper         string `json:"paper"`
	Number        string `json:"number"` // "3", as printed in the PDF caption "Table 3:"
	Label         string `json:"label"`  // LaTeX \label, e.g. "tab:variations"
	Caption       string `json:"caption"`
	Page          int    `json:"page"`
	SectionNumber string `json:"section_number"`
	Section       string `json:"section"`
	SectionPath   string `json:"section_path"`
	Parts         []Grid `json:"parts"` // side-by-side sub-tables stay separate
}

// FetchSource downloads the arXiv e-print once and keeps only .tex files under dir.
func FetchSource(id, dir string) error {
	if files, _ := filepath.Glob(filepath.Join(dir, "*.tex")); len(files) > 0 {
		return nil
	}
	req, _ := http.NewRequest(http.MethodGet, "https://arxiv.org/e-print/"+id, nil)
	req.Header.Set("User-Agent", "relay-assignment/0.1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("e-print %s: HTTP %d", id, resp.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 200<<20))
	if err != nil {
		return err
	}
	gz, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("e-print %s: not gzip: %w", id, err)
	}
	data, err := io.ReadAll(io.LimitReader(gz, 500<<20))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tr := tar.NewReader(bytes.NewReader(data))
	n := 0
	for {
		h, err := tr.Next()
		if err != nil {
			break
		}
		name := filepath.Clean(h.Name)
		if h.Typeflag != tar.TypeReg || !strings.HasSuffix(name, ".tex") || filepath.IsAbs(name) || strings.HasPrefix(name, "..") {
			continue // only plain .tex files, never paths outside dir
		}
		dst := filepath.Join(dir, strings.ReplaceAll(name, string(filepath.Separator), "__"))
		body, err := io.ReadAll(io.LimitReader(tr, 20<<20))
		if err != nil {
			return err
		}
		if err := os.WriteFile(dst, body, 0o644); err != nil {
			return err
		}
		n++
	}
	if n == 0 { // a single gzipped .tex file, not a tar
		return os.WriteFile(filepath.Join(dir, "main.tex"), data, 0o644)
	}
	return nil
}

// LoadSource joins all .tex files with comments removed. File order does not matter:
// table numbers come from the PDF.
func LoadSource(dir string) (string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.tex"))
	if err != nil || len(files) == 0 {
		return "", fmt.Errorf("no .tex files in %s", dir)
	}
	sort.Strings(files)
	var b strings.Builder
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return "", err
		}
		for _, line := range strings.Split(string(data), "\n") {
			b.WriteString(stripComment(line))
			b.WriteByte('\n')
		}
	}
	return b.String(), nil
}

func stripComment(line string) string {
	for i := 0; i < len(line); i++ {
		if line[i] == '\\' {
			i++ // skip escaped char, e.g. \%
			continue
		}
		if line[i] == '%' {
			return line[:i]
		}
	}
	return line
}

// --- small brace-aware helpers ---

// group reads a {...} group starting at s[i] (after optional spaces). ok=false if none.
func group(s string, i int) (content string, end int, ok bool) {
	for i < len(s) && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t') {
		i++
	}
	if i >= len(s) || s[i] != '{' {
		return "", i, false
	}
	depth := 0
	for j := i; j < len(s); j++ {
		switch s[j] {
		case '\\':
			j++
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[i+1 : j], j + 1, true
			}
		}
	}
	return s[i+1:], len(s), true
}

// skipOpt skips [..] or (..) optional arguments.
func skipOpt(s string, i int) int {
	for {
		j := i
		for j < len(s) && s[j] == ' ' {
			j++
		}
		if j < len(s) && (s[j] == '[' || s[j] == '(') {
			closeCh := map[byte]byte{'[': ']', '(': ')'}[s[j]]
			if k := strings.IndexByte(s[j:], closeCh); k >= 0 {
				i = j + k + 1
				continue
			}
		}
		return i
	}
}

// envSpan finds \begin{name}...\end{name} from pos, handling nesting. Returns body bounds.
func envSpan(s, name string, pos int) (start, bodyStart, bodyEnd, end int, ok bool) {
	open, closeTag := `\begin{`+name+`}`, `\end{`+name+`}`
	start = strings.Index(s[pos:], open)
	if start < 0 {
		return 0, 0, 0, 0, false
	}
	start += pos
	bodyStart = start + len(open)
	depth := 1
	for i := bodyStart; i < len(s); {
		o, c := strings.Index(s[i:], open), strings.Index(s[i:], closeTag)
		if c < 0 {
			return 0, 0, 0, 0, false
		}
		if o >= 0 && o < c {
			depth++
			i += o + len(open)
			continue
		}
		depth--
		if depth == 0 {
			return start, bodyStart, i + c, i + c + len(closeTag), true
		}
		i += c + len(closeTag)
	}
	return 0, 0, 0, 0, false
}

// --- macros ---

type macro struct {
	args int
	body string
}

var newcommandRe = regexp.MustCompile(`\\(?:re)?newcommand\*?\s*\{?\\([a-zA-Z]+)\}?\s*(?:\[(\d)\])?`)

func parseMacros(src string) map[string]macro {
	ms := map[string]macro{}
	for _, m := range newcommandRe.FindAllStringSubmatchIndex(src, -1) {
		name := src[m[2]:m[3]]
		args := 0
		if m[4] >= 0 {
			args = int(src[m[4]] - '0')
		}
		if body, _, ok := group(src, m[1]); ok {
			ms[name] = macro{args, body}
		}
	}
	return ms
}

func expandMacros(s string, ms map[string]macro) string {
	for pass := 0; pass < 5; pass++ {
		changed := false
		var b strings.Builder
		for i := 0; i < len(s); {
			if s[i] != '\\' {
				b.WriteByte(s[i])
				i++
				continue
			}
			j := i + 1
			for j < len(s) && isLetter(s[j]) {
				j++
			}
			m, ok := ms[s[i+1:j]]
			if j == i+1 || !ok {
				b.WriteString(s[i:max(j, i+2)])
				i = max(j, i+2)
				continue
			}
			body, k := m.body, j
			for a := 1; a <= m.args; a++ {
				arg, end, ok := group(s, k)
				if !ok {
					break
				}
				body = strings.ReplaceAll(body, fmt.Sprintf("#%d", a), arg)
				k = end
			}
			if m.args == 0 && k < len(s) && s[k] == '{' && k+1 < len(s) && s[k+1] == '}' {
				k += 2 // "\model{}" idiom
			}
			b.WriteString(body)
			i, changed = k, true
		}
		s = b.String()
		if !changed {
			break
		}
	}
	return s
}

func isLetter(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }

// --- cell text cleaning ---

type cmdSpec struct {
	args int    // brace arguments to read
	keep int    // which argument to keep (0-based), -1 for none
	out  string // replacement when keep == -1
}

var commands = map[string]cmdSpec{
	// formatting: keep the content
	"textbf": {1, 0, ""}, "textit": {1, 0, ""}, "emph": {1, 0, ""}, "underline": {1, 0, ""},
	"texttt": {1, 0, ""}, "textsc": {1, 0, ""}, "textrm": {1, 0, ""}, "textsf": {1, 0, ""},
	"mathrm": {1, 0, ""}, "mathbf": {1, 0, ""}, "mathit": {1, 0, ""}, "mathcal": {1, 0, ""},
	"mathbb": {1, 0, ""}, "text": {1, 0, ""}, "mbox": {1, 0, ""}, "makecell": {1, 0, ""},
	"textcolor": {2, 1, ""}, "colorbox": {2, 1, ""}, "scalebox": {2, 1, ""}, "hyperref": {1, 0, ""},
	"resizebox": {3, 2, ""}, "sqrt": {1, 0, "√"},
	// spans not at the start of a cell (e.g. after \rule): keep the text, lose the span
	"multicolumn": {3, 2, ""}, "multirow": {3, 2, ""}, "multirowcell": {2, 1, ""},
	"eqref": {1, -1, ""}, "hfill": {}, "protect": {},
	"Delta": {0, -1, "Δ"}, "Phi": {0, -1, "Φ"}, "varnothing": {0, -1, "∅"},
	// drop entirely
	"cite": {1, -1, ""}, "citep": {1, -1, ""}, "citet": {1, -1, ""}, "label": {1, -1, ""},
	"footnote": {1, -1, ""}, "ref": {1, -1, ""}, "cref": {1, -1, ""}, "rule": {2, -1, ""},
	"vspace": {1, -1, ""}, "hspace": {1, -1, ""}, "hphantom": {1, -1, ""}, "phantom": {1, -1, ""},
	"cmidrule": {1, -1, ""}, "cline": {1, -1, ""}, "specialrule": {3, -1, ""}, "cellcolor": {1, -1, ""},
	"rowcolor": {1, -1, ""}, "color": {1, -1, ""}, "includegraphics": {1, -1, ""},
	"hline": {}, "toprule": {}, "midrule": {}, "bottomrule": {}, "addlinespace": {}, "centering": {},
	"small": {}, "footnotesize": {}, "scriptsize": {}, "tiny": {}, "normalsize": {}, "bf": {}, "it": {},
	"boldmath": {}, "xspace": {}, "noindent": {}, "displaystyle": {}, "left": {}, "right": {},
	// symbols
	"cdot": {0, -1, "·"}, "times": {0, -1, "×"}, "pm": {0, -1, "±"}, "approx": {0, -1, "≈"},
	"leq": {0, -1, "≤"}, "le": {0, -1, "≤"}, "geq": {0, -1, "≥"}, "ge": {0, -1, "≥"}, "neq": {0, -1, "≠"},
	"sim": {0, -1, "~"}, "uparrow": {0, -1, "↑"}, "downarrow": {0, -1, "↓"}, "rightarrow": {0, -1, "→"},
	"to": {0, -1, "→"}, "checkmark": {0, -1, "✓"}, "dagger": {0, -1, "†"}, "ddagger": {0, -1, "‡"},
	"ast": {0, -1, "*"}, "star": {0, -1, "*"}, "infty": {0, -1, "∞"}, "ldots": {0, -1, "…"},
	"dots": {0, -1, "…"}, "quad": {0, -1, " "}, "qquad": {0, -1, " "}, "in": {0, -1, "∈"},
	"alpha": {0, -1, "α"}, "beta": {0, -1, "β"}, "gamma": {0, -1, "γ"}, "delta": {0, -1, "δ"},
	"epsilon": {0, -1, "ε"}, "lambda": {0, -1, "λ"}, "mu": {0, -1, "μ"}, "tau": {0, -1, "τ"},
	"theta": {0, -1, "θ"}, "sigma": {0, -1, "σ"}, "log": {0, -1, "log"},
}

var spaces = regexp.MustCompile(`\s+`)

// clean turns a LaTeX cell into plain text. unknown collects commands it had to guess at.
func clean(s string, unknown map[string]int) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\\':
			j := i + 1
			for j < len(s) && isLetter(s[j]) {
				j++
			}
			if j == i+1 { // escaped character: \% \& \_ \# \, \; \!
				if i+1 < len(s) {
					switch s[i+1] {
					case '%', '&', '_', '#', '$', '{', '}':
						b.WriteByte(s[i+1])
					default:
						b.WriteByte(' ')
					}
				}
				i += 2
				continue
			}
			name := s[i+1 : j]
			if j < len(s) && s[j] == '*' {
				j++
			}
			j = skipOpt(s, j)
			if name == "ding" {
				arg, end, _ := group(s, j)
				b.WriteString(map[string]string{"51": "✓", "55": "✗"}[strings.TrimSpace(arg)])
				i = end
				continue
			}
			spec, known := commands[name]
			if !known {
				unknown[name]++
				spec = cmdSpec{1, 0, ""} // unknown: keep its argument if it has one
				if _, _, ok := group(s, j); !ok {
					spec = cmdSpec{}
				}
			}
			var args []string
			for a := 0; a < spec.args; a++ {
				arg, end, ok := group(s, j)
				if !ok {
					break
				}
				args = append(args, arg)
				j = skipOpt(s, end)
			}
			if spec.keep >= 0 && spec.keep < len(args) {
				b.WriteString(spec.out + clean(args[spec.keep], unknown))
			} else {
				b.WriteString(spec.out)
			}
			i = j
		case c == '$' || c == '{' || c == '}':
			i++
		case c == '~':
			b.WriteByte(' ')
			i++
		default:
			b.WriteByte(c)
			i++
		}
	}
	return strings.TrimSpace(spaces.ReplaceAllString(b.String(), " "))
}

// --- tabular parsing ---

var ruleCmd = regexp.MustCompile(`^\s*\\(hline|toprule|midrule|bottomrule|cmidrule|cline|specialrule|addlinespace)\b`)

// splitTop splits at sep outside braces, ignoring escaped separators.
func splitTop(s, sep string) []string {
	var parts []string
	depth, last := 0, 0
	for i := 0; i < len(s); i++ {
		switch {
		case s[i] == '\\' && !strings.HasPrefix(s[i:], sep):
			i++
		case s[i] == '{':
			depth++
		case s[i] == '}':
			depth--
		case depth == 0 && strings.HasPrefix(s[i:], sep):
			parts = append(parts, s[last:i])
			i += len(sep) - 1
			last = i + 1
		}
	}
	return append(parts, s[last:])
}

type cell struct {
	text     string
	span     int // columns covered
	rowsDown int // \multirow: rows covered
}

func parseCell(raw string, unknown map[string]int) cell {
	s := strings.TrimSpace(raw)
	c := cell{span: 1, rowsDown: 1}
	for { // \multicolumn{n}{spec}{x} and \multirow{n}{w}{x} (\multirowcell{n}{x}), possibly nested
		switch {
		case strings.HasPrefix(s, `\multicolumn`):
			n, e1, _ := group(s, len(`\multicolumn`))
			_, e2, _ := group(s, e1)
			x, _, _ := group(s, e2)
			fmt.Sscan(n, &c.span)
			s = strings.TrimSpace(x)
			continue
		case strings.HasPrefix(s, `\multirowcell`):
			n, e1, _ := group(s, len(`\multirowcell`))
			x, _, _ := group(s, skipOpt(s, e1))
			fmt.Sscan(n, &c.rowsDown)
			s = strings.TrimSpace(x)
			continue
		case strings.HasPrefix(s, `\multirow`):
			n, e1, _ := group(s, skipOpt(s, len(`\multirow`)))
			_, e2, _ := group(s, skipOpt(s, e1))
			x, _, _ := group(s, skipOpt(s, e2))
			fmt.Sscan(n, &c.rowsDown)
			s = strings.TrimSpace(x)
			continue
		}
		break
	}
	c.text = clean(s, unknown)
	return c
}

// parseTabular turns a tabular body into a grid. Header = content rows before the first rule
// that follows a content row; multi-row headers merge per column ("BLEU EN-DE").
func parseTabular(body string, unknown map[string]int) Grid {
	// a tabular nested in a cell becomes one line of text
	for {
		s, bs, be, e, ok := envSpan(body, "tabular", 0)
		if !ok {
			break
		}
		_, afterSpec, _ := group(body, bs)
		body = body[:s] + strings.ReplaceAll(body[afterSpec:be], `\\`, " ") + body[e:]
	}
	if _, end, ok := group(body, 0); ok { // column spec
		body = body[end:]
	}
	type row struct {
		cells    []cell
		ruleOver bool
	}
	var rows []row
	pendingRule := false
	for _, r := range splitTop(strings.ReplaceAll(body, `\tabularnewline`, `\\`), `\\`) {
		r = strings.TrimLeft(r, " \n")
		if strings.HasPrefix(r, "[") { // \\[2pt]
			if k := strings.IndexByte(r, ']'); k >= 0 {
				r = r[k+1:]
			}
		}
		for {
			m := ruleCmd.FindStringSubmatchIndex(r)
			if m == nil {
				break
			}
			kind := r[m[2]:m[3]]
			// a partial rule (\cmidrule, \cline) underlines a header group; it does not end the header
			pendingRule = pendingRule || kind != "cmidrule" && kind != "cline"
			rest := skipOpt(r, m[1])
			n := map[string]int{"cmidrule": 1, "cline": 1, "specialrule": 3}[kind]
			for a := 0; a < n; a++ {
				_, end, ok := group(r, rest)
				if !ok {
					break
				}
				rest = end
			}
			r = r[rest:]
		}
		if strings.TrimSpace(r) == "" {
			continue
		}
		var cells []cell
		for _, raw := range splitTop(r, "&") {
			cells = append(cells, parseCell(raw, unknown))
		}
		rows = append(rows, row{cells, pendingRule})
		pendingRule = false
	}

	// expand spans into a rectangular grid, carrying \multirow values down
	width := 0
	for _, r := range rows {
		w := 0
		for _, c := range r.cells {
			w += c.span
		}
		width = max(width, w)
	}
	headerRows := 1
	for i := 1; i < len(rows); i++ {
		if rows[i].ruleOver {
			headerRows = i
			break
		}
	}
	headerRows = min(headerRows, len(rows))

	grid := make([][]string, len(rows))
	down := make([]struct {
		text string
		left int
	}, width)
	for i, r := range rows {
		if i == headerRows {
			clear(down) // a header \multirow never spills into the data rows
		}
		line := make([]string, width)
		nonEmpty := 0
		for _, c := range r.cells {
			if c.text != "" {
				nonEmpty++
			}
		}
		col := 0
		for _, c := range r.cells {
			// a data row whose only text is one spanning cell is a group label
			// ("Temporal Transformer", "#FLOPs matched to GPT-3 Large"): write it once
			fill := c.span
			if i >= headerRows && nonEmpty == 1 && c.span > 1 {
				fill = 1
			}
			for k := 0; k < fill && col+k < width; k++ {
				line[col+k] = c.text
				if c.rowsDown > 1 && c.text != "" {
					down[col+k].text, down[col+k].left = c.text, c.rowsDown-1
				}
				// \multirow{-n} sits on the last row it covers: fill upward
				for up := 1; c.rowsDown < 0 && up < -c.rowsDown && i-up >= 0; up++ {
					if (i-up >= headerRows) == (i >= headerRows) && grid[i-up][col+k] == "" {
						grid[i-up][col+k] = c.text
					}
				}
			}
			col += c.span
		}
		for k := range line {
			if line[k] == "" && down[k].left > 0 {
				line[k] = down[k].text
				down[k].left--
			}
		}
		grid[i] = line
	}

	header := make([]string, width)
	for k := 0; k < width; k++ {
		var parts []string
		for i := 0; i < headerRows; i++ {
			if t := grid[i][k]; t != "" && (len(parts) == 0 || parts[len(parts)-1] != t) {
				parts = append(parts, t)
			}
		}
		header[k] = strings.Join(parts, " ")
	}
	data := grid[headerRows:]

	// drop spacer columns that are empty everywhere
	var keep []int
	for k := 0; k < width; k++ {
		used := header[k] != ""
		for _, r := range data {
			used = used || r[k] != ""
		}
		if used {
			keep = append(keep, k)
		}
	}
	g := Grid{Header: pick(header, keep)}
	for _, r := range data {
		g.Rows = append(g.Rows, pick(r, keep))
	}
	return g
}

func pick(row []string, idx []int) []string {
	out := make([]string, len(idx))
	for i, k := range idx {
		out[i] = row[k]
	}
	return out
}

var (
	captionRe = regexp.MustCompile(`\\caption(?:of\{table\})?\s*(?:\[[^\]]*\])?`)
	labelRe   = regexp.MustCompile(`\\label\{([^}]*)\}`)
)

// ParseTables finds every table float (and \captionof{table} minipages) in the source.
func ParseTables(paper, src string) ([]Table, []string) {
	lines := strings.Split(src, "\n")
	for i, l := range lines {
		lines[i] = stripComment(l)
	}
	src = strings.Join(lines, "\n")
	src = expandMacros(src, parseMacros(src))
	unknown := map[string]int{}
	var blocks []string
	for _, env := range []string{"table", "table*"} {
		for pos := 0; ; {
			_, bs, be, e, ok := envSpan(src, env, pos)
			if !ok {
				break
			}
			blocks = append(blocks, src[bs:be])
			pos = e
		}
	}
	for pos := 0; ; { // \captionof{table} inside a minipage of a figure
		_, bs, be, e, ok := envSpan(src, "minipage", pos)
		if !ok {
			break
		}
		if b := src[bs:be]; strings.Contains(b, `\captionof{table}`) {
			blocks = append(blocks, b)
		}
		pos = e
	}

	var tables []Table
	var warns []string
	for _, b := range blocks {
		t := Table{Paper: paper}
		if m := captionRe.FindStringIndex(b); m != nil {
			if c, _, ok := group(b, m[1]); ok {
				t.Caption = clean(labelRe.ReplaceAllString(c, ""), unknown)
			}
		}
		if m := labelRe.FindStringSubmatch(b); m != nil {
			t.Label = m[1]
		}
		for pos := 0; ; {
			_, bs, be, e, ok := envSpan(b, "tabular", pos)
			if !ok {
				break
			}
			if g := parseTabular(b[bs:be], unknown); len(g.Rows) > 0 {
				t.Parts = append(t.Parts, g)
			}
			pos = e
		}
		if len(t.Parts) == 0 {
			continue // a table float holding only an image
		}
		if t.Caption == "" {
			warns = append(warns, fmt.Sprintf("table %q has no caption", t.Label))
		}
		tables = append(tables, t)
	}
	if len(unknown) > 0 {
		names := make([]string, 0, len(unknown))
		for n := range unknown {
			names = append(names, n)
		}
		sort.Strings(names)
		warns = append(warns, "LaTeX commands guessed at (argument kept, command dropped): \\"+strings.Join(names, ", \\"))
	}
	return tables, warns
}

var (
	pdfCaptionRe = regexp.MustCompile(`Table ([0-9]+|[A-Z][0-9]*)[:.] `)
	nonAlnum     = regexp.MustCompile(`[^a-z0-9]+`)
)

func norm(s string) string { return nonAlnum.ReplaceAllString(strings.ToLower(s), "") }

// LocateTables sets each table's number and page from the PDF caption that best matches its
// LaTeX caption, and its section from where that caption sits in the text.
func LocateTables(tables []Table, pages []string, segs []Segment) []string {
	type pdfCap struct {
		number, text string
		page         int
	}
	var caps []pdfCap
	for p, pg := range pages {
		flat := strings.Join(strings.Fields(ligatures.Replace(pg)), " ")
		for _, m := range pdfCaptionRe.FindAllStringSubmatchIndex(flat, -1) {
			caps = append(caps, pdfCap{flat[m[2]:m[3]], norm(flat[m[1]:min(m[1]+300, len(flat))]), p + 1})
		}
	}
	var warns []string
	used := map[string]bool{}
	for i := range tables {
		want := norm(tables[i].Caption)
		best, bestLen := -1, 0
		for k, c := range caps {
			n := 0
			for n < len(want) && n < len(c.text) && want[n] == c.text[n] {
				n++
			}
			if n > bestLen && !used[c.number] {
				best, bestLen = k, n
			}
		}
		if best < 0 || bestLen < 20 {
			warns = append(warns, fmt.Sprintf("table %q: caption not found in PDF", tables[i].Label))
			continue
		}
		c := caps[best]
		used[c.number] = true
		tables[i].Number, tables[i].Page = c.number, c.page
		marker := "Table " + c.number + ":"
		for _, s := range segs {
			for _, sen := range s.Sentences {
				if sen.Page == c.page && strings.Contains(sen.Text, marker) {
					tables[i].SectionNumber, tables[i].Section, tables[i].SectionPath = s.Number, s.Title, s.Path
				}
			}
		}
	}
	sort.SliceStable(tables, func(a, b int) bool { return tables[a].Page < tables[b].Page })
	return warns
}

func markdown(g Grid, rows [][]string) string {
	var b strings.Builder
	line := func(cells []string) {
		b.WriteString("| " + strings.Join(cells, " | ") + " |\n")
	}
	line(g.Header)
	sep := make([]string, len(g.Header))
	for i := range sep {
		sep[i] = "---"
	}
	line(sep)
	for _, r := range rows {
		line(r)
	}
	return b.String()
}

// TableChunks renders each table as caption + Markdown, one chunk per table when it fits,
// otherwise split by rows with the caption and header repeated.
func TableChunks(tables []Table, maxTokens int, count CountFunc, startOrder int) []Chunk {
	var chunks []Chunk
	for _, t := range tables {
		if t.Number == "" {
			continue // not located in the PDF: cannot cite a page
		}
		head := fmt.Sprintf("Table %s (page %d): %s\n\n", t.Number, t.Page, t.Caption)
		var texts []string
		var split func(g Grid, rows [][]string)
		split = func(g Grid, rows [][]string) {
			text := head + markdown(g, rows)
			if count([]string{text})[0] <= maxTokens || len(rows) <= 1 {
				texts = append(texts, text)
				return
			}
			split(g, rows[:len(rows)/2])
			split(g, rows[len(rows)/2:])
		}
		for _, g := range t.Parts {
			split(g, g.Rows)
		}
		for _, text := range texts {
			chunks = append(chunks, Chunk{
				Kind: "table", Table: "Table " + t.Number, Paper: t.Paper,
				SectionNumber: t.SectionNumber, Section: t.Section, SectionPath: t.SectionPath,
				PageStart: t.Page, PageEnd: t.Page, Text: text,
			})
		}
	}
	if len(chunks) == 0 {
		return nil
	}
	all := make([]string, len(chunks))
	for i := range chunks {
		all[i] = chunks[i].Text
	}
	for i, n := range count(all) {
		chunks[i].ID = fmt.Sprintf("%s#t%03d", chunks[i].Paper, i)
		chunks[i].Order = startOrder + i
		chunks[i].Tokens = n
	}
	return chunks
}
