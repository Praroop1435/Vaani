// Package textnorm removes formatting differences between the PDFs, the LaTeX source, the model's
// replies and people's questions: Unicode forms, spaces, hyphens, punctuation and magnitude units.
// The grader (internal/eval) and the BM25 index (internal/retrieve) share it, so a fact written
// "12.5Hz", "12.5 Hz", "7 billion" or "7B" compares the same way everywhere.
package textnorm

import (
	"regexp"
	"strings"

	"golang.org/x/text/unicode/norm"
)

var (
	// NFKC folds ligatures (ﬁ), math italics (𝑏𝑡 → bt) and narrow/no-break spaces; these are left.
	space  = regexp.MustCompile(`[\p{Zs}\x{200B}\s]+`)
	hyphen = strings.NewReplacer("‐", "-", "‑", "-", "‒", "-", "–", "-", "—", "-", "−", "-")
	// "7 billion" / "7B" / "7-billion" → 7B; same for million (M) and trillion (T).
	magnitude = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*-?\s*((?i:billion|trillion|million)|B|T|M)\b`)
	unitOf    = map[string]string{"billion": "B", "trillion": "T", "million": "M", "b": "B", "t": "T", "m": "M"}
	// a number glued to its unit ("12.5Hz", "1.1kbps", "80ms") gets a space, like "12.5 Hz"
	glued = regexp.MustCompile(`(\d)((?i:hz|khz|kbps|ms|gb|tflops)\b)`)
	// possessive "'s" ("H-Net's") would otherwise become a stray token "s"
	possessive = regexp.MustCompile(`(\pL)['\x{2019}]s\b`)
)

// Normalize returns s with Unicode folded (NFKC), whitespace collapsed, hyphen look-alikes made
// ASCII, magnitude words written as B/M/T and units separated from their numbers.
func Normalize(s string) string {
	s = hyphen.Replace(norm.NFKC.String(s))
	s = magnitude.ReplaceAllStringFunc(s, func(m string) string {
		p := magnitude.FindStringSubmatch(m)
		return p[1] + unitOf[strings.ToLower(p[2])]
	})
	s = glued.ReplaceAllString(s, "$1 $2")
	s = possessive.ReplaceAllString(s, "$1")
	return strings.TrimSpace(space.ReplaceAllString(s, " "))
}

var (
	token = regexp.MustCompile(`\d+(?:\.\d+)?|\pL+`) // punctuation separates tokens; decimals stay whole
	stop  = map[string]bool{}
)

func init() {
	for _, w := range strings.Fields(`a an and are as at be by does did do for from has have how in is it its
		of on or that the their this to was were what when where which who why with use used uses using
		than then there these they into we our can not no`) {
		stop[w] = true
	}
}

// Tokens is the BM25 view of s: normalized, lower-cased, split on punctuation, stopwords removed.
// Letters and digits split apart ("7B" → "7", "b"; "P100" → "p", "100"), so "7B" and "7 B" match.
func Tokens(s string) []string {
	var out []string
	for _, t := range token.FindAllString(strings.ToLower(Normalize(s)), -1) {
		if !stop[t] {
			out = append(out, t)
		}
	}
	return out
}
