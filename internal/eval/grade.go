package eval

import (
	"regexp"
	"strings"

	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/textnorm"
)

var (
	// citation markers that differ between PDF text and LaTeX-derived text
	authorYear = regexp.MustCompile(`\s*\([A-Z][^()]{0,80}?\d{4}[a-z]?\)`) // "(Cieri et al., 2004)"
	numericRef = regexp.MustCompile(`\s*\[\d+(?:\s*,\s*\d+)*\]`)           // "[20]", "[3, 7]"
)

// NormText removes the formatting differences between the model, the PDF and the LaTeX source
// (Unicode, spaces, hyphens, magnitude words, glued units); see internal/textnorm. Used on answers
// before matching Accept patterns.
func NormText(s string) string { return textnorm.Normalize(s) }

// NormEvidence is NormText plus dropping citation markers and case, for matching a benchmark
// quote against a chunk (the Table 1 caption from LaTeX has no "(Cieri et al., 2004)").
func NormEvidence(s string) string {
	s = numericRef.ReplaceAllString(authorYear.ReplaceAllString(s, ""), "")
	return strings.ToLower(NormText(s))
}

// AnswerGrade is the answer_grade column: correct, wrong, refused_correctly, refused_wrongly.
type AnswerGrade struct {
	Grade  string
	CiteOK bool     // every evidence item is inside a chunk the answer cites, on the right page
	Notes  []string // why it is not correct / not cited
}

// GradeAnswer grades a final answer. cited are the chunks the answer actually cites.
func GradeAnswer(q Question, answer string, refused bool, cited []ingest.Chunk) AnswerGrade {
	g := AnswerGrade{CiteOK: true}
	switch {
	case !q.Answerable && refused:
		g.Grade = "refused_correctly"
		return g
	case !q.Answerable:
		g.Grade, g.CiteOK = "wrong", false
		g.Notes = append(g.Notes, "answered a question the papers do not answer")
		return g
	case refused:
		g.Grade = "refused_wrongly"
	default:
		g.Grade = "correct"
		a := NormText(answer)
		for _, p := range q.Accept {
			if !regexp.MustCompile(p).MatchString(a) {
				g.Grade = "wrong"
				g.Notes = append(g.Notes, "missing "+p)
			}
		}
	}
	for i, ev := range q.Evidence {
		ok := false
		for _, c := range cited {
			for _, s := range ev.AnyOf {
				ok = ok || s.Covers(c) && s.Page >= c.PageStart && s.Page <= c.PageEnd
			}
		}
		if !ok {
			g.CiteOK = false
			g.Notes = append(g.Notes, "evidence item "+string(rune('1'+i))+" not cited")
		}
	}
	return g
}
