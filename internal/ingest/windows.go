package ingest

import "strings"

// Window is a few sentences cut from one chunk. Retrieval setting C searches windows, then hands
// the model the whole parent chunk ("small-to-big"): a one-line answer like "we employ h = 8
// parallel attention layers" gets its own vector instead of being drowned by the formulas around it.
type Window struct {
	Parent int    `json:"parent"` // index into the chunk list it was cut from
	Text   string `json:"text"`
}

// Windows cuts every text chunk into runs of `size` sentences, moving `stride` sentences at a time.
// A table chunk stays one window: its caption and header already say what it holds.
func Windows(chunks []Chunk, size, stride int) []Window {
	var out []Window
	for i, c := range chunks {
		ss := splitSentences(c.Text)
		if c.Kind == "table" || len(ss) <= size {
			out = append(out, Window{i, c.Text})
			continue
		}
		for start := 0; start < len(ss); start += stride {
			end := min(start+size, len(ss))
			out = append(out, Window{i, strings.Join(ss[start:end], " ")})
			if end == len(ss) {
				break
			}
		}
	}
	return out
}
