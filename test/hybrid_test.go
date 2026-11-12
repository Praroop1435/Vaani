package test

import (
	"slices"
	"testing"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/retrieve"
	"github.com/Praroop1435/Vaani/internal/textnorm"
)

func TestTokensNormaliseUnicodePunctuationAndUnits(t *testing.T) {
	same := [][2]string{
		{"Mimi runs at 12.5Hz", "Mimi runs at 12.5 Hz"},
		{"a 7 billion-parameter model", "a 7B-parameter model"},
		{"pretrained on 2.1 trillion tokens", "pretrained on 2.1T tokens"},
		{"WMT 2014 English‑to‑German", "WMT 2014 English-to-German"},
		{"ﬁne-tuning", "fine-tuning"},            // ligature
		{"𝑏𝑡 = 1 {𝑝𝑡 ≥0.5}", "bt = 1 {pt ≥0.5}"}, // math italics from pdftotext
		{"bitrate of 1.1kbps", "bitrate of 1.1 kbps"},
	}
	for _, p := range same {
		if a, b := textnorm.Tokens(p[0]), textnorm.Tokens(p[1]); !slices.Equal(a, b) {
			t.Errorf("%q -> %q but %q -> %q", p[0], a, p[1], b)
		}
	}
	if got := textnorm.Tokens("What GPUs were the H100-based FineWeb-Edu models trained on?"); !slices.Equal(got,
		[]string{"gpus", "h", "100", "based", "fineweb", "edu", "models", "trained"}) {
		t.Errorf("tokens %q (stopwords out, punctuation splits, letters/digits split)", got)
	}
	if got := textnorm.Tokens("How much data are H-Net’s main models trained on?"); slices.Contains(got, "s") {
		t.Errorf("possessive left a stray token: %q", got)
	}
	if got := textnorm.Tokens("P_drop = 0.1 and d_model 512"); !slices.Contains(got, "0.1") || !slices.Contains(got, "512") {
		t.Errorf("decimals and numbers must stay whole: %q", got)
	}
}

func TestBM25RanksRareExactTermFirst(t *testing.T) {
	b := retrieve.NewBM25([]string{
		"Training ran on many GPUs for the language model.",
		"All models are trained on H100 GPUs, using FSDP.",
		"The GPUs were busy; GPUs, GPUs, GPUs everywhere.",
		"Mimi is the audio codec of Moshi.",
	})
	if got := b.Search("Which H100 GPUs trained the models?", 4); len(got) == 0 || got[0].Doc() != 1 {
		t.Errorf("the only document with the rare term H100 must rank first: %+v", got)
	}
	if got := b.Search("sourdough bread", 4); len(got) != 0 {
		t.Errorf("no shared terms must return nothing, got %+v", got)
	}
}

func TestRRFPrefersDocumentsInBothLists(t *testing.T) {
	got := retrieve.RRF([]int{1, 2, 3}, []int{3, 4})
	// 3: 1/63 + 1/61; 1: 1/61; 4: 1/62; 2: 1/62 → 3, 1, then 2 before 4 (2 seen first)
	if want := []int{3, 1, 2, 4}; !slices.Equal(got, want) {
		t.Errorf("RRF = %v, want %v", got, want)
	}
	if got := retrieve.RRF([]int{5, 6}); !slices.Equal(got, []int{5, 6}) {
		t.Errorf("one list must keep its order: %v", got)
	}
}

func TestHybridKeepsCosineAndNeedsQuestion(t *testing.T) {
	unit := func(x, y float32) []float32 {
		v := make([]float32, embed.Dim)
		v[0], v[1] = x, y
		return v
	}
	ix := &retrieve.Index{
		Chunks: []ingest.Chunk{{ID: "sem", Text: "attention heads"}, {ID: "lex", Text: "All models are trained on H100 GPUs"}, {ID: "none", Text: "Mimi codec"}},
		Vecs:   [][]float32{unit(1, 0), unit(0, 1), unit(0.6, 0.8)},
	}
	s := retrieve.Setting{Name: "h", K: 2, Hybrid: true, Candidates: 2}
	hits := ix.SearchHybrid(unit(1, 0), "H100 GPUs", s)
	var ids []string
	for _, h := range hits {
		ids = append(ids, h.Chunk.ID)
	}
	if !slices.Contains(ids, "lex") || !slices.Contains(ids, "sem") {
		t.Errorf("hybrid must surface the lexical-only and the semantic-only match: %v", ids)
	}
	for _, h := range hits {
		if want := embed.Cosine(unit(1, 0), ix.Vecs[slices.IndexFunc(ix.Chunks, func(c ingest.Chunk) bool { return c.ID == h.Chunk.ID })]); h.Score != want {
			t.Errorf("%s: score %v must stay the BGE cosine %v (the off-topic threshold depends on it)", h.Chunk.ID, h.Score, want)
		}
	}
	r := &retrieve.Retriever{Indexes: map[int]*retrieve.Index{0: ix}}
	if _, err := r.Search(unit(1, 0), s); err == nil {
		t.Error("Search without the question text must refuse a hybrid setting")
	}
}
