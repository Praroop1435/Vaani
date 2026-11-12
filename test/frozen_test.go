package test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Praroop1435/Vaani/internal/eval"
)

func TestFrozenSetIsIntact(t *testing.T) {
	qs, fz, err := eval.LoadFrozen("../data/eval/benchmark.json", "../data/eval/frozen.json")
	if err != nil {
		t.Fatal(err) // benchmark.json edited after the freeze: re-freeze deliberately or revert
	}
	if len(fz.Quality) != 15 || len(fz.Unanswerable) != 3 || len(qs) != 18 {
		t.Fatalf("want 15 quality + 3 unanswerable, got %d + %d", len(fz.Quality), len(fz.Unanswerable))
	}
	seen := map[string]bool{}
	for i, q := range qs {
		if seen[q.ID] {
			t.Errorf("%s twice", q.ID)
		}
		seen[q.ID] = true
		if q.Answerable != (i < 15) {
			t.Errorf("%s answerable=%v in the wrong list", q.ID, q.Answerable)
		}
	}
	for _, x := range fz.Excluded {
		if seen[x.ID] || x.Reason == "" {
			t.Errorf("excluded %s must be absent and have a reason", x.ID)
		}
	}
	for _, id := range []string{"q01", "q02", "q12", "q17"} { // A's known failures stay in
		if !seen[id] {
			t.Errorf("%s (a known failure) was dropped", id)
		}
	}
}

func TestFrozenRejectsAnEditedBenchmark(t *testing.T) {
	dir := t.TempDir()
	b, _ := os.ReadFile("../data/eval/benchmark.json")
	os.WriteFile(filepath.Join(dir, "b.json"), append(b, ' '), 0o644) // one byte changed
	if _, _, err := eval.LoadFrozen(filepath.Join(dir, "b.json"), "../data/eval/frozen.json"); err == nil {
		t.Error("an edited benchmark was accepted")
	}
}
