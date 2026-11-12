package test

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/Praroop1435/Vaani/internal/pipeline"
	"github.com/Praroop1435/Vaani/internal/results"
)

func TestResultsCSVHasTheBriefsExactColumns(t *testing.T) {
	want := []string{"request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status",
		"retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"}
	if !slices.Equal(results.Columns, want) {
		t.Fatalf("columns %q", results.Columns)
	}
	path := filepath.Join(t.TempDir(), "results.csv")
	start := time.Date(2026, 11, 11, 20, 0, 0, 0, time.UTC)
	tr := pipeline.Trace{RequestID: "q05-1", Start: start, FirstTokenMs: 1569, TotalMs: 1575, Status: "ok", Retries: 1, TokensIn: 1489, TokensOut: 122}
	cut := pipeline.Trace{RequestID: "q14-1", Start: start, FirstTokenMs: -1, TotalMs: 900, Status: "rate_limited"}

	w, err := results.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(results.FromTrace(tr, "quality", "real", results.SettingID("A: 300 tok, k=4"), "correct")); err != nil {
		t.Fatal(err)
	}
	// a passages-only reply is not an answer: no grade even if the caller passes one
	if err := w.Write(results.FromTrace(cut, "burst", "stand-in", "A", "wrong")); err != nil {
		t.Fatal(err)
	}
	w.Close()
	w2, err := results.Open(path) // re-open appends without a second header
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []results.Row{
		{RequestID: "x", Scenario: "s", Source: "groq", Status: "ok", Setting: "A"},
		{RequestID: "x", Scenario: "s", Source: "real", Status: "slow", Setting: "A"},
		{RequestID: "x", Scenario: "s", Source: "real", Status: "ok", Setting: "A", Grade: "partial"},
	} {
		if w2.Write(bad) == nil {
			t.Errorf("invalid row accepted: %+v", bad)
		}
	}
	w2.Close()

	f, _ := os.Open(path)
	recs, err := csv.NewReader(f).ReadAll()
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	want2 := [][]string{
		want,
		{"q05-1", "quality", "real", "2026-11-11T20:00:00Z", "1569", "1575", "ok", "1", "1489", "122", "A", "correct"},
		{"q14-1", "burst", "stand-in", "2026-11-11T20:00:00Z", "-1", "900", "rate_limited", "0", "0", "0", "A", ""},
	}
	if len(recs) != len(want2) {
		t.Fatalf("rows %q", recs)
	}
	for i := range want2 {
		if !slices.Equal(recs[i], want2[i]) {
			t.Errorf("row %d = %q, want %q", i, recs[i], want2[i])
		}
	}
}

func TestResultsWriterIsSafeForLoadTests(t *testing.T) {
	path := filepath.Join(t.TempDir(), "results.csv")
	w, _ := results.Open(path)
	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			w.Write(results.Row{RequestID: "r", Scenario: "burst", Source: "stand-in", Status: "ok", Setting: "A", Start: time.Now()})
		}(i)
	}
	wg.Wait()
	w.Close()
	f, _ := os.Open(path)
	recs, err := csv.NewReader(f).ReadAll()
	f.Close()
	if err != nil || len(recs) != 201 {
		t.Errorf("%d records, err %v (want header + 200 intact rows)", len(recs), err)
	}
	os.WriteFile(path, []byte("a,b,c\n"), 0o644)
	if _, err := results.Open(path); err == nil {
		t.Error("a file with another header must be refused, not mixed")
	}
}
