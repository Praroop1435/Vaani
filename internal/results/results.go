// Package results writes results.csv: one row per request, exactly the columns the brief asks for.
// Every measured run (real Groq or the stand-in) appends to the same file; cmd/tables recomputes
// every table number from it.
package results

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Praroop1435/Vaani/internal/pipeline"
)

// Columns is the brief's header, in order.
var Columns = []string{"request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
	"status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"}

var (
	sources  = map[string]bool{"real": true, "stand-in": true}
	statuses = map[string]bool{"ok": true, "rate_limited": true, "timeout": true, "error": true}
	grades   = map[string]bool{"correct": true, "wrong": true, "refused_correctly": true, "refused_wrongly": true, "": true}
)

type Row struct {
	RequestID, Scenario, Source  string
	Start                        time.Time
	FirstTokenMs, TotalMs        int64 // FirstTokenMs -1: nothing was shown
	Status                       string
	Retries, TokensIn, TokensOut int
	Setting                      string // "A", "H10", "H15", ...
	Grade                        string // "" when the request produced no gradable answer (not ok)
}

// FromTrace maps a pipeline trace to a row. grade is ignored unless the request finished ok:
// a passages-only or cut-short reply is not an answer, so it gets no answer_grade.
func FromTrace(tr pipeline.Trace, scenario, source, setting, grade string) Row {
	if tr.Status != "ok" {
		grade = ""
	}
	return Row{RequestID: tr.RequestID, Scenario: scenario, Source: source, Start: tr.Start,
		FirstTokenMs: tr.FirstTokenMs, TotalMs: tr.TotalMs, Status: tr.Status, Retries: tr.Retries,
		TokensIn: tr.TokensIn, TokensOut: tr.TokensOut, Setting: setting, Grade: grade}
}

// SettingID shortens "A: 300 tok, k=4" to "A".
func SettingID(name string) string { return strings.TrimSpace(strings.SplitN(name, ":", 2)[0]) }

func (r Row) record() ([]string, error) {
	switch {
	case !sources[r.Source]:
		return nil, fmt.Errorf("results: source %q must be real or stand-in", r.Source)
	case !statuses[r.Status]:
		return nil, fmt.Errorf("results: status %q not one of ok, rate_limited, timeout, error", r.Status)
	case !grades[r.Grade]:
		return nil, fmt.Errorf("results: answer_grade %q not allowed", r.Grade)
	case r.RequestID == "" || r.Scenario == "" || r.Setting == "":
		return nil, fmt.Errorf("results: request_id, scenario and retrieval_setting are required")
	}
	return []string{r.RequestID, r.Scenario, r.Source, r.Start.UTC().Format(time.RFC3339Nano),
		strconv.FormatInt(r.FirstTokenMs, 10), strconv.FormatInt(r.TotalMs, 10), r.Status,
		strconv.Itoa(r.Retries), strconv.Itoa(r.TokensIn), strconv.Itoa(r.TokensOut), r.Setting, r.Grade}, nil
}

// Writer appends rows to one CSV file; safe for concurrent use (load tests).
type Writer struct {
	mu sync.Mutex
	f  *os.File
	w  *csv.Writer
}

// Open appends to path, writing the header only if the file is new or empty. An existing file
// with a different header is refused rather than mixed.
func Open(path string) (*Writer, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	st, _ := f.Stat()
	w := &Writer{f: f, w: csv.NewWriter(f)}
	if st.Size() == 0 {
		if err := w.w.Write(Columns); err != nil {
			return nil, err
		}
		w.w.Flush()
		return w, w.w.Error()
	}
	rf, err := os.Open(path)
	if err != nil {
		f.Close()
		return nil, err
	}
	head, err := csv.NewReader(rf).Read()
	rf.Close()
	if err != nil || strings.Join(head, ",") != strings.Join(Columns, ",") {
		f.Close()
		return nil, fmt.Errorf("results: %s has a different header %q", path, head)
	}
	return w, nil
}

// Write validates and appends one row, flushing so a crashed run keeps what it measured.
func (w *Writer) Write(r Row) error {
	rec, err := r.record()
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.w.Write(rec); err != nil {
		return err
	}
	w.w.Flush()
	return w.w.Error()
}

func (w *Writer) Close() error { return w.f.Close() }
