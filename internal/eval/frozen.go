package eval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"slices"
)

// Frozen is the measured question set: data/eval/frozen.json pins it to one exact benchmark.json.
type Frozen struct {
	FrozenAt     string   `json:"frozen_at"`
	BenchSHA256  string   `json:"benchmark_sha256"`
	Quality      []string `json:"quality"`
	Unanswerable []string `json:"unanswerable"`
	Excluded     []struct {
		ID     string `json:"id"`
		Reason string `json:"reason"`
	} `json:"excluded"`
}

// LoadFrozen returns the frozen questions (quality first, then unanswerable). It fails if
// benchmark.json changed since the freeze, so a measured run can never use an edited answer key.
func LoadFrozen(benchPath, frozenPath string) ([]Question, *Frozen, error) {
	raw, err := os.ReadFile(benchPath)
	if err != nil {
		return nil, nil, err
	}
	fz := &Frozen{}
	fj, err := os.ReadFile(frozenPath)
	if err != nil {
		return nil, nil, err
	}
	if err := json.Unmarshal(fj, fz); err != nil {
		return nil, nil, err
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != fz.BenchSHA256 {
		return nil, nil, fmt.Errorf("eval: %s changed since the freeze (sha256 %s, frozen %s)", benchPath, got[:12], fz.BenchSHA256[:12])
	}
	b, err := Load(benchPath)
	if err != nil {
		return nil, nil, err
	}
	var qs []Question
	for _, id := range append(slices.Clone(fz.Quality), fz.Unanswerable...) {
		i := slices.IndexFunc(b.Questions, func(q Question) bool { return q.ID == id })
		if i < 0 {
			return nil, nil, fmt.Errorf("eval: frozen id %s not in benchmark", id)
		}
		qs = append(qs, b.Questions[i])
	}
	return qs, fz, nil
}
