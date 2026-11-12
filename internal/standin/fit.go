package standin

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"slices"

	"github.com/Praroop1435/Vaani/internal/pipeline"
)

// Call is one real model call as the pipeline traced it.
type Call struct {
	Step                         string
	TokensIn, TokensOut, Content int // Content: reply tokens shown (chars/4); the rest is hidden reasoning
	FirstMs                      int64
	WorkMs                       float64
}

// Calls reads the successful model calls from smoke/calibration trace files (one graded trace per line).
func Calls(paths ...string) ([]Call, error) {
	var out []Call
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 16<<20)
		for sc.Scan() {
			var line struct {
				Trace pipeline.Trace `json:"trace"`
			}
			if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
				f.Close()
				return nil, err
			}
			for _, s := range line.Trace.Steps {
				if s.TokensIn == 0 || s.FirstMs <= 0 || s.Retries > 0 || s.Note != "" {
					continue
				}
				out = append(out, Call{Step: s.Name, TokensIn: s.TokensIn, TokensOut: s.TokensOut,
					Content: len(s.Raw)/4 + 1, FirstMs: s.FirstMs, WorkMs: s.WorkMs})
			}
		}
		f.Close()
		if err := sc.Err(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Fit sets c's latency fields from real calls: first token = a + b·prompt (least squares) with a
// lognormal spread from the residuals; streaming speed and hidden reasoning tokens from medians.
func Fit(c Config, calls []Call) Config {
	n := float64(len(calls))
	if n < 3 {
		return c
	}
	var sx, sy, sxx, sxy float64
	for _, k := range calls {
		x, y := float64(k.TokensIn)/1000, float64(k.FirstMs)
		sx, sy, sxx, sxy = sx+x, sy+y, sxx+x*x, sxy+x*y
	}
	b := (n*sxy - sx*sy) / (n*sxx - sx*sx)
	a := (sy - b*sx) / n
	if b < 0 || a < 50 { // too few points to trust a slope: use the median, no slope
		ys := make([]float64, 0, len(calls))
		for _, k := range calls {
			ys = append(ys, float64(k.FirstMs))
		}
		a, b = median(ys), 0
	}
	var logs, tokMs []float64
	var reason []float64
	for _, k := range calls {
		pred := a + b*float64(k.TokensIn)/1000
		logs = append(logs, math.Log(float64(k.FirstMs)/pred))
		if k.Content > 1 {
			tokMs = append(tokMs, (k.WorkMs-float64(k.FirstMs))/float64(k.Content))
		}
		reason = append(reason, float64(max(k.TokensOut-k.Content, 0)))
	}
	mean := 0.0
	for _, l := range logs {
		mean += l / n
	}
	v := 0.0
	for _, l := range logs {
		v += (l - mean) * (l - mean) / (n - 1)
	}
	// re-centre so the lognormal factor has median 1: the fitted line, not the noise, sets the median
	c.FirstBaseMs, c.FirstPerKTokMs, c.FirstSigma = a*math.Exp(mean), b*math.Exp(mean), math.Sqrt(v)
	c.ContentTokMs = math.Max(median(tokMs), 0)
	c.ReasoningTok = int(median(reason))
	return c
}

func median(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	s := slices.Clone(x)
	slices.Sort(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}
