# The Relay: report

The 2-page report is **[`Docs/report/report.pdf`](report/report.pdf)**, built from [`Docs/report/report.tex`](report/report.tex):

```bash
cd Docs/report && tectonic report.tex
```

Every number in it is recomputed from `results/results.csv` by `go run ./cmd/tables` (output: `results/tables.md`).
