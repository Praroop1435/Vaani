# The Relay

A question-answering service over three arXiv papers (*Attention Is All You Need*, *Moshi* and *H-Net (Dynamic Chunking)*) that runs on Groq's free tier. It cites every sentence, refuses when the papers don't say, and stays inside the provider's real rate limits under load.

**Report (2 pages):** [`Docs/report/report.pdf`](Docs/report/report.pdf)

| Result | Value | Source |
|---|---|---|
| Answerable questions correct and cited (real Groq, `qwen/qwen3.8-27b`) | **13 of 15** | Table 3 |
| Wrong answers | **0** | Table 3 |
| Unanswerable questions correctly refused | **3 of 3** | Table 3 |
| Right passage retrieved (benchmark) | 16 of 18 (was 12 with plain search) | `results/retrieval_fix_benchmark.csv` |
| Load scenarios, all on the stand-in | 5 | Tables 2a, 2b |

Every number in the report is recomputed from one file, [`results/results.csv`](results/results.csv), by `go run ./cmd/tables`.

---

## For reviewers: where to look

| What | Where |
|---|---|
| The report | `Docs/report/report.pdf` (source: `report.tex`) |
| One row per request | `results/results.csv` |
| Tables 1, 2a, 2b, 3 recomputed from that file | `results/tables.md` (`go run ./cmd/tables`) |
| Raw traces behind every row | `results/quality/`, `results/load/`, `results/calibration/`, `results/calibration_qwen/` |
| Real service vs stand-in: measured vs guessed | `Docs/STANDIN.md` |
| Known limits and decisions | `Docs/ISSUES_AND_PLAN.md` |
| Full, unedited conversation with the assistant | `conversation/` |

## How it answers a question

```
question
   │
   ├─ 1. retrieve ───── BGE + BM25 over 300-token chunks and every LaTeX table
   ├─ 2. decide ─────── best match < 0.50  →  "the papers don't cover this" (no model call)
   ├─ 3. understand ─── model splits the question into parts + a search query
   ├─ 4. re-rank ────── paper scope, top-20 from each search, local cross-encoder keeps 6
   ├─ 5. draft ──────── streamed; a sentence is shown only if its cited passage
   │                    holds every number in it
   └─ 6. fill gaps ──── a missing part gets one targeted search and redraft,
                        if the limits and the 20 s deadline allow
```

**What shaped the design:** on Groq's free tier the scarce resource is tokens per minute. Each call is charged prompt + maximum reply the moment it's admitted, against an 8,000-token bucket refilling at 133.3 tokens/s. A limiter in front of Groq (`internal/governor`) models that bucket exactly. It admits a call once it fits, and refuses one that would miss the deadline instead of making the person wait for nothing.

## Run it

Needs macOS, Go 1.27 and Homebrew. One-time setup (downloads about 1.5 GB of local models):

```bash
brew install onnxruntime
```

```bash
go run -tags ORT ./cmd/getmodel
```

Put your own Groq key in `.env`, which is gitignored:

```bash
echo 'GROQ_API_KEY=your_key_here' > .env
```

Ask a question (1–3 real Groq calls):

```bash
go run -tags ORT ./cmd/ask "How many attention heads does the base Transformer use?"
```

The search index is already in `data/index/`. `go run -tags ORT ./cmd/ingest` rebuilds it from arXiv, and `-meta-only` rebuilds just the paper/section vectors the re-ranking setting uses.

## Reproduce the results

| Command | What it does | Groq cost |
|---|---|---|
| `go run ./cmd/tables` | Tables 1, 2a, 2b, 3 from `results.csv` | none |
| `go run -tags ORT ./cmd/load` | the five load scenarios on the stand-in (~35 min) | none |
| `go run -tags ORT ./cmd/load -retrieval` | retrieval settings A, H10, H15, R on the stand-in | none |
| `go run -tags ORT ./cmd/evalretrieval -fix` | retrieval ablation, evidence found per setting | none |
| `go run -tags ORT ./cmd/smoke -set frozen -pace 30s -max-tokens 75000` | the 18 scored questions on real Groq | ~55K tokens |
| `go test -tags ORT ./test -count=1` | the test suite (~20 s) | none |

`smoke` and `load` append to `results/results.csv`. Run `go run ./cmd/tables` again afterwards. The answer cache is off for every measured run.

### What's in `results.csv`

| `scenario` | What it is | Feeds |
|---|---|---|
| `calibration`, `calibration_probe` | 10 rows from the calls that measured the limits, including the forced 429 | Table 1 |
| `quality` | the 18 scored questions on `gpt-oss-120b` (settings A and R) | Table 3 |
| `quality_qwen` | the 18 scored questions on `qwen3.8-27b`, setting R (the submission) | Table 3 |
| `retrieval_standin` | settings A, H10, H15, R with a perfect-reader model | Table 3 |
| `steady`, `burst`, `overload`, `rpm_halved`, `daily_exhausted` | load on the stand-in | Tables 2a, 2b |

Rows from superseded runs (before the limiter fix, before the seed fix, earlier settings) are kept in `results/load/archive_*/`. No table reads them.

## Code map

| Package | Role |
|---|---|
| `internal/pipeline` | the service: retrieve, decide, understand, draft, check citations, fill gaps |
| `internal/retrieve` | BGE + BM25 search, paper scope, cross-encoder re-rank |
| `internal/embed` | the two local ONNX models (embedder, re-ranker) |
| `internal/llm` | streaming Groq client, retries, rate-limit headers |
| `internal/governor` | token-bucket limiter for the per-minute and per-day limits |
| `internal/standin` | the fake model service for load: slow, failing, rate-limited, streaming |
| `internal/ingest`, `internal/textnorm` | PDFs and LaTeX tables into chunks; text normalisation |
| `internal/eval`, `internal/fakellm`, `internal/results` | questions and grading, perfect-reader model, CSV rows |
| `cmd/*` | one command per job (see the table above) |

## Honest limits

- The re-ranking setting and the switch to qwen were chosen **after** the first scored run. The setting was tuned on a separate 18-question dev set (`data/eval/dev.json`), but the scored questions were no longer unseen.
- The limits were probed on `gpt-oss-120b`. Qwen's headers show the same limits, but its 429 behaviour wasn't probed.
- Load runs use the stand-in, not Groq: loading the free tier would break its terms. The stand-in's slowdown under load is a guess.
- One question (q17, H-Net's 100B-token training set) is never retrieved by any setting.

The API key is never in the repository: it lives only in `.env`, which is gitignored.
