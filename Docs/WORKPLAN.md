# Work Plan: who does what

**Language: Go.** Deadline: ~2026-11-13 (4 days from 2026-11-09).

## The split

**Your part (Praroop)** is what the reviewers say must be *you*: the account, the questions, the judgement calls, the grading, and the personal sections of the write-up. You will defend all of it in the live call.

**My part (Claude)** is the Go code, running the experiments, computing the tables, and drafting the technical sections for you to check.

## Tick rule

A box gets ticked **only after you confirm it in chat** ("verified", "ok", "looks good"). Built-but-not-confirmed stays unticked. `[You]` / `[Claude]` says who does the work.

## Checklist

### Phase 0: Setup
- [x] `[You]` Create Groq free account + API key, put it in `.env` as `GROQ_API_KEY=...`
- [x] `[You]` Pick the Groq model (Claude lists 2-3 options with their limits) → `openai/gpt-oss-120b`
- [ ] `[Claude]` `go mod init`, folder skeleton from ARCHITECTURE.md

### Phase 1: Paper library
- [ ] `[Claude]` Fetch the 3 PDFs at exact versions → `data/papers/` (downloaded 2026-11-09: 15 + 66 + 39 pages)
- [x] `[You]` Pick the local embedder → `bge-base-en-v1.5`, official BAAI weights from Hugging Face
- [x] `[Claude]` Embedder: weights saved in `models/`, run in-process via hugot + ONNX Runtime, tests in `test/`
- [x] `[Claude]` Ingest: `pdftotext` per page, chunk with paper / section / page, embed with BGE
- [x] `[You]` Spot-check 5 chunks against the PDFs (right text, right page?) → `Docs/SPOT_CHECK.md`
- [x] `[Claude]` Tables from arXiv LaTeX source → structured rows/cells (`data/index/tables.json`) + one chunk per table
- [x] `[You]` Check 3 tables against the PDFs → `Docs/TABLE_CHECK.md`
- [x] `[Claude]` Semantic top-k retrieve tool: settings A, B, C built and approved (architecture frozen 2026-11-11)
- [ ] `[Claude]` Retrieval quality on the benchmark: measured, A 66.7% / B 38.9% / C 44.4% (`Docs/RETRIEVAL_EVAL.md`, raw in `results/`). Answer-level quality still to measure with the pipeline
- [x] `[You]` Decide the fix for buried answers → small-to-big windows (setting C); k sweep dropped
- [ ] ~~Page graph~~ and ~~reranker~~: deferred. Only built if the benchmark shows a gap they fix
- [ ] `[Claude]` Experimental hybrid BGE + BM25 (RRF, top 10/15 → 4) with shared text normalisation: retrieval 13/18 vs A 12/18, end-to-end no gain; A stays default (`Docs/RETRIEVAL_EVAL.md`)
- [ ] `[You]` Review `Docs/ISSUES_AND_PLAN.md` and pick the next step

### Phase 2: Questions
- [x] `[Claude]` Draft benchmark: 22 questions in `data/eval/benchmark.json`, every answer verified against the PDFs by `test/benchmark_test.go`
- [ ] `[You]` Review, edit and add your own questions to the benchmark (the brief requires questions you wrote)

### Phase 3: Pipeline
- [ ] `[Claude]` Model client: Groq streaming, first-token timing, first-token timeout, retry policy (`internal/llm`), offline tests pass
- [ ] `[Claude]` Governor: RPM / TPM / daily budget from Groq headers, fail-fast when a wait would miss the deadline (`internal/governor`)
- [ ] `[Claude]` 5-step pipeline (`internal/pipeline`): understand → retrieve → decide (off-topic refusal) → draft (streamed) → check (per-sentence citation + number check, one re-draft); passages-only fallback at the daily limit; `cmd/ask`. Offline tests pass, **not yet run against Groq**
- [x] `[You]` Approve a live smoke run (3 benchmark questions, ≤15K tokens) → ran 2026-11-11: 5,254 tokens, q05 ✓, q19 refused ✓, q14 incomplete (`results/smoke/`)
- [ ] `[Claude]` Smoke-test fixes: parts + completeness check + targeted retrieval, unique citation labels, daily-limit wording, grader normalisation, network timing; regression tests; offline benchmark A 14/22 correct, 0 wrong (`Docs/PIPELINE_OFFLINE_EVAL.md`)
- [x] `[You]` Approve a second small live run → ran 2026-11-11 18:55: 5,690 tokens; q05 ✓, q14 and q19 ✗ (model wrote `P1:` not `[P1]`), cold connection = 1,015 ms (`results/smoke/20261111-185506_summary.md`)
- [ ] `[Claude]` Smoke-2 fixes: part-tag parsing, MISSING/content check, 350/150 token caps + lean targeted prompt, connection warm-up, oracle real-model formats; offline A 14/22 in all 3 formats
- [x] `[You]` Approve a third small live run → ran 2026-11-11 19:26: 6,742 tokens; q14 complete ✓, q05 ✓ (Table 2 citation lost to a split tag), q19 ✗ (targeted round unaffordable → passages-only); warm-up used no quota (`results/smoke/20261111-192654_summary.md`)
- [ ] `[Claude]` Smoke-3 fixes: keep split tag groups together (fakes now stream 3-char pieces), skip an unaffordable targeted round, q05 §6.1 source; both regression tests fail without their fix; offline A 14/22 in all 3 formats
- [ ] `[You]` Run 3 questions end to end and judge the answers

### Phase 4: Probe the real service
- [ ] `[You]` Approve the request budget before it runs (daily limit is shared by everything)
- [ ] `[Claude]` Probe script: latency spread, limits, what a 429 looks like, recovery time
- [ ] `[Claude]` Fill Table 1 from the probe

### Phase 5: Stand-in model
- [ ] `[Claude]` Stand-in HTTP server: slow, fail, 429, slower under load, streaming; calibrated from Phase 4
- [ ] `[You]` Review where it is least like Groq (you must explain this live)

### Phase 6: Experiments
- [ ] `[Claude]` Load generator: steady, burst, sustained overload, RPM halved, daily budget gone → `results.csv`
- [ ] `[Claude]` 3+ retrieval settings run
- [ ] `[You]` Grade answers, or verify Claude's grading (`answer_grade` column)
- [ ] `[Claude]` Script that recomputes Tables 2a, 2b, 3 from `results.csv`

### Phase 7: Write-up (2 pages max)
- [ ] `[Claude]` Draft sections 1-6, Retrieval, Misleading retriever, plus the tables
- [ ] `[You]` Write section 8 (unsure / left out / learned), Vaani feedback, How I worked with the assistant
- [ ] `[You]` Export this conversation history
- [ ] `[You]` Final read: can you defend every number?
- [ ] `[Claude]` Final check: no key in the repo, tables match the CSV

## Submission plan (2026-11-11 20:00, Praroop: stop tuning, ship)

Same tick rule: a box is ticked only after you confirm it. Groq calls only where a step says so.

- [x] `[Claude]` 1. Freeze: 15 quality + 3 unanswerable in `data/eval/frozen.json`, pinned to the benchmark's sha256 (`test/frozen_test.go`) (approved 2026-11-11: the four drops q09, q11, q16, q20)
- [ ] `[Claude]` 2. `results.csv` logging with the brief's 12 columns (`internal/results`, `test/results_test.go`); real runner appends one row per request; answer cache off in every measured run
- [ ] `[Claude]` 3. Stand-in (`cmd/standin`): slow, fails, 429 at limits, slower under load, streams; built on `internal/fakellm` *(built: `internal/standin`, `cmd/standin`, `test/standin_test.go`; see `Docs/STANDIN.md`)*
- [ ] `[Claude]` 4. Calibrate the stand-in from 10–20 real Groq calls at ≤ 3 calls/min (Groq: ~20K tokens) *(done: 15 calls, 14,469 tokens (8,748 pipeline + 5,721 probes), `results/calibration/`; first real 429 captured)*
- [ ] `[Claude]` 5. Real quality run: frozen 15 on A, paced (`cmd/smoke -set quality -pace 30s`) (Groq: ~45 calls, ~45K tokens)
- [ ] `[Claude]` 6. Load harness + 5 scenarios on the stand-in: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted *(harness: `cmd/load`; run 2026-11-11 21:5x)*
- [ ] `[Claude]` 7. Table 3 for A / H10 / H15: retrieval columns measured for all three; answer quality real for A, offline/stand-in for H10/H15, labelled as such
- [ ] `[Claude]` 8. `cmd/tables`: every table number recomputed from `results.csv`, "not measured" where true
- [ ] `[Claude]` 9. Report: 2 pages, the brief's headings in order, incl. retrieval quality vs cost, misleading-retriever, Vaani feedback, real conversation history
- [ ] `[You]` 10. Vaani feedback in your words, conversation export, final read
- [ ] `[Claude]` 11. Package: code, `results.csv`, raw logs, tables, report; key scan (no key in repo)
- [ ] `[You]` 12. Approve commit + push

## Decision log

Feeds the "where I overruled it" part of the write-up. Add a row whenever you change or reject something Claude proposed.

| Date | Decision | Who decided | Why |
|---|---|---|---|
| 2026-11-09 | Go instead of Python (uv) | You | Your choice |
| 2026-11-09 | Semantic retrieval with local BAAI BGE, no BM25 (Claude had proposed BM25) | You | You judged BM25 not good enough |
| 2026-11-09 | Add a page graph for retrieval expansion, built by code (regex), not by the LLM | You | Pull in cited tables/figures and split paragraphs at zero Groq cost |
| 2026-11-09 | `bge-base-en-v1.5` over small / large / m3 (Claude had suggested bge-m3) | You | Middle of the latency/quality trade-off; caps chunks at 512 tokens |
| 2026-11-11 | Official BAAI weights from Hugging Face, in-process (hugot + ONNX Runtime), instead of Ollama's community conversion | You (from your YTransformer setup) | Official weights, no extra server that can stop mid-test |
| 2026-11-11 | Save the weights on disk in `models/`; nothing downloads at runtime | You | No repeated 417 MB downloads |
| 2026-11-11 | Sections from each PDF's own outline, not regex on text; References not indexed; chunk sizes 300 and 480 | Claude (needs your OK) | Regex mistook tables and page numbers for headings; bibliographies match every query; 480 leaves margin under 512 |
| 2026-11-11 | Tables as structured data from the arXiv LaTeX source, not Groq structured output | You | Exact cells, 0 Groq tokens (LLM rebuild would cost ~50K, a quarter of the daily budget, and can misplace numbers) |
| 2026-11-11 | Tried k = 5 / 7 / 10, then dropped it; keep 3 settings A / B / C with C = small-to-big windows | You | Measured on 5 probes: 3/5 found at every k and both chunk sizes; k=10 costs 2.2x the tokens of k=5 (2678 vs 1241 at 300-token chunks) for no gain. The 2 misses rank 25-73, so k alone cannot reach them |
| 2026-11-11 | Freeze retrieval at A / B / C; no knowledge graph or reranker without benchmark evidence; priority is a working end-to-end system | You | Deadline; avoid over-engineering retrieval |
| 2026-11-11 | Retrieval default = A (300 tok, k=4); B and C kept for the Table 3 comparison; A+meta not adopted (+1/18, within noise) | Claude (needs your OK) | Best of the frozen settings on the 22-question benchmark |
| 2026-11-11 | Stream the draft, but show a sentence only after it passes the check (cited passage exists, every number is in it) | Claude (needs your OK) | First-token time is the first verified sentence: a little slower, never an unsupported claim on screen |
| 2026-11-11 | Refuse without a model call when the best retrieval score < 0.50 | Claude (needs your OK) | Every answerable benchmark question scores >= 0.626; off-library 0.38-0.47; saves budget |
| 2026-11-11 | Answer completeness: split into parts, check each is covered, one targeted retrieval for missing parts; up to 3 model calls per request | You (design), Claude (implementation) | q14 in the smoke test answered 2 of 3 parts |
| 2026-11-11 | Retrieve before understand (not in parallel) | Claude (needs your OK) | Off-topic questions refused with 0 model calls; costs ~20 ms of overlap |
| 2026-11-11 | Draft max 350, understand max 150 tokens; targeted draft carries only new passages | You approved (fix 3) | Groq charges max_completion_tokens against the 8K TPM up front; 3-call answers hit the limit in smoke 2 |
| 2026-11-11 | Warm the Groq connection at startup with an unauthenticated GET | You approved (fix 4) | Measured: cold DNS+TCP+TLS = 1,015 ms of the first call |
| 2026-11-11 | Try BM25 again, but only as an experimental hybrid with BGE (RRF); keep A unless it measurably improves | You (reverses the 2026-11-09 "no BM25" for an experiment only) | Exact-term misses (H100, FineWeb) |
| 2026-11-11 | Calibrate on the 4 questions the freeze dropped (q09, q11, q16, q20), not on the frozen 18 | Claude (needs your OK) | Real pipeline-shaped calls without spending or seeing the frozen set |
| 2026-11-11 | Second limit probe with ~5K-token prompts (3 more calls, 15 total) | Claude (within your 10-20 calls) | Probe 1 (max 4,500 tokens) was not charged in full, so it hit no limit; prompt tokens are always charged |
| 2026-11-11 | Stand-in TPM = token bucket (8,000, refill 133.3/s), not a 60 s window | Claude (from measurement) | Every Groq header and the real 429's "try again in 19.845s" fit a bucket exactly |
| 2026-11-11 | Governor left as is (60 s window, waits for full reset) before the load run | Claude (needs your OK) | "No more tuning"; it is more cautious than Groq, so the load run shows its cost; candidate fix afterwards |
| 2026-11-11 | Hybrid not adopted: +1/18 retrieval, 0 end-to-end, 1 new wrong answer (q14) | Claude, by your rule | `results/retrieval_hybrid.csv`, `results/pipeline_offline_20261111-194541_hybrid.csv` |
| 2026-11-11 | Freeze 15 quality + 3 unanswerable; drop q09, q11 (redundant evidence chunk), q16 (formula artifact), q20 (least library-like unanswerable); keep all known failures | You (freeze), Claude (selection rule, needs your OK) | Result-blind rule; the 3 dropped answerable questions all pass today, so the freeze can only lower A's score |
| 2026-11-11 | Table 3 settings = A, H10, H15; A stays default; answer cache off in measured runs; no 22-question live run; no further tuning | You | Ship a complete submission |
| 2026-11-09 | `openai/gpt-oss-120b` for all model calls (Claude suggested 20b for Understand) | You | Llama 3.3 70B shut down 2026-08-16; 120b is Groq's named replacement |
