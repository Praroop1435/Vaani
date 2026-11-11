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
- [ ] `[Claude]` Semantic top-k retrieve tool (cosine over BGE vectors)
- [ ] `[Claude]` Page graph: section neighbours + Table/Figure/Section/Eq refs, 1-hop expand
- [ ] `[You]` Check graph expansion on 3 questions that cite a table or figure

### Phase 2: Questions
- [ ] `[You]` Write 15+ questions in `data/questions.json` (Claude gives the template): detail questions, unanswerable ones, and traps for a naive retriever, each with expected answer + citation

### Phase 3: Pipeline
- [ ] `[Claude]` Model client: Groq streaming, first-token timing, timeout, retry policy
- [ ] `[Claude]` 5-step pipeline with re-retrieve loop and refuse branch
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
| 2026-11-09 | `openai/gpt-oss-120b` for all model calls (Claude suggested 20b for Understand) | You | Llama 3.3 70B shut down 2026-08-16; 120b is Groq's named replacement |
