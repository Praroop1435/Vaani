# What is going wrong, and how we fix it (2026-11-11 19:45)

Deadline: **2026-11-13** (4 days from the brief on 2026-11-09). Every item below cites the run that shows it. Raw data is in `results/`.

**Status key:** 🔴 open, blocks the submission · 🟠 open, hurts a result · 🟡 open, known limit to document · ✅ fixed, regression test in `test/`

## 1. Answer quality

| # | Problem | Evidence | Root cause | Fix | Status |
|---|---|---|---|---|---|
| Q1 | **4 of 18 answerable questions are wrongly refused** (q01 heads, q02 dropout, q12 H100, q17 FineWeb 100B) | `pipeline_offline_20261111-193209.csv`: A 14/22 correct, 4 refused wrongly | **Retrieval:** the answering chunk isn't in the first top 4 or in the targeted search. q01: the answer is the last sentence of a formula-heavy chunk (BGE rank 27). q02: the answer is in the chunk *after* a retrieved one (rank 244). q12, q17: the question has none of the answer's words ("H100", "FineWeb") | Tried: hybrid BM25 (§4). Not tried: (a) a neighbour-chunk expansion for q02's case; (b) a BM25-weighted fusion (BM25 alone ranks q12 #2, q01 #5, q02 #8). Both need questions outside the benchmark to avoid tuning to these 4 | 🟠 |
| Q2 | **The check can't catch a sentence that answers a different question** | Smoke 1, q14: "4.2M tokens per step" for "how many tokens", numbers grounded, so the check passed | The check verifies citations and numbers, not meaning | Prompt rule "never substitute a related figure". It held in smoke runs 2 and 3: the model wrote `MISSING`. No code-level guard. Option: require one content word of the part (e.g. "tokens") in the sentence; cheap, but would need measuring | 🟡 |
| Q3 | **Answers are terse fragments** ("Fisher dataset [Moshi, p.8, Table 1]") | Smoke 3, q14 | The prompt asks for tagged sentences, not full ones | One prompt line: "Write each answer as a complete sentence". Needs one live check | 🟠 |
| Q4 | **The real model has only answered 3 questions** | Smoke runs 1–3 | Budget: the full benchmark is ~22 × 2.5 calls ≈ 55 requests, ~60–70K tokens, ~1/3 of the daily budget, run at ≤ 3 questions/min | One paced live run of all 22 questions on A (~10 min), after your approval. This produces Table 3's real answer quality | 🔴 |
| Q5 | **The benchmark is Claude's draft, revised 4 times after seeing results** | `data/eval/benchmark.json` notes (runs 2, 3, 4) | Some evidence lists were incomplete. Every addition was a passage that fully answers, and is noted | **Freeze the benchmark now.** Praroop reviews and adds his own questions (the brief requires it). No more edits after the live run | 🔴 |
| Q6 | **A refusal can list the same table twice** ("Moshi, p.33, Table 11; Moshi, p.33, Table 11", q20; also q22 in the real unanswerable run) | Calibration run, q20 | A table split by rows is two chunks with the same label | Known limit for the submission (Praroop, 2026-11-11): does not change grades or citations' correctness | 🟡 |
| Q7 | **A cited answer from the wrong paper passes the check** ("The Transformer models were trained on H100 GPUs [Moshi, p.20, §4.4]", q04; truth: 8 P100s) | Real quality run, `results/quality/20261111-222925_traces.jsonl` | The check needs a real citation and every number in the cited passage; it never asks whether the passage is about the thing asked | Known limit, not built (no tuning on the scored set). Next fix: reject a citation from a paper the question does not name | 🟡 |
| Q8 | **An answer can repeat its one sentence** (q12, real run) | Real quality run | Model wrote the same cited sentence twice; no de-duplication | Known limit, cosmetic | 🟡 |
| Q10 | **R and qwen were chosen after seeing the frozen run** | Real runs 2026-11-11 (A, 9/15) and 2026-11-12 (R) | The first frozen run showed the weak spots; fixing them means the scored set is no longer unseen | R tuned only on `data/eval/dev.json` (18 new questions, 18/18 evidence found); stated in the report §8 | 🟡 |
| Q11 | **Qwen's understand reply is cut off** on 7 of 18 scored questions ("not JSON: unexpected end of JSON input") | `results/quality/run_qwen_stdout.txt` | Qwen's hidden reasoning (~103 tokens) eats most of the 150-token cap | Pipeline falls back to the raw question, which still answered correctly. Next fix: raise the cap to ~300 or set `reasoning_effort` lower for qwen. Not changed after the scored run | 🟡 |
| Q12 | **q17 is never retrieved** (H-Net's 100B FineWeb-Edu) | Every setting, both models | No rare word shared with the question | Known limit | 🟡 |
| Q9 | **A per-minute 429 that outlasts its retries tells the person "the model service failed"** (status `error`), not the per-minute message | Load rerun, overload, request 008-q01 (`results/load/20261111-235628_overload_traces.jsonl`) | `onError` in `internal/pipeline/pipeline.go` maps only governor waits and daily 429s to `rate_limited`; a provider per-minute 429 falls through to the default | Known limit (Praroop, 2026-11-12): fixing it changes that row and would force an overload rerun. One-line fix: treat a provider per-minute 429 like `ErrDeadlineWait` | 🟡 |
| ✅ | q14 answered 2 of 3 parts | Smoke 1 | No part tracking | Parts + completeness check + targeted retrieval. Smoke 3: q14 correct | ✅ `TestPipelineCompletesQ14WithTargetedRetrieval` |
| ✅ | Model wrote `P1:`, and a content-free "answer" reached q19 | Smoke 2 | Parser knew only `[P1]`; no content check | Parser accepts all three forms; MISSING never shown; content check | ✅ `TestLiveFormat*` |
| ✅ | `[S1][S2]` split mid-stream lost a citation | Smoke 3, q05 | Groq streams small pieces | `tagMayContinue`; fakes stream 3-char pieces | ✅ `TestLiveSplitTagGroupKeepsBothCitations` |

## 2. Rate limits and reliability

| # | Problem | Evidence | Root cause | Fix | Status |
|---|---|---|---|---|---|
| R1 | **Per-minute tokens (8K) are the binding limit: about 3 questions/min live** | Smoke 3: tokens left 7,611 → 1,903 over 3 questions in ~15 s | 2–3 model calls per question. Groq charges `max_completion_tokens` against TPM up front (measured: 486 = 186 prompt + 300 max) | Done: draft cap 700 → 350, understand cap 300 → 150, the targeted draft sends only new passages (~half the prompt). Next: measure the real reservation per question in the full live run | 🟠 |
| R2 | **Every refusal costs 3 calls** (understand, draft, targeted draft) | Offline: 2.45 calls/question for A | The targeted round runs before giving up | Keep, but skip it when it doesn't fit (done) or below 100 daily requests (done). Option: skip it for single-part questions whose first draft says MISSING with a best score < 0.70 (unanswerable top scores 0.59–0.72, answerable 0.63–0.80). Overlapping ranges: needs measuring | 🟡 |
| R3 | Fixes for split tags and an unaffordable targeted round are **only tested offline** | Offline tests | - | Covered by the Q4 live run | 🟠 |
| R5 | **Limiter v1 waited for the whole TPM bucket** (~40 s) where Groq admits a call once available ≥ cost (19.8 s in the real probe) | `results/calibration/20261111-214432_probe.jsonl` | 60 s sliding window + waiting for `reset-tokens` (time to *full*) | v2: continuous token bucket, 133.3/s; all five load scenarios rerun on v2 with per-scenario seeds; earlier rows in `results/load/archive_pre_rerun/` | ✅ `TestGovernorReplaysRealTPMProbe` |
| R6 | **The local re-ranker is the queue under a burst** (re-rank p50 8.6 s in the burst, 1.8 s at steady load) | `results/load/*_burst_traces.jsonl`, step `retrieve_rerank` | One cross-encoder on the CPU behind a mutex; 20 questions at once wait in line | Known limit. Next fix: re-rank fewer candidates (top-12 found 17/18 on dev in 1.2 s) or a second worker | 🟡 |
| R4 | One call was slow for no visible reason (understand first byte 2,430 ms against 235–604 ms) | Smoke 3, q14 | Provider-side | First-token timeout 6 s with retry. Record the distribution in the live and load runs | 🟡 |
| ✅ | Daily-limit text conflated the next request with the full reset | Smoke 1 headers | Groq refills 1 request / 86.4 s | `governor.Daily()`, gradual refill | ✅ `TestGovernorDailyBudgetRefillsGradually` |
| ✅ | Cold connection added ~1 s to the first question | Smoke 2: DNS 879 + TCP 56 + TLS 80 ms | First request in a process | Unauthenticated warm-up. Smoke 3: no quota used, first call 369 ms | ✅ `TestWarmSendsNoAPIKey` |
| ✅ | An unaffordable targeted round turned a refusal into "can't answer" | Smoke 3, q19 | Round 2 treated as essential | Skip it and finish with round 1 | ✅ `TestTargetedRoundSkippedWhenTokensRunOut` |

## 3. Not built yet (required by the brief)

| # | Missing piece | Brief section | Plan | Status |
|---|---|---|---|---|
| B1 | **Stand-in model service**: slow, fails, 429s at limits, slower under load, streams | "What to build" | Extend `internal/fakellm` into `cmd/standin`. Latency from the smoke traces (first byte 235–2,430 ms, ~60–230 output tokens), Groq's real limit semantics (TPM charges prompt + max at request time; RPD refills 1/86.4 s; `retry-after`), and a load-dependent slowdown | 🔴 |
| B2 | **Load harness + 5 scenarios**: steady, burst, sustained overload, RPM halved, daily budget gone | Tables 2a, 2b | `cmd/load`: open-loop arrivals, a give-up timer per request, writes `results.csv` rows. Steady on real Groq at ≤ 2 q/min. Others on the stand-in | 🔴 |
| B3 | **`results.csv` in the brief's exact columns + a script that recomputes every table** | "Raw file" | One `cmd/tables` reads `results.csv` and prints Tables 2a, 2b and 3. Table 1 comes from headers | 🔴 |
| B4 | **Retrieval Table 3 with answer quality** for A, B, C | Table 3 | From the Q4 live run (A) plus offline oracle rows (B, C), labelled as such. Or a smaller live run for B and C if budget allows | 🔴 |
| B5 | **The 2-page write-up**, Vaani feedback, conversation export | "What your 2 pages must contain" | Last. Most content already exists in `Docs/*.md` | 🔴 |

## 4. Experimental hybrid retrieval (this change): not adopted

Hybrid BM25 is BGE top-N plus BM25 top-N, fused by reciprocal rank fusion (RRF) into the top 4, with N = 10 or 15. Details are in `Docs/RETRIEVAL_EVAL.md`.

| | A | H10 | H15 |
|---|---|---|---|
| Retrieval: answerable questions fully found | 12/18 | **13/18** (+q15) | **13/18** (+q15) |
| End-to-end, oracle model: correct | **14/22** | 14/22 (+q12, −q14) | 13/22 |
| End-to-end: wrong answers | **0** | 1 (q14 partial) | 1 |

**A stays the default.** Hybrid's retrieval gain is 1 question, and end-to-end it's no better: it trades q12 for q14, because the targeted search for "tokens Helium was pretrained on" now prefers chunks that merely repeat "tokens".

## 5. Other things to know
- **Answer cache (another session's work)** in `internal/pipeline/cache.go`: its fingerprint covers config and index but **not the prompt text**. Prompts changed twice today. Coordinate before it's used in any measured run.
- **Table extraction:** H-Net Table 1 "Input" column (a centred multirow); table section is where the caption falls. Minor, documented.

## Order of work to the deadline

**Superseded 2026-11-11 20:00** by the submission plan in `Docs/WORKPLAN.md`: frozen 15 + 3 set, A / H10 / H15 for Table 3, the stand-in for every load scenario, no full 22-question live run, no more tuning.


| When | Step | Groq cost | Needs your OK |
|---|---|---|---|
| Now | Q3 (full-sentence prompt) and Q5 (freeze benchmark; you review and add questions) | 0 | yes (Q5 is yours) |
| Next | **Q4: one paced live run of all 22 questions on A**: real answer quality, tokens, latency, reservations | ~60–70K tokens, ~55 requests | yes |
| Then | **B1 stand-in**, calibrated from Q4's traces | 0 | - |
| Then | **B2 load scenarios**: steady on Groq (small), the other 4 on the stand-in | small | yes, for the Groq part |
| Then | **B3 tables script, B4 Table 3** | 0 | - |
| Last | **B5 write-up** | 0 | - |
