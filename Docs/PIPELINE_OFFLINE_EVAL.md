# Offline pipeline evaluation (2026-11-11, no Groq tokens)

`go run -tags ORT ./cmd/evalpipeline` runs all 22 benchmark questions through the full pipeline against `internal/fakellm.Oracle`. Raw rows: `results/pipeline_offline_20261111-184923.csv` (results.csv columns plus extras). Traces: `results/pipeline_offline_20261111-184923_traces.jsonl`.

**What the oracle measures:** it plays a model that reads perfectly and never guesses. It answers a part only when a passage in its prompt contains that part's PDF-verified evidence. Otherwise it writes `[P#] MISSING`. So these numbers measure **retrieval + completeness check + targeted retrieval + citations + grading, with the model held perfect**.

**What they don't measure:**
- Real model errors, such as the smoke test's q14 substitution of batch size for total tokens.
- Latency and token counts: oracle tokens are chars/4 and its latency is ~0 ms.
- The upper-bound framing: this is the best answer quality that this retrieval allows.

## Results

| Setting | Correct | Wrong | Refused correctly | Refused wrongly | Citation ok (answerable) | Targeted rounds | Rescued by targeted | Model calls / question |
|---|---|---|---|---|---|---|---|---|
| **A** 300 tok, k=4 | **14/22 (63.6%)** | 0 | 4/4 | 4 (q01 q02 q12 q17) | 14/18 | 10 | q14, q15 | 2.45 |
| B 480 tok, k=3 | 10/22 (45.5%) | 0 | 4/4 | 8 | 10/18 | 15 | q07, q14, q16 | 2.68 |
| C windows, k=4 | 11/22 (50.0%) | 0 | 4/4 | 7 | 11/18 | 14 | q08, q14, q15 | 2.64 |

For comparison, retrieval alone (run 3, `results/retrieval_eval_run3.csv`) found all evidence for **A 12/18**, B 7/18 and C 8/18. Targeted retrieval adds **2–3 answerable questions per setting**. In every setting, every correct answer also cites a passage that holds its evidence, on the right page.

**What it costs:** a targeted round is one extra model call. A runs one for 10 of 22 questions. These are the 4 unanswerable questions (so every refusal now costs 3 calls), the 4 that stay wrongly refused, and the 2 it rescues. That's 2.45 calls per question, against 2.0 before.

**Still refused wrongly in A (4):** q01, q02, q12 and q17. Neither the first nor the targeted search finds their passage. The answer is then "The papers do not say", which is wrong but not a made-up claim. For q02 the answer is the first sentence of the chunk after a retrieved one. That's the one case where neighbour expansion would help.

## Changes made after the live smoke test

| Issue (smoke test) | Change | Regression test |
|---|---|---|
| q14 answered 2 of 3 parts | Understand returns `{"query", "parts"}`. The draft tags each sentence `[P#]` or writes `[P#] MISSING`. The check tracks which parts are covered. One **targeted retrieval** per missing part (≤2 new chunks per part, ≤4 total), then a draft for the missing parts only. Whatever is still missing is said plainly ("I could not find …") | `TestPipelineCompletesQ14WithTargetedRetrieval`, `TestMissingPartIsParsedAndRetargeted` |
| Duplicate labels `[…, p.8] […, p.8]` | Unique tag per passage (S1..Sn). The label adds the section or table (`p.8, §6.1` vs `p.8, Table 2`). A tag group renders as one bracket, each label once | `TestCitationLabelsAreDistinctOnSamePage` |
| "Budget refills at HH:MM" was the next single request | `governor.Daily()` holds the limit, requests left, refill interval (24 h / limit = 86.4 s), next request and full reset (`x-ratelimit-reset-requests`). The estimate also refills from the last header, so the system isn't stuck for 24 h | `TestGovernorDailyBudgetRefillsGradually`, `TestPipelineDailyBudgetGivesPassagesNotGuesses` |
| Grader: "7 billion" vs `7\s*B`, U+202F spaces, "(Cieri et al., 2004)" missing from LaTeX captions | `eval.NormText`: Unicode spaces and hyphens, billion/million/trillion → B/M/T. `eval.NormEvidence` also drops author-year and `[n]` citations. Shared grader `eval.GradeAnswer` used by smoke and benchmark runs | `TestNormTextUnitsUnicodeAndHyphens`, `TestEvidenceIgnoresCitationMarkers`, `TestGradeSmokeTestQ14Answer` |
| Benchmark: 3 sources truncated or inconsistent (q13 ×2, q15) | Quotes completed. A new test requires every source of a single-part question to satisfy its accept patterns | `TestBenchmarkIsVerifiedAgainstThePDFs` |
| (found offline) Passage header lost the table name | Header now reads `(Moshi, page 23, section 5.1 …, Table 4)` again, for the model and the oracle | offline benchmark |

## First-token latency: what the smoke test shows, and what changed

**Measured** (`results/smoke/20261111-182937_traces.jsonl`):

| Call | q05 | q14 | q19 |
|---|---|---|---|
| Understand total | 1030 ms | 527 ms | 240 ms |
| Draft first token | 584 ms | 918 ms | 352 ms |

**What's measured:** the understand step's time falls question by question (1030 → 527 → 240 ms) while its output stays similar (81, 92, 45 tokens). Each call also records its reasoning tokens and the rate-limit headers.

**My hypothesis, not yet measured:** much of q05's 1,030 ms is setting up the connection. The first request in a process pays for DNS, TCP and TLS, and later requests reuse that connection. Every call now records `net` timings (connection reused, DNS, connect, TLS, time to first byte). The next live run can separate network time from model time.

**Changed:**
- Retrieval with the raw question now runs **before** understand. It takes about 20 ms, and an off-topic question is refused without any model call. Running it in parallel with understand would save those ~20 ms but spend a Groq request on every off-topic question.
- Understand now does more work (parts plus query, as JSON), so its output is a little longer.

**Options to measure live before choosing, all with understand kept:**
1. **Warm the connection at startup** with one request. This removes cold-connection time from the first user question. Whether it's worth it depends on the network timings.
2. **Cap understand's output lower** (300 → 150 tokens). Groq charges `max_completion_tokens` against the per-minute token limit up front, so this halves understand's reservation. Its latency impact is unknown.
3. **Use `openai/gpt-oss-20b` for understand only.** It's faster, and its rate limits are counted separately from 120b's. This reverses your all-120b decision, so it needs your OK.

## After live smoke test 2 (2026-11-11 19:17)

Smoke test 2 (`results/smoke/20261111-185506_summary.md`) showed that gpt-oss writes `P1: …`, `P2: [P2] MISSING.` and `P1: [S1] MISSING` instead of `[P1]`. This broke part tracking and let a content-free sentence answer an unanswerable question. Five fixes:

| Fix | What changed | Regression test (raw replies verbatim) |
|---|---|---|
| 1. Part tags | Parser accepts `[P1]`, `P1:` and `**P1**`, with 1–2 digits so "P100 GPUs" stays text. Tags are never shown. The prompt now lists parts as `[P1] …`, the form the model copies | `TestLiveFormatColonPartTagIsNotShown`, `TestLiveFormatMixedMissingRetargetsTheRightPart` |
| 2. Check | A unit containing `MISSING` is never shown and covers nothing, with or without `[S#]`. A shown sentence must contain a word or a number once tags are removed. A first version required 3 words, which dropped real terse answers (q06 table cell, q16 formula). The offline run caught that | `TestLiveFormatTagThenMissingIsARefusalNotAnAnswer`, `TestContentFreeAndP100` |
| 3. Token budget | Draft reservation 700 → 350, understand 300 → 150 (observed max 231 and 101). The targeted draft sends **only the new passages** | prompt-size check in `TestLiveFormatMixedMissingRetargetsTheRightPart` |
| 4. Cold start | `llm.Client.Warm`: one unauthenticated `GET /models` at startup (no API key, so no quota). Used by `cmd/smoke` and `cmd/ask` | `TestWarmSendsNoAPIKey` |
| 5. Oracle formats | `fakellm.Oracle.Style` = "" / `colon` / `mixed`, the last being the formats gpt-oss wrote live. `cmd/evalpipeline -style` | the 3 runs below |

**Offline results are the same in all 3 formats:** A 14/22 correct, 0 wrong, 4/4 refused correctly. B 10/22, C 11/22. Raw: `results/pipeline_offline_20261111-191717*.csv`, `…191720_colon*`, `…191723_mixed*`.

**Targeted-round prompt (A):** median 1,730 → 823 tokens, max 2,272 → 976. With the smaller completion cap, the per-minute reservation drops from ~2,430 to ~1,173 tokens per targeted draft. These are oracle token counts (chars/4); real Groq counts come from the next live run.

## After live smoke test 3 (2026-11-11 19:32)

Smoke test 3 (`results/smoke/20261111-192654_summary.md`) confirmed live that the model writes `[P1]`, that the completeness loop works (q14 answered all 3 parts, with "2.1 trillion tokens" coming from the targeted round), and that the warm-up uses no quota. It found two new failures and one benchmark gap:

| Fix | What changed | Regression test | Fails without the fix? |
|---|---|---|---|
| 1. Split tag group | A unit no longer ends at `[S1]` while the next text could still be another `[S#]` tag (`tagMayContinue`). The test fakes now stream 3-character pieces like Groq (`fakellm.Pieces`) instead of whole words | `TestLiveSplitTagGroupKeepsBothCitations` (q05's raw `[P1] 28.4 BLEU [S1][S2]`) | yes, checked by reverting |
| 2. Unaffordable targeted round | If the targeted draft can't get budget before the deadline (per-minute wait, daily limit, too large) and nothing was shown yet, the round is skipped (`targeted_skipped: …`) and the answer finishes with round 1's result. q19 then refuses ("The papers do not say") instead of "can't answer right now" | `TestTargetedRoundSkippedWhenTokensRunOut` (headers as Groq sent them: 0 tokens left, reset 45.727 s) | yes, checked by reverting |
| 3. Benchmark | q05 adds the §6.1 source "establishing a new state-of-the-art BLEU score of 28.4" (p.8), noted as run 4 | `TestBenchmarkIsVerifiedAgainstThePDFs` | - |

**Offline, all 3 formats:** A 14/22 correct, 0 wrong, 4/4 refused correctly. B 10/22, C 11/22. Unchanged, as expected: the oracle never sends split tags in a way the old code mishandled at word level, and offline runs have no per-minute limit. Raw: `results/pipeline_offline_20261111-193209*`, `…193213_colon*`, `…193216_mixed*`.
