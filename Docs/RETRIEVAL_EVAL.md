# Retrieval evaluation (2026-11-11)

Retrieval only, no Groq calls. Answer correctness, citation accuracy of generated answers and refusals are measured later, once the pipeline exists. Groq token cost is estimated as the context tokens of the retrieved chunks.

## Benchmark

`data/eval/benchmark.json` has **22 questions**: 10 numeric, 4 table, 2 multi-passage, 2 definition and 4 unanswerable. Every expected answer is tied to an exact quote on a stated PDF page, or to a table cell. `test/benchmark_test.go` checks every quote, cell and "absent" term against the PDF text itself, not against our chunks. It passes.

**Ownership:** the brief asks for questions *you* wrote. This set is Claude's draft, so Praroop must review it, edit it and add his own.

**Metric:** a question counts as *found* when every evidence item is covered by a retrieved chunk (quote contained, or table chunk with that row and value). *Page ok* means the covering chunk's page range includes the evidence page, which is a proxy for citation accuracy.

## Results: frozen settings, run 2 (`results/retrieval_eval_run2.csv`)

| Setting | Answerable found | Evidence items | Page ok | Context tokens/answer | Retrieval ms p50 / p95 |
|---|---|---|---|---|---|
| **A**: 300 tok, k=4 | **12/18 (66.7%)** | 13/20 | 12/18 | 1066 | 17.8 / 26.5 |
| B: 480 tok, k=3 | 7/18 (38.9%) | 8/20 | 8/18 | 1088 | 17.5 / 26.3 |
| C: 2-sentence windows → 300 tok parent, k=4 | 8/18 (44.4%) | 9/20 | 8/18 | 1098 | 19.0 / 29.3 |

**Run 1** (`results/retrieval_eval_run1.csv`) scored A 10/18, B 7/18 and C 8/18. Between runs I added two sources to the answer key (q04, q13) after reading run-1 hits: passages that fully answer but weren't listed. This is noted in each question. Nothing else changed.

## Run 3 (`results/retrieval_eval_run3.csv`), after the smoke-test fixes

Same settings. The matcher now ignores Unicode spaces and citation markers, and q14 has one evidence item per part. A is 12/18 (q14 2 of 3 parts), B is 7/18 and C is 8/18, with headline numbers unchanged. Evidence items found: A 15/21, B 10/21, C 11/21.

## Experiments (`results/retrieval_variants.csv`), not adopted

| Variant | Answerable found | Evidence items | Page ok | Context tokens/answer |
|---|---|---|---|---|
| A+meta: "paper title \| section" prefixed to each chunk before embedding | 13/18 (72.2%) | 15/20 | 14/18 | 1038 |
| C+rows: C plus one window per table row (347) | 8/18 (44.4%) | 9/20 | 8/18 | 1127 |

- **A+meta: +1 question over A** (gained q01 and q15, lost q13). With 18 answerable questions, one question is 5.6 points, which is too small to call. It costs one re-embed and no Groq tokens. It's a candidate, not adopted: the architecture is frozen.
- **C+rows: no change.** C's misses are text passages (q10 latency, q15, q17), not tables, so table-aware windows don't fix C.
- **Latency in this run (33–37 ms) isn't comparable** to the table above. It ran right after 3 minutes of embedding on the same CPU. The query embedding itself is identical.

## Findings

1. **The Mimi "12.5 Hz" miss was my test bug, not retrieval.** The paper writes the rate 3 ways: "12.5 frames per second" (p10), "12.5Hz" (p11, p16, p23) and "12.5 Hz" (Table 1). The probe matched only "12.5 Hz". With a correct matcher, A ranks a 12.5 Hz chunk 2nd, B ranks it 1st (Table 4), and C ranks it 5th, so C misses at k=4. The benchmark now lists all four sources.
2. **Small-to-big (C) wins the 5-probe smoke test but loses the benchmark** (44% against A's 67%). Two-sentence windows match isolated sentences ("h = 8") but lose to other windows on facts that need the surrounding paragraph or a whole table.
3. **Bigger chunks (B) are worst.** At 480 tokens with k=3, the answer sentence is a smaller share of each vector, and there are fewer chunks to hit.
4. **Similarity can't decide refusals.** Mean top score is 0.71 for answerable questions and 0.65 for unanswerable ones. The ranges overlap. q12 ("What GPUs trained Moshi?") looks unanswerable but is answered on p20. The pipeline's draft and check steps must decide refusals, not a cosine threshold.
5. **One miss is a neighbour-chunk miss.** For q02, the answer sentence is the first sentence of the chunk after the retrieved one. That's the only benchmark evidence for graph or neighbour expansion so far, which isn't enough to build it.
6. **Misses in every setting:** q12 (H100, p20) and q17 (FineWeb-Edu 100B, p11). Those passages never reach the top 4.

## Decision for the pipeline

Use **A** as the default retrieval setting. It's the best of the frozen settings on the benchmark, and the cheapest in tokens. B and C stay for the end-to-end Table 3 comparison. No graph and no reranker: the benchmark doesn't show a gap they'd close better than A.

## Experiment: hybrid BGE + BM25 with RRF (2026-11-11 19:45), not adopted

**Setup.** BGE top-N and BM25 top-N over the same 300-token chunks, fused by reciprocal rank fusion. The fused score is Σ 1/(60 + rank), and the top 4 are kept. N is 10 (H10) or 15 (H15). BM25 uses Okapi k1 = 1.2 and b = 0.75; neither BM25 nor RRF is tuned on the benchmark. Both the BM25 index and the grader use `internal/textnorm`. It handles NFKC Unicode (ligatures, math italics, narrow spaces), hyphen look-alikes, "7 billion" = "7B", "12.5Hz" = "12.5 Hz", possessives, and splits tokens at punctuation and letter/digit boundaries. `Hit.Score` stays the BGE cosine, so the pipeline's 0.50 off-topic cutoff means the same thing. Raw: `results/retrieval_hybrid.csv` (5 repetitions per query for latency).

| Setting | Answerable fully found | Evidence items | Page ok | Context tokens/answer | Search-only ms p50 / p95 (110 queries) | Embed + search ms p50 / p95 |
|---|---|---|---|---|---|---|
| **A** (default) | 12/18 (66.7%) | 15/21 | 13/18 | 1066 | 0.459 / 0.728 | 18.0 / 35.8 |
| H10 | 13/18 (72.2%) | 16/21 | 14/18 | 1116 | 0.623 / 0.829 | 18.3 / 30.9 |
| H15 | 13/18 (72.2%) | 16/21 | 14/18 | 1078 | 0.633 / 0.785 | 18.0 / 27.6 |

**End-to-end with the oracle model** (`results/pipeline_offline_20261111-194541_hybrid.csv`): A 14/22 correct, 0 wrong. H10 14/22, 1 wrong. H15 13/22, 1 wrong. H10 gains q12 (H100) but loses q14: its targeted search for "number of tokens Helium was pretrained on" returns c021 and t006 (chunks that repeat "tokens") instead of c065 ("the 2.1T tokens used to train Helium").

**Why RRF doesn't fix q12, q01 and q02, although BM25 finds them.** BM25 alone ranks their evidence **#2, #5 and #8**, where BGE ranks it 24th, 27th and 244th. RRF rewards chunks that appear in *both* lists, about 2 × 1/61, over a chunk found by BM25 alone, 1/62. So the 4 slots go to chunks both methods half-like. q17 can't be helped by BM25: the question shares no rare word with the answer ("FineWeb", "100B").

**Decision:** A stays the default. Hybrid's gain is 1 retrieval question (5.6 points on 18) and none end-to-end, with a new wrong answer. Next experiment to try, on questions beyond these 18: weighted fusion, or one slot reserved for the best BM25-only hit.
