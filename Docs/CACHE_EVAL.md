# Semantic answer cache: offline evaluation

**Status: built, tested offline, OFF by default.** Not yet measured on live Groq traffic.

## What it does

`Pipeline.AnswerCached` (`internal/pipeline/cache.go`) wraps `Answer`; `Answer` itself is unchanged.

1. Embed the question with BGE (same query prefix as retrieval).
2. Look it up. A cached answer is returned only if **all three** hold:
   - **Fingerprint** matches: hash of the index files, the whole pipeline `Config` (retrieval setting, k, thresholds, token limits), the model id and `cacheVersion`. Otherwise the entry is stale and gets dropped.
   - **Cosine ≥ threshold** (default 0.90, `-cache-threshold`).
   - **Key terms match**: same content words after dropping phrasing words and light stemming, plus one `a>b` term per direction word (`to`, `than`, `into`, `vs`).
3. On a miss, run the normal pipeline. Store the answer only if: status `ok`, not refused, no missing parts, no dropped sentences, at least one citation, and nothing degraded except a skipped query rewrite.

Use it: `go run -tags ORT ./cmd/ask -cache results/answer_cache.json "..."`. Without `-cache`, nothing changes.

## Matching on 37 labelled pairs (`data/eval/cache_pairs.json`)

18 `same` (2 exact repeats, 16 paraphrases) and 19 `different` (modifier, entity, attribute, number, word order). Re-run with `go run -tags ORT ./cmd/evalcache`. Raw rows: `results/cache_eval.csv`.

| threshold | same hit, with guard | different wrongly hit, with guard | same hit, cosine only | different wrongly hit, cosine only |
|---|---|---|---|---|
| 0.80 | 13/18 | **0/19** | 18/18 | 14/19 |
| 0.85 | 13/18 | **0/19** | 17/18 | 8/19 |
| 0.90 | 13/18 | **0/19** | 17/18 | 7/19 |
| 0.94 | 13/18 | **0/19** | 14/18 | 6/19 |
| 0.96 | 11/18 | **0/19** | 11/18 | 3/19 |
| 0.98 | 5/18 | **0/19** | 5/18 | 2/19 |

**Cosine alone is unsafe at every threshold.** These pairs need different answers, yet BGE scores them as near-duplicates:

| cosine | cached | asked |
|---|---|---|
| 0.999 | …WMT 2014 English-to-German? | …WMT 2014 German-to-English? |
| 0.987 | What is the input to the decoder…? | What is the decoder an input to…? |
| 0.970 | …2-stage H-Net… GPT-3 **Large**… | …2-stage H-Net… GPT-3 **XL**… |
| 0.957 | …**2**-stage H-Net… | …**1**-stage H-Net… |
| 0.956 | …English-to-**German**? | …English-to-**French**? |
| 0.907 | …the **base** Transformer use? | …the **big** Transformer use? |

The key-term guard blocks every one. Without the direction term, both word-order pairs got through: they have the same words.

**Why 0.90:** with the guard, any threshold from 0.80 to 0.94 gives the same result. 0.90 is the middle of that range, so the threshold only has to catch same-words / different-meaning cases the direction rule misses.

**What the guard costs:** 5 of 18 `same` pairs miss. A miss is safe: it just runs the normal pipeline.
- Dropped detail: "parallel attention heads" vs "attention heads"; "Mimi codec" vs "Mimi"; "Helium language model" vs "Helium".
- Extra detail: "…use to place a chunk boundary" vs "…boundary threshold".
- Infinitive "to": "Which GPUs were used **to** train Moshi?" counts as a direction.

## Latency and cost

| | measured |
|---|---|
| Lookup (embed + scan of 1,000 entries) | ~23 ms + ~1.1 ms |
| Live miss, smoke run (q05, q14) | ~1.7 s total, 2 Groq requests, 1.5–2.4K tokens |
| Expected saving per hit | ~1.7 s, 2 requests and ~2K tokens, a ~70× faster answer |
| Extra cost per miss | ~23 ms (the question is embedded twice: once in the cache, once in retrieval) |

Hit/miss counts and the time saved (sum of original time minus hit time) are in `Cache.Stats()`, printed by `cmd/ask -cache`.

## Tested (`test/cache_test.go`, offline, fake model)

- Exact repeat and paraphrase: hit, emitted text and citations identical, model called once in total, hit faster than the original.
- False match: "base" → "big" heads is blocked (cosine 0.907 ≥ 0.90), so the model is called again.
- Not stored: refusal, nothing verified, missing part, cut short, rate-limited, no citations. Stored: a verified answer with only the rewrite skipped.
- Invalidation: retrieval setting change, k change, index hash change (re-ingest), model change. Also: survives save/load, and `Invalidate()` empties it.
- The threshold setting changes the verdict.

## Open (before it can be the default)

1. Live A/B on Groq: hit rate and saved time on real repeated questions. Needs your approval for the requests.
2. You should review the 37 pairs: Claude drafted them.
3. `cacheVersion` has to be bumped by hand when prompts or check logic change. It cannot see edits to `pipeline.go`.
4. The completeness gate trusts the pipeline's part tracking (`Trace.Missing`). A verified answer that misreads the question would be cached and served again.
5. No size cap or expiry. Fine for thousands of entries.
