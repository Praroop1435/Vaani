# The real service and the stand-in (2026-11-11, model switch 2026-11-12)

> **Model switch (2026-11-12).** The pipeline now runs on `qwen/qwen3.8-27b` (13 of 15 answerable correct, 0 wrong, against 11 and 3 for `gpt-oss-120b` on the same retrieval; Table 3). The limits below were measured on `gpt-oss-120b`; qwen's headers on 41 responses show the same 8,000 tokens/min bucket (130 tokens/s) and 1,000 requests/day. No 429 probe was run on qwen: its 429 behaviour is assumed the same. The stand-in's latency is now fitted to 35 real qwen calls from the scored run (`results/calibration_qwen/`): first token = **427 ms + 128 ms per 1K prompt tokens × lognormal(σ 0.31)**, 2.7 ms per content token, ~103 hidden reasoning tokens per call. Real qwen first token on those calls: p50 591 ms, p95 1,183 ms. Sections 1–3 below are the original `gpt-oss-120b` calibration, kept because Table 1 comes from it.


## 1. What we measured on Groq (`openai/gpt-oss-120b`, free tier)

15 real calls for calibration, never more than 3 in any minute. Raw data is in `results/calibration/`:
- `20261111-213734_traces.jsonl`: 4 pipeline questions, 9 model calls. They are the 4 questions the freeze dropped (q09, q11, q16, q20), so the frozen set stays unseen.
- `*_probe.jsonl`: 2 limit probes of 3 calls each.
- `results.csv`: rows `calibration*`.

The 3 earlier smoke runs (19 more calls, `results/smoke/`) are used only as a held-out check.

| What | Value | How we know | Confidence |
|---|---|---|---|
| Tokens per minute | **8,000, as a token bucket**: 8,000 tokens refilling at 133.3/s | `x-ratelimit-limit-tokens: 8000` on every call. On all 26 header snapshots, (8,000 − remaining) ÷ `reset-tokens` = 133.3 tokens/s = 8,000/60. The 429 message says "Limit 8000" | high |
| What a call is charged | prompt + `max_completion_tokens`, at admission | The 429 says "Requested 5328" for a 5,278-token prompt with max 50. Smoke 1: 486 = 186 + 300. **Exception:** a 4,500 max was *not* reserved in full: two such calls at once were both admitted, ~149 tokens each (probe 1). Our calls use max ≤ 350, where prompt + max holds | high for max ≤ 350; rule above that unknown |
| Requests per day | **1,000, refilling 1 every 86.4 s** | `x-ratelimit-limit-requests: 1000`. `reset-requests` grows 86.4 s per request (1m26.4s after 1, 14m24s after 10) | high |
| Requests per minute | 30 | Groq docs only. No header reports it and we never reached it | medium |
| Tokens per day | not reported in headers | Groq docs only | low |
| Requests in flight at once | ≥ 2 accepted simultaneously | Probes 1 and 2 sent 2 at once. More would break the ≤ 3/min rule | low |
| What it returns at a limit | HTTP 429, `retry-after: 20`, JSON `{"error":{"message":"Rate limit reached for model … on tokens per minute (TPM): Limit 8000, Used 5318, Requested 5328. Please try again in 19.845s. …","type":"tokens","code":"rate_limit_exceeded"}}`, plus the usual `x-ratelimit-*` headers | Probe 2, `20261111-214432_probe.jsonl` | high (n = 1) |
| Cost of a refused call | nothing | After the 429, remaining requests stayed at 989 and tokens were not charged | high (n = 1) |
| Time until it recovers | exactly `retry-after`: (used + requested − limit) ÷ 133.3/s = 2,646 ÷ 133.3 = 19.85 s | Recovery call at 20.5 s: HTTP 200 | high (n = 1) |
| Prompt tokens | ≈ 71 + 0.889 × chars/4 | 78 tokens for 30 chars, 5,278 for 23,430 chars. Check: q05's understand prompt predicted 226, real 239 | medium |
| First content token | 9 calibration calls: p50 624 ms, p95 1,786 ms. 19 held-out smoke calls: p50 598, p95 2,532, max 2,532 | per-call `first_ms` in traces | medium (n = 28) |
| Hidden reasoning | ~35 completion tokens per call that are never shown | `tokens_out` minus reply length ÷ 4 | medium |
| Failures | 0 of 34 calls (5xx, cut streams, no-token stalls) | all traces | 95% upper bound ≈ 9% |

**Our governor (v1) was more cautious than Groq.** It counted tokens over a 60 s sliding window and, after a header, waited until `reset-tokens`, the time until the bucket is *full*, rather than until *this call* fits. Fixed in v2 (`governor.Version = "v2-token-bucket"`): a continuous bucket refilling at 133.3/s that admits a call once the available tokens cover it. `TestGovernorReplaysRealTPMProbe` replays the real 429 (wait 19.845 s). Every load row in `results.csv` is v2; earlier rows are in `results/load/archive_pre_rerun/`.

## 2. What the stand-in does

`internal/standin` is the service. `cmd/standin` runs it on its own port, and `cmd/load` runs it in-process. It serves OpenAI-style `POST /chat/completions` with SSE streaming, plus `GET /models` for the warm-up.

| Brief asks | Stand-in |
|---|---|
| refuse at a limit | Groq's measured semantics: TPM token bucket charged prompt + max at admission, 429 with Groq's message, `retry-after` = time until this call fits, refused calls charged nothing. RPD refills continuously. RPM is a sliding 60 s window (assumed). A call larger than the whole TPM gets 413 (from Groq's docs, not observed) |
| send its reply a piece at a time | 3-character SSE pieces (Groq's pieces split `[S1][S2]`), `x_groq.usage` at the end, `[DONE]` |
| be slow | first token = (547 ms + 155 ms per 1K prompt tokens) × lognormal(σ 0.47), fitted to the 9 calibration calls (`standin.Fit`). Then 3.5 ms per content token |
| be slow for no visible reason | 2% of calls send their first token only after 8 s, past the pipeline's 6 s first-token timeout (assumed rate) |
| fail | 1% HTTP 500 and 0.5% of streams cut half way (assumed rates) |
| become slower under load | up to 3 calls in flight run at full speed. Each call beyond that makes all of them 30% slower (assumed) |
| what it says | The oracle (`internal/fakellm`): a perfect reader that answers only from the passages it's given. So answer grades on the stand-in measure retrieval and the pipeline, not the model |

Fit vs held-out real calls (first token):

| | p50 | p95 |
|---|---|---|
| real, 9 calibration calls (the fit set) | 624 ms | 1,786 ms |
| stand-in on those prompts | ~690 ms | ~1,500 ms |
| real, 19 smoke calls (held out) | 598 ms | 2,532 ms |
| stand-in on those prompts | ~670 ms | ~1,490 ms |

The median is ~10% slow and the p95 ~40% fast: the stand-in is too uniform.

## 3. Where the stand-in is least like Groq

1. **Slowdown under load is invented.** We could not load the free tier to measure it, so capacity 3 and +30% per extra call are guesses. Real Groq serves thousands of tenants, and its latency depends on *their* load, not ours.
2. **The tail is too light.** The real p95 is ~2.5 s against the stand-in's ~1.5 s. One real understand call took 2.4 s with nothing visible to explain it. With 34 real calls, the tail beyond p95 is unknown.
3. **The fault rates are assumptions.** We saw 0 failures in 34 calls. 1% / 0.5% / 2% are chosen, not measured.
4. **The model is perfect.** The oracle never misreads, never substitutes a related figure, and always uses the format asked for. Real gpt-oss wrote `P1:` instead of `[P1]` and split tags mid-stream. Answer quality on the stand-in is an upper bound.
5. **The large-`max_completion_tokens` charge rule is not modelled.** The stand-in always reserves prompt + max. Groq didn't for max = 4,500. Our calls never go above 350.
