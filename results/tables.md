### Table 1: Limits found in the real service (Groq free tier; probed on openai/gpt-oss-120b, headers re-read on qwen/qwen3.8-27b, the model used)

| Limit | Value found | How found | Confidence |
|---|---|---|---|
| Requests per minute | 30 | Groq rate-limits docs; no header reports it; never reached | medium |
| Tokens per minute | 8000, a token bucket refilling at 133.3 tokens/s | `x-ratelimit-limit-tokens` on 15 responses; (limit - remaining) / reset-tokens, median of 15 | high |
| Requests per day | 1000, refilling one every 86.4 s | `x-ratelimit-limit-requests`; reset-requests grows by that much per request (median of 12) | high |
| Tokens per day | 200,000 | Groq rate-limits docs; no header reports it; never reached | low |
| Requests in flight at once | >= 2 (sent together, all admitted); more not tested | results.csv calibration_probe rows | low |
| What it returns when a limit is hit | HTTP 429 `rate_limit_exceeded`, `retry-after: 20`, "Limit 8000, Used 5318, Requested 5328. Please try again in 19.845s"; the refused call is not charged | results.csv calibration_probe row + raw probe log | high (n=1) |
| Time until it recovers | 21405 ms after the 429 a call succeeded (retry-after said 20 s) | results.csv calibration_probe rows | high (n=1) |
| Same limits on qwen/qwen3.8-27b | 8000 tokens/min refilling at 130.0 tokens/s; 1000 requests/day | headers on 41 responses of the scored run (results/calibration_qwen/); no 429 probe on this model | high for the limits, 429 behaviour assumed the same |

### Table 2a: Latency, one row per load scenario

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady (limiter v2) | stand-in | 16 | 454 | 2.0 | 3745 / 12798 / 12798 (n=16) | 3779 / 12798 / 12798 (n=16) |
| burst (limiter v2) | stand-in | 32 | 334 | 5.6 | 4167 / 11545 / 11545 (n=11) | 4167 / 11545 / 11545 (n=11) |
| sustained overload (limiter v2) | stand-in | 64 | 478 | 8.0 | 12762 / 18652 / 18991 (n=22) | 12762 / 18693 / 18991 (n=22) |
| per-minute limit halved (limiter v2) | stand-in | 16 | 466 | 2.0 | 10463 / 19675 / 19675 (n=12) | 10463 / 19675 / 19675 (n=12) |
| daily budget exhausted (limiter v2) | stand-in | 16 | 227 | 4.0 | 3161 / 3384 / 3384 (n=4) | 3161 / 3384 / 3384 (n=4) |

### Table 2b: Outcomes, same rows

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| steady | 100.0 | 0.0 | 0.0 | 0.06 | 1 | 2407 | $0.49 (free tier: $0) |
| burst | 34.4 | 62.5 | 0.0 (+3.1% error) | 0.12 | 21 | 3206 | $0.74 (free tier: $0) |
| sustained overload | 34.4 | 62.5 | 1.6 (+1.6% error) | 0.03 | 3 | 2770 | $0.66 (free tier: $0) |
| per-minute limit halved | 75.0 | 18.8 | 0.0 (+6.2% error) | 0.12 | 1 | 2404 | $0.51 (free tier: $0) |
| daily budget exhausted | 25.0 | 75.0 | 0.0 | 0.00 | 1 | 1942 | $0.36 (free tier: $0) |

### Table 3: Retrieval settings

| Setting (piece size, pieces fetched, re-rank) | Model | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| A: 300-token pieces, 4 fetched, no re-rank | real Groq, gpt-oss-120b | 18 | 50.0 | 11.1 | 16.7 | 2378 | 1546 / 4956 | 1546 / 5409 |
| R: 300-token pieces; paper scope, BGE top-20 + BM25 top-20, cross-encoder re-rank, 6 kept | real Groq, gpt-oss-120b | 18 | 61.1 | 16.7 | 16.7 | 2852 | 2854 / 5784 | 2854 / 6537 |
| R: 300-token pieces; paper scope, BGE top-20 + BM25 top-20, cross-encoder re-rank, 6 kept | real Groq, qwen3.8-27b | 18 | 72.2 | 0.0 | 16.7 | 2987 | 2979 / 6280 | 2979 / 6280 |
| A: 300-token pieces, 4 fetched, no re-rank | stand-in (oracle model) | 18 | 61.1 | 0.0 | 16.7 | 1903 | 1816 / 3989 | 1884 / 3989 |
| H10: BGE top-10 + BM25 top-10 fused (RRF re-rank), 4 kept | stand-in (oracle model) | 18 | 61.1 | 5.6 | 16.7 | 1921 | 1753 / 2932 | 1754 / 2932 |
| H15: BGE top-15 + BM25 top-15 fused (RRF re-rank), 4 kept | stand-in (oracle model) | 18 | 55.6 | 5.6 | 16.7 | 1904 | 1899 / 3806 | 1899 / 3806 |
| R: 300-token pieces; paper scope, BGE top-20 + BM25 top-20, cross-encoder re-rank, 6 kept | stand-in (oracle model) | 18 | 72.2 | 0.0 | 16.7 | 2215 | 3650 / 7186 | 3650 / 7186 |

"Correct and cited": answer_grade correct. Every sentence the pipeline shows carries a citation to a passage that contains its numbers (the check refuses any other), so a correct answer is always cited. Percentages are of questions asked, so the four grades (with refused wrongly) sum to 100%.

Computed by `go run ./cmd/tables` from `results/results.csv` (280 rows). Percentiles nearest-rank; latencies over ok requests; tokens per answer = all tokens / ok requests; cost at $0.15 / $0.60 per 1M input / output tokens (Groq paid price; free tier cost $0).
