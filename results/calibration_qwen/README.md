Timing calibration for the stand-in after the model switch to `qwen/qwen3.8-27b` (2026-11-12).

`20261112-154102_traces.jsonl` is a copy of the scored qwen run's raw traces (`results/quality/`, 18 questions, 45 model calls). `cmd/load` fits the stand-in's first-token and per-token times from it, and `cmd/tables` reads its rate-limit headers for Table 1. No separate calibration calls were spent on qwen; the 429 probe in `results/calibration/` was run on `openai/gpt-oss-120b` only.
