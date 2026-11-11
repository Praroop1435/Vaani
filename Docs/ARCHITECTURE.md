# The Relay: High-Level Design

**Language: Go.** Single module. Only stdlib + a PDF-to-text step at ingest time.

## What we are building

An assistant that answers questions **only** from 3 arXiv papers, cites where each claim came from, and says "the papers don't say" instead of guessing. It has to stay honest and usable when the model provider's rate limits bite.

## The picture

```mermaid
flowchart LR
    LG[Load generator<br/>steady / burst / overload /<br/>RPM halved / daily gone] -->|questions| ADM

    subgraph RELAY[Relay server]
        ADM[Admission + queue<br/>deadline per request] --> P1
        P1[1. Understand<br/>LLM: rewrite query] --> P2
        P2[2. Retrieve<br/>tool: BGE semantic top-k<br/>+ 1-hop graph expand] --> P3
        P3{3. Decide}
        P3 -->|weak hits, max 2 tries| P2
        P3 -->|nothing relevant| REF[Refuse:<br/>papers don't say]
        P3 -->|good hits| P4
        P4[4. Draft<br/>LLM: cited answer, streamed] --> P5
        P5{5. Check<br/>tool: every cite exists<br/>and quote matches chunk}
        P5 -->|fail, once| P4
        P5 -->|pass| OUT[Reply]
    end

    P1 & P4 --> GOV[Limit governor<br/>RPM / TPM / daily budget]
    GOV --> MC[Model client<br/>timeout + retry policy]
    MC -->|real| GROQ[(Groq free tier)]
    MC -->|load tests| STUB[(Stand-in model<br/>our HTTP server)]

    LIB[(Paper index<br/>chunk vectors + page graph)] --- P2
    EMB[(BGE embedder in-process<br/>BAAI bge-base-en-v1.5 ONNX<br/>weights saved in models/)] --- P2
    OUT & REF --> CSV[results.csv<br/>+ per-request trace]
```

## Components

| Component | Job | Notes |
|---|---|---|
| **Ingest** (one-off) | 3 PDFs at exact versions (`data/papers/`, 120 pages), `pdftotext` per page, split into chunks tagged `paper / section / page`, embed each chunk with BGE | Chunk size is a knob, so we can compare 3 retrieval settings |
| **Embedder** | BAAI's official `bge-base-en-v1.5` ONNX weights from Hugging Face (110M params, 768-dim), saved once in `models/` and run inside the Go process with `hugot` + ONNX Runtime | Zero Groq budget, no extra server, no runtime download. [CLS] pooling as BGE was trained (hugot defaults to mean pooling, which would be wrong). Max 512 tokens: longer text is an error, never cut. Queries get BGE's "Represent this sentence for searching relevant passages: " prefix. Matches Ollama's independent conversion at cosine 0.99999 |
| **Paper index** | Chunk vectors in memory, brute-force cosine top-k | ~2K vectors: a vector DB would add a service for no speed gain |
| **Page graph** | Nodes = chunks. Edges: next/prev in the same section, section → parent section, and explicit refs ("Table 3", "Figure 2", "Section 4.1", "Eq. 5", "Appendix B") → the chunk that holds that caption or heading, same paper only | Built with regex at ingest, no LLM. Used to pull in a cited table or the rest of a split paragraph after top-k |
| **Admission + queue** | Bounded queue; each request has a give-up deadline; drop or shed when the queue is full | Goroutine worker pool sized from the measured "requests in flight" limit |
| **Pipeline** | The 5 steps above; one loop (re-retrieve) and one branch (refuse) | Step 5 is a plain code check, not an LLM call, so verifying costs no tokens |
| **Limit governor** | Token buckets for RPM and TPM, plus a daily counter; updated from Groq's `x-ratelimit-*` headers | Decides: wait, use a cheaper path (fewer chunks, skip step 1), or refuse |
| **Model client** | One interface, two backends (Groq / stand-in); streams tokens, records first-token time | Retries only on 429/5xx, honours `retry-after`, capped by the request's deadline |
| **Stand-in model** | `net/http` server speaking the OpenAI-style chat API with streaming (SSE) | Configurable: latency distribution, failure rate, 429 at RPM/TPM/daily limits, slows down as in-flight count rises. Calibrated from real Groq runs |
| **Load generator** | Fires questions in each load shape, writes one CSV row per request | Real Groq for small runs; stand-in for heavy ones |
| **Grader** | 15+ hand-written questions (some unanswerable) with expected answer + citation | Produces `answer_grade` column |

## The one decision that shapes everything

**The daily request budget is the scarcest thing, so every request spends at most 2 model calls** (understand + draft). Retrieval and citation checking are plain code. When the budget runs low we drop step 1 first, then shrink retrieved context, then refuse new requests with an honest "out of budget, try at X" reply, never a guessed answer.

## Retrieval settings to compare (Table 3)

| Setting | Chunk | Top-k | Graph expand | Purpose |
|---|---|---|---|---|
| A | ~300 tokens | 4 | no | Cheap baseline |
| B | ~480 tokens (bge-base max is 512) | 3 | no | More context per hit, more tokens |
| C | ~300 tokens | 4 | yes, 1 hop, re-ranked by cosine, token cap | Small chunks plus the table/figure/neighbour they point to |

**Not building an LLM-made knowledge graph (GraphRAG style).** Building it would send all 120 pages through Groq: roughly 120K+ tokens against a 200K daily budget, before answering a single question.

## How to run

One-time setup: `brew install onnxruntime`, then `go run -tags ORT ./cmd/getmodel` (saves the weights to `models/`, skips if already there).

Tests: `go test -tags ORT ./test -v`. Add `GROQ_LIVE=1` to also spend one real Groq request.

## Repo layout (planned)

```
cmd/relay/      pipeline server
cmd/standin/    fake model server
cmd/loadgen/    load shapes -> results.csv
cmd/ingest/     fetch + chunk papers
internal/       pipeline, retrieve, governor, llm client
data/           chunked papers, questions.json
results/        results.csv, raw traces
Docs/           brief, this file, 2-page write-up
```

## Open questions (decide after measuring Groq)

- Which Groq model: limits differ per model.
- Give-up deadline for a waiting person (start with 15 s).
- Queue size and worker count: set from measured in-flight limit.
