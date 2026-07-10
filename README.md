# Deep Research Pipeline (Perplexity-style Agentic Search)

An agentic research pipeline that decomposes a query into sub-tasks, iteratively
searches/scrapes/retrieves until it's confident in an answer (or honestly admits
it isn't), and synthesizes a final grounded response — with the full reasoning
trace saved to `brain.md`.

This is not a wrapper around a single RAG call. It's a multi-stage system with
retry logic, two-stage retrieval, and an explicit grounding contract: **the
synthesizer is only allowed to state what the retrieved sources actually say.**

---

## Architecture

```
Query
  │
  ▼
Decompose into 2 sub-tasks (task_decomposer.py)
  │
  ▼
For each sub-task:
  │
  ├─ Generate 3 search queries (targeted using known_info / missing_info
  │  from the previous iteration, not blind retries)
  │
  ├─ Search (Tavily) → Scrape (BeautifulSoup, Tavily-extract fallback)
  │  → Semantic chunk (meaning-boundary splitting, not fixed token windows)
  │  → Insert into Pinecone (384-dim embeddings)
  │
  ├─ Retrieve (two-stage):
  │    1. Vector search → top 15
  │    2. LLM filter (relevance) → cross-encoder rerank → top 10
  │
  ├─ Answer strictly from retrieved sources (JSON output):
  │    { answer, confidence, grounded, known_info, missing_info }
  │
  └─ If confidence < 0.85 → retry (max 2 iterations), targeting
     missing_info specifically instead of re-searching blindly
  │
  ▼
Synthesize final answer from brain.md
  (explicitly instructed: don't fabricate, flag ungrounded sections)
  │
  ▼
Final answer + full reasoning trace saved to brain.md
```

**Two-level loop structure:**
- Outer loop: sub-tasks (what to answer)
- Inner loop: iterations per sub-task (how confident, with targeted re-querying)

---

## Key design decisions

- **Confidence-gated retries, not fixed-count retries.** A sub-task stops
  iterating once the model's self-assessed confidence (grounded in the actual
  retrieved chunks, not answer length) crosses 0.85, or after 2 iterations.
- **Known/missing info tracking across iterations.** Each iteration's answer
  call reports what it could confirm and what's still missing. The next
  iteration's search queries are generated *against that gap*, not blindly
  regenerated from scratch.
- **Full transparency.** `brain.md` logs every query, every retrieved chunk,
  every confidence score, and every retry — not just the final answer. This
  was essential for debugging (see below) and is also just a better trust
  story than a black-box answer.
- **Grounding is enforced at the prompt level, explicitly.** The synthesizer
  is instructed to only state facts present in the retrieved context sections
  and to flag sub-tasks that didn't reach the confidence threshold instead of
  writing around the gap.

---

## The debugging story (the actual interesting part)

Early versions of this pipeline looked like they worked — the final answer was
detailed, well-cited, and confident. It was also, on inspection, **almost
entirely fabricated.** Tracing that down surfaced a chain of real bugs, most of
which wouldn't show up unless you specifically went looking for them:

1. **The retrieved source chunks were never actually written to `brain.md`.**
   The synthesizer received sub-task answers and confidence scores, but zero
   real source content — meaning every specific detail in early outputs
   (including fabricated regulatory filing numbers) was invented, not
   retrieved. This was the root cause of the hallucination, not a synthesizer
   prompting issue.
2. **Query generation leaked model reasoning into the actual search queries.**
   A free-tier model's chain-of-thought (`"Okay, the user wants me to
   generate..."`) was being parsed as if it were a search query and sent
   straight to Tavily.
3. **Confidence was a string-length heuristic, not a real measure.** The
   original implementation scored confidence based on answer length, meaning
   a long, fluent, fabricated answer scored *higher* than a short, honest
   "insufficient data" response.
4. **Silent failure defaults masked all of the above.** A bare `except`
   returning a hardcoded confidence of 0.4 meant the retry loop had no idea
   it was retrying against broken output.
5. **A cross-encoder reranker was being reloaded from disk on every single
   retrieval call** instead of once at startup — a real (if less dramatic)
   performance bug once the correctness issues were fixed.
6. **Scraper fallback crashes on paywalled/academic sources.** Tavily's
   extraction fallback returned empty results for many paywalled journals
   (Lancet, AJCN, NEJM), and the code indexed into the empty result list
   without checking, crashing the whole pipeline mid-run instead of skipping
   the source.

Fixes applied: structured JSON output enforced via `response_format`, robust
JSON extraction that tolerates a model prepending reasoning text before valid
JSON, defensive `None`/empty-response checks on every API call, real
retrieval-grounded confidence scoring, dedup on retrieved chunks across
queries, model reload moved to `__init__`, and fail-safe (not fail-crash)
handling on every scrape/search/parse step.

---

## Known limitations (honest, not exhaustive)

- **Sub-tasks are answered independently** — sub-task 2 doesn't have access to
  sub-task 1's confirmed answer, only that it exists. There's no dependency
  chaining (e.g. "given firms X, Y, Z, now find their strategies").
- **No source quality weighting.** A forum post and a peer-reviewed paper are
  ranked purely on embedding similarity + cross-encoder relevance, with no
  signal for domain authority or publication type.
- **No caching or incremental knowledge base.** Every run re-scrapes and
  re-embeds from scratch, even for repeated or overlapping queries.
- **Single-dimensional confidence.** "Are relevant sources present," "does
  the source actually support this specific claim," and "is the source
  trustworthy" are currently collapsed into one self-reported 0–1 score.
- **No streaming output** — this is a batch script, not an interactive
  experience.
- **Confidence threshold (0.85) and iteration cap (2) are fixed constants**,
  not tuned against a labeled eval set. There is currently no formal eval
  harness — validation so far has been manual, on both a data-sparse query
  (niche financial topic) and a data-rich query (general health topic), to
  check grounding behavior holds up in both conditions.

---

## Project structure

```
main.py              # Entry point
pipeline.py           # Orchestrator: decomposition, retry loop, retrieval, answer generation
task_decomposer.py    # Breaks query into sub-tasks
search.py             # Tavily search wrapper
scraper.py             # BeautifulSoup fetch with Tavily-extract fallback
chunker.py             # Semantic (meaning-boundary) chunking
vector_utils.py        # Pinecone insert/retrieve, embedding + cross-encoder rerank
llm.py                 # classify_niche, decompose, rank, filter_chunks utilities
synthesizer.py         # Final grounded synthesis from brain.md
brain.md               # Output: full reasoning trace + final answer (generated per run)
```

## Setup

```bash
pip install -r requirements.txt  # openai, tavily-python, beautifulsoup4,
                                  # sentence-transformers, pinecone-client, python-dotenv
```

`.env`:
```
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
PINECONE_API_KEY=...
```

```bash
python3 main.py
```

## Model

All LLM calls run through OpenRouter (`openai/gpt-4o-mini`) with
`response_format={"type": "json_object"}` enforced on every structured-output
call. Embeddings via `all-MiniLM-L6-v2`, reranking via
`cross-encoder/ms-marco-MiniLM-L-6-v2`, both loaded once at startup.

---

## What's next

- Chain sub-task context (pass confirmed entities from sub-task 1 into
  sub-task 2's search/answer prompts)
- Source authority weighting in the rerank stage
- A small labeled eval set to validate the 0.85 confidence threshold rather
  than treating it as a given constant
- Basic caching for repeated/overlapping queries