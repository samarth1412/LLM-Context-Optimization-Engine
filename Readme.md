# LLM Context Optimization Engine

A portfolio-grade FastAPI project for reducing long-conversation LLM cost while preserving useful memory. The app stores chat sessions in SQLite, keeps recent turns in full, summarizes older turns incrementally, caches summaries, and reports token/cost usage across both chat calls and background memory operations.

This is designed to show AI/ML software engineering work, not just prompt wrapping: context-window management, memory compression, usage accounting, benchmarking, and a dashboard for inspecting what the model actually receives.

## What It Demonstrates

- Incremental summarization for long conversations
- SQLite-backed session persistence and summary caching
- Real model selection with OpenRouter-compatible model IDs
- Deterministic `mock/echo` model for local demos and tests without an API key
- Token and cost accounting for chat, summary, and compression calls
- Context preview showing system memory, cached summary, and recent messages
- Synthetic benchmark comparing full-history, sliding-window, and summary-memory strategies
- Unit tests for context construction, caching, ordering, deletion, and usage accounting

## Architecture

```text
Browser dashboard
  -> FastAPI API
      -> context.py      builds optimized model context
      -> llm_utils.py    calls OpenRouter or mock model
      -> database.py     stores messages, summaries, pinned context, usage ledger
      -> benchmark.py    reproducible synthetic cost comparison
```

The memory strategy is:

```text
Older messages                 Recent messages
summarized once and cached  +  preserved verbatim
```

For each request, the backend sends:

```text
system prompt + pinned context + cached conversation memory + recent messages + current prompt
```

## Benchmark

Run:

```bash
python benchmark.py --turns 100 --words-per-message 90
```

To export artifacts for the README/dashboard:

```bash
python benchmark.py --json --export
```

This writes:

- `results/benchmark.json`
- `results/benchmark.csv`
- `results/benchmark.png`

Current synthetic benchmark output:

```text
Model: x-ai/grok-4-fast
Conversation: 100 turns, 200 messages, 90 words/message

strategy                 input      bg_input   bg_output  max_req   cost       savings
--------------------------------------------------------------------------------------
full_history             2,110,545          0          0   42,055  $0.439609    0.00%
sliding_window             316,200          0          0    3,306  $0.080740   81.63%
incremental_summary        492,812     38,537      6,899    5,518  $0.127219   71.06%
```

The incremental-summary result includes background summary overhead, so the comparison is more honest than only counting final chat requests.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your OpenRouter key to `.env`:

```text
OPENROUTER_API_KEY=your_key_here
```

Run:

```bash
python main.py
```

Open:

```text
http://localhost:9000
```

For local testing without an API key, select `mock/echo` in the model dropdown.

## Tests

```bash
python -m unittest discover -s tests
```

The tests use a temporary SQLite database and do not call external APIs.

## API

```text
GET    /                         dashboard
GET    /api/health               service status
GET    /api/models               configured model registry
POST   /api/chat                 non-streaming chat
POST   /api/chat/stream          streaming chat
GET    /api/messages/{session}   recent stored messages
GET    /api/context/{session}    context preview sent to the LLM
GET    /api/stats/{session}      token and cost accounting
GET    /api/summary/{session}    cached/generated summary
GET    /api/sessions             saved sessions
POST   /api/set-story/{session}  pinned source context
DELETE /api/session/{session}    delete all session state
```

## Project Structure

```text
.
├── main.py              FastAPI app and endpoints
├── context.py           context building and incremental summarization
├── llm_utils.py         OpenRouter/mock model calls
├── database.py          SQLite persistence and usage accounting
├── benchmark.py         synthetic strategy benchmark
├── index.html           dashboard UI
├── tests/               unit tests
├── config.py            model, pricing, prompt, and memory settings
├── delete.py            guarded admin cleanup helper
├── requirements.txt
└── .env.example
```

## Resume Positioning

Suggested project title:

```text
LLM Context Optimization Engine with Incremental Memory and Cost Benchmarking
```

Suggested resume bullet:

```text
Built a FastAPI-based LLM context optimization engine that compares full-history, sliding-window, and incremental-summary memory strategies; added SQLite persistence, cached summaries, model-aware token estimation, cost accounting for foreground/background LLM calls, deterministic tests, and a dashboard for inspecting context composition.
```

## Next Improvements

- Add retrieval memory with embeddings/vector search
- Add summary drift evaluation against source messages
- Add entity memory for facts, characters, preferences, and unresolved threads
- Export benchmark results as charts for the dashboard
- Add Docker and CI for one-command deployment
