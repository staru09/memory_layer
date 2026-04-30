# AI Companion with Long-Term Memory

A conversational AI companion that remembers past conversations using a multi-tier memory architecture. Two retrieval strategies: rolling summary for chat (fast, lossy), hybrid search on permanent facts for precise queries (lossless). 2 LLM calls per ingestion.

## Architecture

```
Conversation messages <-> Ingestion / Retrieval libraries <-> Gemini 2.5 Flash
                                       |
                              PostgreSQL + Qdrant
                                       |
                               Memory Pipeline
                       (Ingestion + Profile + Summary + Facts)
```

### Memory System

**Short-term memory:** Last 20 messages from the current thread (regardless of ingestion status) — immediate conversational context.

**Long-term memory:** 5 components:

| Component | Description | Storage |
|-----------|-------------|---------|
| **User Profile** | Explicit facts + implicit traits, updated via add/update/delete ops | PostgreSQL |
| **Foresight** | Time-bounded events with `valid_from` / `valid_until` / `duration_days`, auto-expired | PostgreSQL |
| **Rolling Summary** | Two-tier: Archive (compressed old) + Recent (date-tagged entries) | PostgreSQL |
| **Facts Table** | Permanent, never compressed — every extracted fact stored forever | PostgreSQL (tsvector + GIN) |
| **Facts Vectors** | Embeddings for semantic search | Qdrant (3072-dim, cosine) |

### Two Retrieval Strategies

**Chat endpoint (`/chat`)** — conversational, general awareness:
```
  Profile + Foresight + Rolling Summary + Last 20 messages
```

**Query endpoint (`/query`)** — precise recall:
```
Parallel:
  [Thread 1] Gemini embed query
  [Thread 2] profile + foresight + messages + keyword search
Then:
  Qdrant vector search (~200ms) → RRF fusion with keyword results → top 5 facts
```

---

## Ingestion Pipeline

**2 LLM calls per ingestion (extract + profile update), +1 when compression triggers.**

```
Conversation (20 messages)
       |
  [1] FETCH CONTEXT (1 DB read)
  |   Rolling summary for coreference resolution
  |
  [2] EXTRACT (1 LLM call — Gemini)
  |   Input: conversation + rolling summary
  |   Output: consolidated facts (category-tagged, dated) + foresight signals
  |   Dates sanitized: invalid LLM dates replaced with current_date
  |
  [3] In parallel (3 threads):
  |   [Thread 1] PG store — 3 parallel connections:
  |   |   facts INSERT + foresight expire/INSERT + summary read/append
  |   [Thread 2] Embed facts (Gemini embedding API)
  |   [Thread 3] Profile update (1 LLM call):
  |   |   LLM receives numbered profile + new facts
  |   |   Returns add/update/delete operations as JSON
  |   |   Operations applied programmatically (not LLM rewrite)
  |   |   Conflicts logged to conflict_log table
  |
  [4] Qdrant upsert (needs fact_ids from PG + embeddings from Gemini)
  |
  [5] Compression checks:
      If rolling summary >= 90% of 10k token budget:
      |   → Evict oldest 25% of recent → compress into archive (1 LLM call)
      |   → Archive capped at 20% of budget
      If profile >= 80% of 3k token budget:
          → Merge similar facts (1 LLM call)
```

### Extraction Details
- **1 LLM call** per batch — no segmentation step
- Produces **consolidated facts** — dense, self-contained sentences (not atomic fragments)
- **Attribution rules** — every fact must name the person, pronouns resolved
- **Frequency tracking** — recurring activities captured with explicit frequency
- **Verbatim details** prioritized: signs, paintings, book titles, pet behaviors
- **Quality filters** — greetings, pleasantries, acknowledgments excluded
- Prior rolling summary used as context for coreference resolution (with guard: "do NOT extract from this")

### Profile Update (add/update/delete operations)
- LLM receives numbered profile lines: `[1] - Rampal is 24 years old`
- Returns structured JSON operations referencing line numbers
- `apply_operations()` in Python handles the text editing
- More predictable than full LLM rewrite — LLM decides *what*, Python does *editing*
- Conflict detection only on profile (facts table is append-only)

### Foresight
- Time-bounded events with `valid_from`, `valid_until`, `duration_days`, `evidence`
- Auto-expired during ingestion when `valid_until` < current_date
- Examples: travel plans, illness recovery, deadlines, new jobs

### Compression
- **Trigger**: 90% of 10k token budget
- **Eviction**: oldest 25% of recent entries
- **Recursive**: `new_archive = LLM(existing_archive + evicted_entries)`
- Archive capped at 20% of budget — re-compressed if exceeded
- Token count: char-based estimate (len/4), no API call

### Ingestion Triggers
- **Production**: After 20 unprocessed messages accumulate
- **Periodic**: Every 10 minutes, checks for threads with 4+ old unprocessed messages
- **Manual**: `POST /threads/{thread_id}/ingest`
- **Benchmark**: After each session

---

## Data Storage

### PostgreSQL

| Table | Purpose |
|-------|---------|
| `user_profile` | Explicit facts + implicit traits (bullet-point format) |
| `foresight` | Time-bounded events with validity windows + duration_days |
| `conversation_summaries` | Two-tier: `archive_text` + `recent_text` + `token_count` |
| `facts` | Permanent fact store with tsvector index for keyword search |
| `conflict_log` | Detected contradictions (category, old/new value, resolution) |
| `chat_threads` | Chat session metadata |
| `chat_messages` | Raw messages with ingestion status |
| `query_logs` | Query + response audit trail |

### Qdrant

| Collection | Purpose |
|------------|---------|
| `facts` | 3072-dim embeddings with payload: fact_text, conversation_date, date_int, category, fact_id |

---

## Project Structure

```
main.py                      # Ingestion pipeline orchestrator
config.py                    # Environment config + memory budgets
db.py                        # PostgreSQL operations (compound queries, connection pooling)
models.py                    # Data classes (UserProfile, Foresight, ConflictLog, etc.)
vector_store.py              # Qdrant facts collection (search, upsert, rebuild)
gemini.py                    # Gemini client wrapper
run_locomo.py                # LoCoMo benchmark runner
locomo/                      # LoCoMo benchmark dataset (https://github.com/snap-research/locomo.git)

ingestion/
  extractor.py               # Single-call fact + foresight extraction (LLM)
  profile_extractor.py       # Profile update via add/update/delete ops (LLM)
  profile_ops.py             # Apply operations to profile text (pure Python)
  profile_manager.py         # Profile compression when over budget (LLM)
  summary_manager.py         # Summary compression when over budget (LLM)
  prompts/
    extraction.txt           # Fact + foresight extraction prompt
    profile_update.txt       # Profile add/update/delete operations prompt
    summary_compression.txt  # Recursive archive compression prompt
    profile_compression.txt  # Profile compaction (merge similar) prompt

retrieval/
  fetch_mem_service.py       # retrieve_for_query() + hybrid search + context composers
  vectorize_service.py       # Gemini embedding (single + batch)

Dockerfile                   # Python 3.12 container
docker-compose.yml           # Local PostgreSQL + Qdrant setup
```

## Key Features

- **2 LLM calls per ingestion** — extract + profile update (parallel)
- **Two retrieval strategies** — rolling summary for chat, hybrid search for queries
- **Permanent facts** — never compressed, keyword + vector searchable

---

## Performance (Supabase, Qdrant Cloud)

| Endpoint | Context | LLM | Total |
|----------|---------|-----|-------|
| Chat | ~600ms | ~2.5s | ~3.1s |
| Query | ~1.7s (embed + search) | ~2.2s | ~3.9s |

| Endpoint | Context | LLM | Total |
|----------|---------|-----|-------|
| Chat (local) | ~5ms | ~2.5s | ~2.5s |
| Query (local) | ~40ms + embed 650ms | ~2.2s | ~2.9s |

---

## LoCoMo Benchmark Results

Evaluated on the [LoCoMo](https://github.com/snap-research/locomo) benchmark across 10 sessions, scored by GPT-4 on a 1-5 scale.

| Metric | Score (%) |
|--------|-----------|
| **Overall (all categories)** | 88.3% |
| **Without adversarial** | 90.2% |

Category breakdown:

| Category | Description |
|----------|-------------|
| Temporal | Date/time-based recall |
| Multi-hop | Reasoning across multiple facts |
| Adversarial | Deliberately tricky questions |

Adversarial questions are excluded in the filtered score since they test robustness to misleading prompts rather than memory accuracy.

> **Note on retrieval during the benchmark:** although vector embeddings are generated and stored in Qdrant during ingestion, the LoCoMo run does **not** use the vector service at QA time. `run_locomo.py` retrieves context via `db.get_chat_context` only — i.e. rolling summary + foresight (the chat path, no RAG). The hybrid keyword + vector search exposed by `retrieval/fetch_mem_service.py` is wired up for the `/query` endpoint for the chat application.
---

## Getting Started

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your credentials:
#   GEMINI_API_KEY, PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB
#   QDRANT_HOST, QDRANT_PORT (or QDRANT_URL + QDRANT_API_KEY for cloud)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Local PostgreSQL + Qdrant (Docker)

```bash
docker-compose up -d
```

### 4. Run the LoCoMo benchmark

```bash
python run_locomo.py --samples 0 --workers 5
# or all 10 samples:
python run_locomo.py
```

---