
# SozoGraph (v1) — The Cognitive Passport

**SozoGraph** turns interaction history (transcripts + DB objects) into a **portable cognitive snapshot** you can pass into any AI agent context on the fly.

It answers one question cleanly:

> "Given everything that has happened so far, what should an agent **currently believe** about this user?"

Not:
- what was said
- what is similar
- what might be relevant

But:
- what is true now
- what is stable
- what is unresolved
- what is contradictory (resolved by time)

---

## Why this exists (the problem)

Most "memory" systems are either:
- **prompt stuffing** (expensive, degrades reasoning, no forgetting)
- **vector RAG** (good recall, weak truth/temporal consistency)
- **app-specific notes** (non-portable, brittle schemas)

So agents keep acting like "goldfish" even when data exists.

SozoGraph v1 is a **truth-layer memory object**:
- typed (facts vs preferences vs entities vs open loops)
- temporal (new updates override old; contradictions are explicit)
- portable (a lightweight JSON passport + a compact context string)

---

## Install

```bash
pip install sozograph
```

-----

## Configure

Create a `.env` file (see `.env.example`):

```env
GEMINI_API_KEY=your_key_here
SOZOGRAPH_EXTRACTOR_MODEL=gemini-3-flash-preview
SOZOGRAPH_ENABLE_FALLBACK_SUMMARIZER=true
SOZOGRAPH_MAX_INTERACTION_CHARS=4000
SOZOGRAPH_DEFAULT_CONTEXT_BUDGET=3000
```

-----

## Quickstart

### 1) Single transcript → Passport

```python
from sozograph import SozoGraph

sg = SozoGraph()

passport, stats = sg.ingest(
    "I'm Quantilytix. I build software and want direct answers. I'm working on SozoGraph v1.",
    meta={"user_key": "u_123", "source": "transcript:demo-1"}
)

print(passport.to_compact_dict())
print(stats)  # per-interaction merge stats
```

-----

### 2) List of transcripts / message history (supported ✅)

```python
history = [
    {"createdAt": "2026-02-01T10:00:00Z", "project_title": "SozoFix", "transcript": "I'm renovating my kitchen."},
    {"createdAt": "2026-02-02T09:30:00Z", "project_title": "SozoFix", "transcript": "I prefer rustic style and hate glossy paint."},
    {"createdAt": "2026-02-03T12:10:00Z", "project_title": "SozoGraph", "transcript": "We need portable memory JSON. No infra. Truth-layer."},
]

# You can ingest a list directly. SozoGraph will coerce items internally.
passport, _ = sg.ingest(history, hint="firestore")  # hint optional; see below
```

**Tip:** If your list items aren’t “docs”, you can pass them as plain dicts and let fallback summarization help when needed. If your dicts contain a `transcript` field, extraction will still succeed (it will stringify deterministically).

-----

### 3) Firestore object ingestion (objects-only)

You fetch your Firestore data in your app, then pass the dict here:

```python
firestore_doc = {
  "id": "abc123",
  "createdAt": "2026-02-03T10:00:00Z",
  "title": "User Profile Update",
  "notes": "User says they prefer direct answers.",
  "companyCode": "QX",
}

passport, _ = sg.ingest(
    firestore_doc,
    hint="firestore",
    meta={"source": "firestore:/users/abc123", "user_key": "u_abc123"}
)
```

-----

### 4) Firebase Realtime DB ingestion (path + value)

RTDB is tree-based, so pass an envelope:

```python
rtdb_snapshot = {
  "path": "/users/u1/profile",
  "value": {
    "updatedAt": 1738560000000,
    "displayName": "Quantilytix",
    "preferences": {"tone": "direct"}
  }
}

passport, _ = sg.ingest(rtdb_snapshot, hint="rtdb", meta={"user_key": "u1"})
```

-----

### 5) Supabase ingestion (table + row)

```python
supabase_row = {
  "table": "events",
  "row": {
    "id": 77,
    "created_at": "2026-02-03T11:22:00Z",
    "event": "user_preference_update",
    "notes": "User wants strategy alignment before code."
  }
}

passport, _ = sg.ingest(supabase_row, hint="supabase", meta={"user_key": "u1"})
```
# SozoGraph Test Fixtures

These fixtures are **intentionally small and human-readable**.

They are designed to test:
- transcript ingestion
- Firestore document ingestion
- Firebase Realtime Database snapshots
- Supabase row ingestion

They are NOT meant to simulate production-scale data.
If a fixture grows beyond what a human would comfortably read,
it is probably violating SozoGraph v1 philosophy.
-----

## Export a compact agent “briefing” (context injection)

You can inject this into any agent prompt:

```python
briefing = sg.export_context(passport, budget_chars=2500)
print(briefing)
```

**Example output format:**

```
SOZOGRAPH PASSPORT v1
User: u1
Updated: 2026-02-03T12:34:56+00:00

Facts (current beliefs):
- role: software development
- current_project: sozograph v1
...

Preferences:
- tone: direct
...

Open loops:
- finalize v1 repo + publish pip package
...
```

-----

## How SozoGraph v1 works

### Ingestion pipeline (v1)

1. Coerce input into canonical `Interaction` objects (deterministic)
1. If the derived text is weak/noisy, call Gemini fallback summarizer (optional)
1. Use Gemini extractor (strict JSON) to propose memory updates
1. Use deterministic resolver to merge:

- temporal priority (latest wins)
- explicit contradictions record changes
- de-dupe entities + aliases
- keep open loops short and recent

### What SozoGraph v1 is NOT

- Not a graph database
- Not RAG
- Not embeddings
- Not a long transcript store
- Not a tool that fetches from DB (objects-only by design)

-----

## Roadmap (upcoming features)

### v1.x (near-term)

- Better input detection for common “transcript list” shapes (e.g. `{transcript, createdAt}`)
- CLI:
  - `sozograph ingest transcript.txt --out passport.json`
  - `sozograph render passport.json --budget 3000`
- Stronger JSON recovery if a model response is slightly malformed
- More deterministic evidence linking (source-id mapping improvements)

### v1.5 (planned, optional)

- Graph engine support (Neo4j Aura / Memgraph) via Bolt
- Cypher-style relational queries over memory
- Temporal deprecation on edges
- Export “active truth subgraph” to context

### v2 (optional)

- Foundational model adapters (non-Gemini backends)
- MCP tool server integration
- Hybrid patterns (graph + vector) only where needed

-----

## Contributing

We want contributions, but keep v1 disciplined.

### Good contributions

- Adapters for additional object shapes (still objects-only)
- Resolver improvements (deterministic)
- Tests for merge/contradiction edge-cases
- Prompt tuning for more stable key extraction

### What won’t be accepted in v1

- Adding DB client dependencies (firebase-admin, supabase clients, etc.)
- Building RAG/embeddings into core
- Turning v1 into a graph project

### How to contribute

1. Fork the repo
1. Create a branch: `feat/<short-name>`
1. Add tests where relevant
1. Open a PR with a short explanation and sample input/output

-----

## License

MIT — Sozo Analytics Lab

