# SozoGraph — The Cognitive Passport for AI Agents

[![PyPI version](https://badge.fury.io/py/sozograph.svg)](https://badge.fury.io/py/sozograph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Portable, updatable, truth-preserving memory for agentic AI**

![SozoGraph Thumbnail](https://github.com/Sozo-Analytics-Lab/sozograph/blob/main/examples/B0984453-9159-47AE-B39A-D021F5B474CE.png)

**SozoGraph** turns interaction history into a **portable cognitive snapshot** you can inject into any AI agent context on the fly.

It answers one question cleanly:

> "Given everything that has happened so far, what should an agent **currently believe** about this user?"

Not:
- what was said
- what is similar  
- what might be relevant

But:
- **what is true now**
- **what is stable**
- **what is unresolved**
- **what is contradictory** (resolved by time)

---

## 🎯 Why This Exists

Most "memory" systems treat memory as **text to retrieve**, not **truth to update**:

| Approach | Strengths | Fatal Flaw |
|----------|-----------|------------|
| **Prompt stuffing** | Simple | Token explosion, no forgetting, degraded reasoning |
| **Vector RAG** | Good semantic recall | Answers "what was said" not "what is true now" |
| **App-specific DBs** | Fast queries | Brittle schemas, zero portability |

**The result?** Agents act like goldfish even when data exists.

### SozoGraph is different

It's a **truth-layer memory object**:
- ✅ **Typed** — Facts ≠ preferences ≠ entities ≠ open loops
- ✅ **Temporal** — New updates override old; contradictions are explicit
- ✅ **Portable** — Lightweight JSON passport + compact context string
- ✅ **Deterministic** — Same inputs → same memory state

---

## 🚀 Quick Start

### Install

```bash
pip install sozograph
```

### Try It Now

**[📓 Run the Example Notebook](https://github.com/Sozo-Analytics-Lab/sozograph/blob/main/examples/sozograph_example.ipynb)** — See live demos of ingestion, contradiction tracking, and context export.

### Basic Usage

```python
from sozograph import SozoGraph

sg = SozoGraph()

# Ingest a transcript
passport, stats = sg.ingest(
    "I'm building AI agents. I prefer direct answers and hate jargon.",
    meta={"user_key": "u_123"}
)

# Export compact context for your agent
briefing = sg.export_context(passport, budget_chars=2500)
print(briefing)
```

**Output:**

```
SOZOGRAPH PASSPORT v1
User: u_123
Updated: 2026-02-04T19:26:00+00:00

Facts (current beliefs):
- role: AI agent development

Preferences:
- communication_style: direct, jargon-free
...
```

-----

## 💡 Core Capabilities

### 1. Typed Memory (Not Just Text Blobs)

```python
# Agent sees structured beliefs, not raw transcripts
{
  "facts": {"current_project": "sozograph"},
  "preferences": {"tone": "direct"},
  "entities": ["Gemini 3", "PyPI"],
  "open_loops": ["finalize v1 docs"],
  "contradictions": []
}
```

**Why this matters:** Agents can update facts without losing preferences, distinguish current state from history, and maintain consistency across sessions.

### 2. Temporal Contradiction Tracking

```python
# Jan 15: "I live in NYC"
# Feb 1: "I moved to SF"

# RAG: Retrieves both → confusion
# SozoGraph: 
{
  "facts": {"location": "SF"},
  "contradictions": [{
    "key": "location",
    "old_value": "NYC",
    "new_value": "SF", 
    "changed_at": "2026-02-01"
  }]
}
```

### 3. Cross-Architecture Portability

Because passports are lightweight JSON, they work everywhere:

- **Stateless clients** (ElevenLabs WebSocket, voice agents)
- **Server-side orchestrators** (LangChain, AutoGen)
- **Edge deployments** (Cloudflare Workers, Vercel Edge)

**Real-world applications:**

- 🏥 **Health & Fitness** — Remember dietary restrictions, workout progressions
- 📚 **Education** — Track learning weaknesses, adapt assessments
- 🛍️ **Shopping** — Recall style preferences, purchase history
- 💬 **Support** — Maintain context across channels

-----

## 📖 Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here
SOZOGRAPH_EXTRACTOR_MODEL=gemini-3-flash-preview
SOZOGRAPH_ENABLE_FALLBACK_SUMMARIZER=true
SOZOGRAPH_MAX_INTERACTION_CHARS=4000
SOZOGRAPH_DEFAULT_CONTEXT_BUDGET=3000
```

-----

## 🔧 Advanced Usage

### Multi-Interaction Ingestion

```python
history = [
    {"createdAt": "2026-02-01T10:00:00Z", "transcript": "I'm renovating my kitchen."},
    {"createdAt": "2026-02-02T09:30:00Z", "transcript": "I prefer rustic style."},
    {"createdAt": "2026-02-03T12:10:00Z", "transcript": "Budget is $50k max."},
]

passport, _ = sg.ingest(history)
```

### Database Object Ingestion

**Firestore:**

```python
firestore_doc = {
  "id": "abc123",
  "createdAt": "2026-02-03T10:00:00Z",
  "notes": "User prefers direct answers."
}
passport, _ = sg.ingest(firestore_doc, hint="firestore")
```

**Supabase:**

```python
supabase_row = {
  "table": "events",
  "row": {"event": "preference_update", "notes": "Wants code-first approach"}
}
passport, _ = sg.ingest(supabase_row, hint="supabase")
```

**Firebase RTDB:**

```python
rtdb_snapshot = {
  "path": "/users/u1/profile",
  "value": {"displayName": "Alice", "preferences": {"tone": "casual"}}
}
passport, _ = sg.ingest(rtdb_snapshot, hint="rtdb")
```

-----

## 🏗️ How It Works

### Ingestion Pipeline

1. **Canonicalize** — Coerce inputs into `Interaction` objects (deterministic)
1. **Extract** — Gemini 3 Flash reasons about belief updates (strict JSON schema)
1. **Resolve** — Deterministic merger applies temporal priority, tracks contradictions
1. **Export** — Compact passport ready for context injection

**Key insight:** This is **belief inference**, not keyword extraction. Gemini 3’s reasoning enables distinguishing facts from preferences, detecting implicit updates, and maintaining temporal consistency.

### What SozoGraph Is NOT

- ❌ Not a graph database
- ❌ Not RAG / embeddings
- ❌ Not a conversation logger
- ❌ Not a DB client (objects-only by design)

**SozoGraph is a memory normalization layer** that sits *before* agent planning, tool use, and retrieval.

-----

## 📊 Benchmarks

|Metric                |Before (RAG)   |After (SozoGraph)|
|----------------------|---------------|-----------------|
|Context size          |~2000 tokens   |~300 tokens      |
|Factual consistency   |60%            |95%              |
|Contradictions handled|Silent failures|Explicit tracking|

*Measured on 10-turn conversations with 3 belief updates* (non-scientific experiment)

-----

## 🗺️ Roadmap

### v1.x (Near-term)

- [ ] CLI tools (`sozograph ingest`, `sozograph render`)
- [ ] Enhanced input detection for transcript lists
- [ ] Improved JSON recovery for malformed model outputs
- [ ] Stronger evidence linking

### v1.5 (Planned)

- [ ] Optional graph engine support (Neo4j, Memgraph)
- [ ] Cypher-style relational queries
- [ ] Temporal edge deprecation
- [ ] Active truth subgraph exports

### v2 (Future)

- [ ] Multi-model support (OpenAI, Claude, local models)
- [ ] MCP tool server integration
- [ ] Hybrid graph + vector patterns

-----

## 🤝 Contributing

We welcome contributions that keep v1 **disciplined and portable**.

### ✅ Good Contributions

- Adapters for new object shapes (objects-only)
- Resolver logic improvements (deterministic)
- Tests for edge cases (contradictions, merge conflicts)
- Prompt engineering for extraction quality

### ❌ Won’t Accept in v1

- RAG/embedding features
- Graph database integrations (wait for v1.5)

### How to Contribute

1. Fork the repo
1. Create a branch: `feat/your-feature`
1. Add tests where relevant
1. Open a PR with clear explanation + examples

-----

## 📚 Resources

- **[Example Notebook](https://github.com/Sozo-Analytics-Lab/sozograph/blob/main/examples/sozograph_example.ipynb)** — Interactive demos
- **[Test Fixtures](https://github.com/Sozo-Analytics-Lab/sozograph/tree/main/tests/fixtures)** — Sample data for validation
- **[PyPI Package](https://pypi.org/project/sozograph/)** — Latest release

-----

## 🎓 Philosophy

> “We are not helping agents remember more. We are helping them remember **correctly**.”

SozoGraph enables agents to maintain **consistent beliefs** across sessions, systems, and model providers—something RAG and chat history cannot provide.

-----

## 📄 License

MIT — [Sozo Analytics Lab](https://github.com/Sozo-Analytics-Lab)

-----

## 🏆 Built For

This project was built for the **Gemini 3 Global Hackathon** to demonstrate how structured memory normalization unlocks the next generation of agentic AI applications.

**Try it now:**

```bash
pip install sozograph
```

**Questions?** Open an issue or check the [example notebook](https://github.com/Sozo-Analytics-Lab/sozograph/blob/main/examples/sozograph_example.ipynb).
