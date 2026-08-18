# SozoGraph

Portable JSON memory for LLM agents.

Your agent's memory is a small JSON file. You can read it, diff it, email it, put it in Postgres, ship it to the browser. No vector database. No embedding model. No local weights.

```bash
pip install sozograph
```

```python
from sozograph import SozoGraph

sg = SozoGraph()
passport = sg.ingest(conversation)

print(passport.context(query="Where does Melanie live?"))
passport.save("melanie.json")
```

That is the whole API.

## The idea

A long context window is not memory. Attention dilutes as the sequence grows, and a model that can ingest a million tokens still loses the thread inside them. Retrieval helps and brings its own failure: one missed chunk is a wrong answer, and now you own a vector store.

SozoGraph compresses history into a belief state instead. Two layers:

**Facts and preferences.** What is true now, as keys and values. Small enough that every question sees all of it, so retrieval can never hide a fact.

**Episodes.** What happened, and when. Compact per-segment summaries with timestamps, produced by the same extraction call at no extra cost. This is what answers "what did she say about the painting in session four", which a flat key-value store threw away by construction.

Only episodes are ranked, and ranking is BM25 in pure Python. Microseconds, no model, nothing to download.

## Install

The core install is pydantic and nothing else. Pick your engine:

```bash
pip install "sozograph[anthropic]"
pip install "sozograph[openai]"
pip install "sozograph[gemini]"
pip install "sozograph[ollama]"      # local, no key, no cloud
pip install "sozograph[litellm]"     # ~100 providers
pip install "sozograph[langchain]"   # bring your own chat model
```

Every SDK is imported lazily at call time. Loading, querying, and saving a passport work with no SDK installed at all.

## Use

### Any provider

```python
SozoGraph()                                   # resolves from the environment
SozoGraph("anthropic")                        # default model for that provider
SozoGraph("openai:gpt-4o-mini")               # explicit
SozoGraph("ollama:llama3.2")                  # local
SozoGraph("openai:x", base_url="http://localhost:8000/v1")   # vLLM, Groq, Together
```

Structured output goes to each engine's native mechanism: a forced tool call with a strict schema on Anthropic, `response_format` with `strict: true` on OpenAI, `response_schema` on Gemini, grammar-constrained decoding on Ollama. The schema is never pasted into a prompt and asked for politely.

Bring an existing LangChain model and keep your callbacks, caching, and tracing:

```python
from langchain_openai import ChatOpenAI
from sozograph.providers.langchain import LangChainProvider

sg = SozoGraph(LangChainProvider(chat_model=ChatOpenAI(model="gpt-4o-mini")))
```

### Ingest

Transcripts, chat turns, database rows, or a mixed list.

```python
sg.ingest("I live in Kwekwe and I prefer terse answers.")

sg.ingest([
    {"speaker": "Melanie", "text": "I renovated the kitchen.", "ts": "2026-01-05T10:00:00Z"},
    {"speaker": "Caroline", "text": "What colour?", "ts": "2026-01-05T10:01:00Z"},
])

sg.ingest({"table": "orders", "row": {"id": 1, "notes": "Wants matte black."}})
```

Turns are batched into token-bounded segments, one extraction call each. Check the cost before you spend it:

```python
sg.plan(six_hundred_turns)
# {'interactions': 600, 'segments': 27, 'api_calls': 27,
#  'calls_saved_vs_per_interaction': 573,
#  'estimated_input_tokens': 37721, 'mean_segment_tokens': 1397.1, ...}
```

Segment count depends on how long the turns are. Nothing is called; this is
arithmetic on the input.

### Read

```python
passport.context()                                   # everything, budgeted
passport.context(query="what did she hang up?")      # episodes ranked by relevance
passport.context(budget_chars=1500)
passport.token_estimate()
```

Facts and preferences are always included in full. The query only reorders episodes.

### Move it around

```python
passport.save("user.json")
passport = Passport.load("user.json")

blob = passport.to_compact_dict()     # plain JSON, straight into any database
passport = Passport.from_dict(blob)
```

Round trip is lossless. A 1.0 passport loads and migrates. Unknown keys from a future version are preserved rather than dropped.

## Deduplication

State a preference two ways across fifty sessions and a naive extractor records it twice. Enough of that and the passport reacquires the entropy it exists to remove.

Four tiers, in increasing cost:

**Tier 0. Controlled vocabulary.** The extraction prompt carries the passport's existing keys. A model that can see `code_style` already exists reuses it instead of coining `boilerplate_preference`. Free, and it does more work than the other three combined.

**Tier 1. Exact match** on the normalized key.

**Tier 2. Guarded fuzzy match.** Jaro-Winkler plus token-set overlap, zero dependencies.

The obvious version of this rule is dangerous. "Merge above 0.85 similarity" scores `budget_min` against `budget_max` at 0.92, `is_enabled` against `is_disabled` at 0.89, and `has_access` against `has_no_access` at 0.95. Those are opposites, and a false merge destroys a real belief with no undo. Carrying a duplicate key is the cheaper error.

So Tier 2 generates candidates rather than deciding. It auto-merges only on a conjunction of evidence, refuses outright on a polarity conflict, and defers the uncertain band upward.

**Tier 3. Semantic reconciliation.** One call over the whole key list, offline.

```python
from sozograph import compact
compact(passport)
```

This is the only tier that catches `code_style: "minimal"` and `boilerplate_preference: "low"`. Same belief, no shared key, string similarity 0.51. No threshold reaches it.

Every merge is recorded in `passport.meta["dedupe"]`, so a decision that changed your memory can be audited after the fact.

## Benchmarks

The harness lives in [`bench/`](bench/) and targets LoCoMo, the benchmark [LightMem](https://github.com/zjunlp/LightMem) headlines.

```bash
pip install -e ".[openai,bench]"
python -m bench.locomo.run --data data/locomo10.json
```

It matches LightMem's published protocol: GPT-4o-mini as backbone and judge, the four non-adversarial question categories, per-conversation figures. Token and API-call counts are measured, not estimated. Memory construction and question answering run on separate provider instances so the two columns cannot contaminate each other.

The targets to beat, from LightMem's Table 3 ([arXiv:2510.18866](https://arxiv.org/abs/2510.18866)), per conversation:

| System | Accuracy | Memory tokens | API calls |
|---|---:|---:|---:|
| LightMem (0.8, 768) | 72.99% | 85.19k | 29.83 |
| A-MEM | n/a | 1,149.43k | 1,175.47 |
| Mem0 | 36.49% | 1,693.39k | 1,602.20 |

Those rows are cited, not re-run: reproducing them needs conda, LLMLingua-2, sentence-transformers, Qdrant, and several gigabytes of weights. Run the command above to produce the SozoGraph row on your own hardware. Results are written to `bench/results/` with the full per-conversation breakdown.

Check the dataset parsed before spending anything:

```bash
python -m bench.locomo.run --data data/locomo10.json --dry-run
```

## Compared to LightMem

|  | SozoGraph | LightMem |
|---|---|---|
| Install | `pip install sozograph` | clone, conda env, pre-download weights |
| Required dependencies | pydantic (6 packages, 23 MB) | LLMLingua-2, sentence-transformers, Qdrant or FAISS, SQLite |
| With one cloud provider | 18 packages, 49 MB | the above plus a backbone SDK |
| Model weights to download | none | a BERT compressor plus an embedding model |
| Configuration | one string: `"openai:gpt-4o-mini"` | a nested config dict |
| Where memory lives | a JSON file | a vector database |
| Move memory between machines | copy the file | export and reindex |
| Providers | Anthropic, OpenAI, Gemini, Ollama, LiteLLM, LangChain | OpenAI, DeepSeek, Ollama, vLLM, Transformers |

Both are good at the same job. The difference is what you have to install and what you can do with the result.

## Passport format

```json
{
  "version": "2.0",
  "updated_at": "2026-03-11T09:04:00+00:00",
  "user_key": "u_123",
  "facts": [
    {"key": "location", "value": "Kwekwe", "ts": "...", "confidence": 0.95, "source": "seg_a1b2"}
  ],
  "prefs": [
    {"key": "tone", "value": "direct", "ts": "...", "confidence": 0.9, "source": "seg_a1b2"}
  ],
  "entities": [
    {"name": "SozoGraph", "type": "project", "aliases": ["Sozo Graph"]}
  ],
  "open_loops": [
    {"item": "Book the flight", "ts": "...", "source": "seg_a1b2"}
  ],
  "contradictions": [
    {"key": "location", "old": "Harare", "new": "Kwekwe",
     "ts_old": "...", "ts_new": "...", "source_old": "seg_9f", "source_new": "seg_a1"}
  ],
  "episodes": [
    {"id": "seg_a1b2", "ts": "...", "summary": "Melanie moved to Kwekwe for a new job.",
     "salience": 0.8, "source": "seg_a1b2",
     "participants": ["Melanie"], "keywords": ["kwekwe", "job"]}
  ],
  "sources": [
    {"id": "seg_a1b2", "kind": "chat", "ts": "...", "hash": "sha256:..."}
  ]
}
```

Changes are resolved by time. The newest value wins, and the change is recorded rather than discarded, so you can see what your agent used to believe.

## Determinism

The same inputs produce the same passport. Identifiers are SHA-256 of the content, never Python's `hash()`, which is salted per process and gave different ids on every run. Ordering is stable, so two passports built from the same history compare byte for byte.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SOZOGRAPH_PROVIDER` | auto | `"openai"`, `"anthropic:claude-opus-5"`, ... |
| `SOZOGRAPH_MODEL` | per provider | Override the model |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | | Whichever is set is used |
| `SOZOGRAPH_DEFAULT_CONTEXT_BUDGET` | `3000` | Characters per rendered context |
| `SOZOGRAPH_MAX_INTERACTION_CHARS` | `4000` | Truncation before extraction |
| `SOZOGRAPH_ENABLE_FALLBACK_SUMMARIZER` | `true` | Summarize unreadable database objects |

## Upgrading from 0.1.x

See [MIGRATING.md](MIGRATING.md). `ingest()` now returns a Passport rather than a tuple, and 0.1.x extraction silently discarded most of what it extracted, so expect noticeably fuller passports.

## Development

```bash
pip install -e ".[dev,all]"
pytest
ruff check src tests bench
```

The suite runs with no API key and no network. Providers are tested against fake transports.

## Licence

MIT
