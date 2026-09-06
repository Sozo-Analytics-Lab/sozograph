# SozoGraph

Portable JSON memory for LLM agents.

Your agent's memory is a small JSON file. You can read it, diff it, email it, put it in Postgres, ship it to the browser. No vector database or embedding model is required. Passport 3 can add a disposable semantic sidecar when paraphrase recall matters.

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

SozoGraph compresses history into a belief state plus a recall layer:

**Facts and preferences.** What is true now, as keys and values. Small enough that every question sees all of it, so retrieval can never hide a fact.

**Observations.** Atomic, self-contained statements of what was said or happened, one per line, with the event date resolved from relative language ("last Friday" becomes a calendar date at write time). This is the recall layer a single-hop question reads from, and the difference between remembering that someone exists and remembering what they told you. Append-only: a record of what was seen, never overwritten.

**Episodes.** What happened over a stretch, as compact per-segment summaries with timestamps, produced by the same extraction call at no extra cost.

Everything but the belief state is ranked against the question, and ranking is BM25 in pure Python, widened by named-entity recall so a list answer draws on every record about its subject. Microseconds, no model, nothing to download.

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

### Passport 3 event ledger

Passport 3 is available as a preview path. It writes validated extraction
output directly into an append-only ledger with evidence, valid time,
transaction time, policy fields, revision lineage, and replica merge.

```python
passport = sg.ingest_v3(
    conversation,
    meta={"user_key": "melanie"},
    replica_id="laptop",
    authority="user",
    sensitivity="internal",
)

# Incremental ingestion updates the same ledger.
passport = sg.ingest_v3(new_turns, passport=passport)

# History and synchronization remain local and portable.
old_state = passport.materialize(valid_at=some_past_time)
merged = passport.merge(passport_from_phone)
merged.save("melanie.passport3.json")
```

`SozoGraph.ingest()` remains the stable Passport 2.1 path. Use
`Passport.to_v3()` for an offline snapshot upgrade.

Passport 3 uses lean extraction by default. Exact evidence is anchored from the
source after extraction. Use `evidence_linking="model"` when a governed ingest
should make one bounded evidence-only call for unresolved candidates.

### Optional semantic retrieval

The vector index is a rebuildable sidecar. Every entry is keyed to the record ID,
record revision hash, and embedding model ID. A stale revision cannot rank. Policy
filtering runs before dense ranking.

```python
from sozograph import SemanticSidecar

# `embedder` implements embed_query(), embed_documents(), and model_id.
sidecar = SemanticSidecar.build(passport, embedder)
records = sidecar.search_hybrid(passport, "Which physician did she mention?", embedder)

# Refresh only changed revisions after more ingestion.
sidecar.sync(passport, embedder)
sidecar.save("melanie.passport3.vectors.json")
```

Hybrid search combines BM25F with semantic field vectors through weighted
reciprocal rank fusion. Exact names, dates, and identifiers keep the lexical
path. Paraphrases gain the dense path. The passport stays usable when the
sidecar or embedding model is absent.

### Read

```python
passport.context()                                   # everything, budgeted
passport.context(query="what did she hang up?")      # episodes ranked by relevance
passport.context(budget_chars=1500)
passport.token_estimate()
```

The belief state (facts and preferences) is included in full while the budget allows. Atomic observations, the recall layer a single-hop question reads from, are ranked against the query and rendered under `Details recalled`. With a query, every section is ranked against it (BM25 blended with recency and confidence), so a cap cut drops irrelevant records rather than old ones.

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

### Results

A real run on a free local setup rather than GPT-4o-mini: Unsloth's `Llama-3.1-8B-Instruct-GGUF` (Q4_K_M), served by Ollama on a free Kaggle GPU, as the one model for memory, answering, and grading. All ten LoCoMo conversations, 1,540 questions.

| System | Accuracy | Tokens | Notes |
|---|---:|---:|---|
| full_context, this 8B model | 47.66% | 33.1M | whole conversation in every prompt |
| **sozograph** | **25.39%** | **3.3M** | 10x fewer tokens than full_context |
| LightMem, GPT-4o-mini (published, reference only) | 72.99% | 85.19k/conv | different, stronger backbone |

On a matched six-conversation slice, the atomic observation layer doubled sozograph's accuracy over the belief-state-only version (13.22% → 26.67%). Across all ten it holds at **25.4%, at one tenth of full_context's token cost.** By category it is uneven, and the shape is the point:

| Category | sozograph | full_context |
|---|---:|---:|
| single-hop | 29% | 66% |
| **temporal** | **28%** | 23% |
| multi-hop | 16% | 30% |
| open-domain | 20% | 20% |

**Temporal recall beats full_context**, and not by accident. Observations carry an event date (`when`) that the extractor resolves from relative language, so "last Friday" becomes `2022-01-14` at write time; a temporal query then reads a dated, sorted timeline. full_context has to find "last Friday" in the raw transcript and resolve it at read time, and often does not. sozograph answers 57 temporal questions correctly that full_context gets wrong. The two systems are partly complementary: their union answers 57% of all questions.

**How the layer was found.** The belief-state-only run answered "Not mentioned" to 68% of questions, against 17% for full_context. It was not reasoning badly; the answer was not in the rendered memory. The extractor kept "beliefs, not quotes... only what is stable or actionable," so the incidental detail a single-hop question asks for ("the hike took two hours", "he hid the bone in a slipper") was discarded before it could be retrieved. Extraction now emits **atomic observations** alongside the belief state: self-contained third-person statements, one per line, relative dates resolved, append-only, ranked against the question by the same pure-Python BM25 and rendered under `Details recalled`. This is the portable form of the atomic-fact memory that leads the LoCoMo recall benchmarks (AtomMem, Mem0), with no embedding and no vector store. Four optimizations followed, each holding the portability line: entity-expanded retrieval for multi-hop lists, the event dates above, a conjunction-guarded near-duplicate merge, and a pinned refusal string for clean grading. See [`ADR_Atomic_Observations_2026-08-25.md`](docs/ADR_Atomic_Observations_2026-08-25.md), the [autopsy](docs/Autopsy_Atomic_Observations_LoCoMo_2026-08-26.docx), and the [full-10 post-mortem](docs/PostMortem_Full10_LoCoMo_2026-08-26.docx).

**Two gaps remain, and both point the right way.** The larger is extraction recall: sozograph still abstains on 44% of single-hop questions because the 8B does not write down every fact. That is the coverage ceiling the memory literature names first, and it lifts most with a stronger extractor. The other is the backbone itself: full_context reaches only 47.66% on this 8B, while GPT-4o-mini scores ~73% on LightMem's own table. Much of the residual is the model, not the architecture, and that is a tailwind. SozoGraph is a thin, portable layer over whatever model you bring, so every stronger model lifts it for free, no reindex and no migration.

Read 25.4% as directional. The architecture is sound; the next honest number needs a stronger backbone behind it.

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
  "version": "2.1",
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
  "observations": [
    {"text": "Melanie adopted Oliver from the Kwekwe shelter.",
     "ts": "...", "when": "2026-01-04", "source": "seg_a1b2", "participants": ["Melanie"]}
  ],
  "sources": [
    {"id": "seg_a1b2", "kind": "chat", "ts": "...", "hash": "sha256:..."}
  ]
}
```

Changes are resolved by time. The newest value wins, and the change is recorded rather than discarded, so you can see what your agent used to believe. Observations are the one append-only section: they record what was seen and are never overwritten, only deduplicated.

## Determinism

The same inputs produce the same passport. Identifiers are SHA-256 of the content, never Python's `hash()`, which is salted per process and gave different ids on every run. Ordering is stable, so two passports built from the same history compare byte for byte.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SOZOGRAPH_PROVIDER` | auto | `"openai"`, `"anthropic:claude-opus-5"`, ... |
| `SOZOGRAPH_MODEL` | per provider | Override the model |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | | Whichever is set is used |
| `SOZOGRAPH_DEFAULT_CONTEXT_BUDGET` | `6000` | Characters per rendered context |
| `SOZOGRAPH_MAX_INTERACTION_CHARS` | `4000` | Truncation before extraction |
| `SOZOGRAPH_ENABLE_FALLBACK_SUMMARIZER` | `true` | Summarize unreadable database objects |

## Upgrading from 0.1.x

See [MIGRATING.md](MIGRATING.md). `ingest()` now returns a Passport rather than a tuple, and 0.1.x extraction silently discarded most of what it extracted, so expect noticeably fuller passports.

## Examples

[Recipe Mentor](https://github.com/rapha18th/recipe-mentor) is a full
worked application: a "Collaborative Partner" agent that walks a builder
through a real ML production recipe across two different projects, and
recalls what tripped the user up on one, unprompted, when they start the
other — persistent memory (a Passport), deterministic cross-session recall
with no vector store, and a real LLM judge. Kept as its own repo rather
than in-tree, since it pulls in dependencies (an agent framework, a cloud
persistence layer, ML training libraries) that have nothing to do with
what SozoGraph itself needs to run.

## Development

```bash
pip install -e ".[dev,all]"
pytest
ruff check src tests bench
```

The suite runs with no API key and no network. Providers are tested against fake transports.

## Licence

MIT
