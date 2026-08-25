# ADR: An Atomic Observation Layer for Single-Hop Recall

Date: 2026-08-25
Status: Accepted, implemented on branch `atomic-observations`, pending eval re-measurement

## Context

The post-`caba001` re-run (matched six LoCoMo conversations, local Llama-3.1-8B
backbone and judge) put sozograph at **13.22%** against full_context's **49.27%**
on the identical model. The query-aware-ranking and bounded-extraction changes
of `caba001` did not close the compression tax; the number moved slightly the
wrong way, inside the noise of a single run.

A failure-mode analysis of the saved `qa_log` located the gap precisely:

- sozograph answers **"Not mentioned" to 68.2%** of all questions; only 18.5%
  are attempted-but-wrong. full_context abstains on 17.4%.
- When sozograph *does* have the context, its wrong-answer rate (18.5%) is
  **lower** than full_context's (33.3%). It reasons fine; it lacks the evidence.
- Of the questions sozograph abstained on, full_context (same model, same
  question) answered **285 correctly** — 32.2% of all questions. The information
  was in the conversation and the pipeline dropped it. Fixing that coverage
  raises the ceiling to ~45.4%, essentially full_context's score.
- **226 of those 285 are single-hop.** The lost items are incidental episodic
  details: "Oliver hid his bone in Melanie's slipper", "the trail hike took two
  hours", "Tim is learning German".

The cause is the extraction layer, not retrieval. `EXTRACTOR_SYSTEM_PROMPT` told
the model to keep "beliefs, not quotes... only what is stable or actionable".
A one-off episodic detail is neither stable nor keyable, so it was never
extracted into any retrievable record. What little survived lived only inside a
one-to-three-sentence episode summary, which `_TRIM_ORDER` cut first. Retrieval
cannot surface a record that was never stored.

Budget starvation (6000 chars rendering ~7% of a 20k-token passport) is real but
secondary: with query-ranking, a matching record is ranked first and survives
even a tight budget. The dominant loss is that the matching record does not exist.

## Literature

The winning LoCoMo memory systems store granular, retrievable units rather than
a distilled belief state:

- **AtomMem** (arXiv:2606.19847) extracts *atomic facts*: self-contained,
  third-person statements with pronouns and relative dates resolved. It beats
  LightMem — SozoGraph's own comparison target — by +5.5% (multi-hop) and +31.1%
  (temporal). Its ablation ("AtomMem-Flat") shows a flat store of atomic facts
  retrieved by keyword/participant/temporal signals keeps most of the gain:
  "memory storage quality is paramount". The embedding is only one of three
  retrieval signals; the entity/event/temporal edges are symbolic.
- **Mem0** (92.5 on LoCoMo, 3-4x lower token cost than full-context) uses
  single-pass atomic extraction plus fused keyword+entity+semantic retrieval.

The portable subset of both — atomic statements retrieved by keyword and entity
signals — needs no embedding and no vector store. That is exactly what SozoGraph
already has in `retrieve.py`.

## Decision

Add an **observation** layer, inside the portability philosophy: pydantic-only
core, no vectors, no embeddings, no weights, one schema across every provider.

### 1. A new record type

`Observation(text, ts, source, participants)`. Atomic, self-contained,
third-person, append-only, never keyed. It carries a `search_text()` (statement
plus participants) and is ranked by the existing BM25. There is no belief to
overwrite, so the only merge is exact-text deduplication (participants union,
earliest timestamp kept), which keeps re-ingestion idempotent.

### 2. Extraction emits observations

`EXTRACTION_SCHEMA` gains an `observations` array of `{text}`, bounded by
`ARRAY_LIMITS["observations"] = 30` like every other array. The system prompt now
describes two layers: facts are the stable belief state; observations are the
recall layer, generous, one statement each, pronouns and relative dates resolved.
The model supplies only the statement text; participants come from the segment,
so the array cannot pad itself into the runaway loop the `maxItems` bound exists
to prevent.

### 3. Retrieval renders observations into the context

`render.py` gains a `Details recalled` section, query-ranked against the
question and trimmed late (`_TRIM_ORDER` places it after episodes/prefs and
before facts, floor 12). A single-hop question is answered from here.

## Alternatives considered

- **Loosen `facts` to hold episodic detail.** Rejected. Forcing "hid the bone in
  a slipper" into a snake_case key/value fragments the belief state and breaks
  contradiction resolution, which keys off `Fact.key`. Observations are keyless
  by design.
- **Just raise `--budget-chars`.** Necessary for multi-hop headroom, insufficient
  alone: it cannot render a record extraction never produced. Pair it with this,
  do not substitute it.
- **Embedding retrieval (AtomMem/Mem0 in full).** Rejected. It breaks the install
  promise for a corpus of hundreds of short records. BM25 + participant text is
  the portable subset and the ablation says it keeps most of the gain.
- **Store raw dialogue turns.** Rejected. That is full_context by another name;
  the point is a compressed, retrievable form.

## Consequences

- Extraction output roughly doubles per segment (facts + up to 30 observations),
  raising memory-phase tokens and latency on a weak model. Still far below
  full_context: the whole point is that observations are retrieved, not that they
  are all injected. The `maxItems` bound keeps the cost bounded and crash-free.
- Passport size grows. It remains plain JSON, diffable and portable; the new
  section is omitted when empty and old (2.0) passports load unchanged.
- `PASSPORT_VERSION` -> "2.1". `from_dict` is tolerant, so pre-observation files
  load with an empty list.

## Verification

152 passed, 4 skipped, offline, no API key. New coverage: observation
search_text, query-ranked rendering past the cap and under a tight budget,
append-only text dedupe with participant union, end-to-end extraction through the
fake provider, and a 2.0-passport backward-compat load. The two pre-existing
ruff B017 warnings in `tests/test_providers.py` remain untouched. The honest
next step is a re-run of the matched six conversations against the 13.22% and
16.61% baselines.
