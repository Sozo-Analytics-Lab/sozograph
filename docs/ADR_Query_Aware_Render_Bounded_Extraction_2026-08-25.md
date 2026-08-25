# ADR: Query-Aware Rendering, Bounded Extraction, Absolute Dates

Date: 2026-08-25
Status: Accepted, implemented on `main`

## Context

The matched LoCoMo run of 2026-08-24 (`docs/Loss_Analysis_LoCoMo_Local_8B_2026-08-24.html`) put sozograph at 16.61% against full_context's 47.34% on the identical 8B backbone. Same model, same judge. The 30.7-point gap is SozoGraph's own compression tax.

Two mechanisms produced most of that tax:

1. **Facts and preferences were invisible to ranking.** `render.py` passed `text_of=lambda x: ""` for facts, prefs, loops, and contradictions. Only episodes were ranked against the query. Past `caps.facts` (60), recency x confidence alone decided what survived, so an old, topic-specific fact could be dropped exactly when the question needed it. This lines up with the loss concentrating on multi-hop and temporal categories.
2. **Unbounded extraction arrays crashed four conversations.** Grammar-constrained decoding removes the model's "I'm done" signal at each array element. A weak model can choose "continue" indefinitely, restating near-duplicates until the completion cap kills the call mid-JSON. `num_predict` bounded the damage; nothing terminated the loop by construction.

A third, smaller loss: temporal questions failed on relative dates ("last Saturday") standing in for absolute ones. The extractor already receives the segment timestamp and never used it for this.

## Decision

Three changes, each inside the portability philosophy: pydantic-only core, no vectors, no embeddings, no weights, one schema across every provider dialect.

### 1. Every section is query-ranked

`search_text()` added to `Fact`, `Preference`, `Entity`, `OpenLoop`, and `Contradiction`. `_select` in `render.py` now feeds real text to BM25 for every section, blended with the recency x confidence prior through the existing `rank()`.

Below the caps and the budget, behavior is unchanged: sections render in full. Ranking only decides what survives a cut. Without a query, prior order alone decides, byte-identical to before. The old property "a retrieval miss can never hide a known fact" still holds below the caps; past them, the tradeoff is now explicit: an irrelevant recent record gets dropped before a relevant old one.

Zero new dependencies. BM25 already ran in microseconds over episodes; facts and prefs are shorter.

### 2. `maxItems` on every extraction array

New `ARRAY_LIMITS` in `prompts.py`, applied to `facts`, `prefs`, `entities`, `open_loops`, `aliases`, `participants`, `keywords`. OpenAI strict mode accepts `maxItems`; Gemini's `response_schema` accepts it; Ollama's `format` accepts it. One schema still serves every dialect, asserted by test against the strict-mode keyword subset.

`Extractor.validate()` truncates defensively for providers that ignore the bound. The loop failure now terminates by construction instead of by tuning sampling parameters.

Chosen bounds: facts 24, prefs 16, entities 12, open_loops 10, aliases 6, participants 8, keywords 10. Sized to a token-bounded segment (~1,400 turns of dialogue); generous relative to what a conservative extractor should emit.

### 3. Relative dates resolved at extraction

One rule added to `EXTRACTOR_SYSTEM_PROMPT`: the TIMESTAMP is "now"; every relative date or duration must become an absolute calendar date computed from it before being written anywhere. Prompt change only.

## Alternatives considered

- **Embedding-based fact retrieval.** Rejected. It breaks the install promise, adds weights to download, and a store to migrate, for a corpus of hundreds of short records.
- **Raise `caps.facts` until nothing is ever cut.** Rejected as the primary fix. It trades render budget for the same blind ordering. Worth pairing with `--budget-chars` headroom in the bench, but ordering had to become relevance-aware regardless.
- **Cap arrays per-provider** (Ollama-only bounds). Rejected. Divergent schemas per engine recreate the drift the shared schema exists to prevent.
- **Parse relative dates downstream, at render time.** Rejected. The information needed to resolve "last Saturday" lives in the segment, not the passport. Resolution belongs where the context is.

## Consequences

- With a query, section ordering within a rendered block is relevance order rather than recency order. Callers displaying the block see this.
- Extraction payloads from well-behaved providers are capped at the array limits. Passports holding more than 24 facts from a single segment were always noise.
- The 2026-08-24 eval numbers are now stale as a description of the library. The README limitation bullets record what was fixed. The honest next step is a re-run of the matched six conversations against the 16.61% baseline.
- Two ruff B017 warnings in `tests/test_providers.py` remain pre-existing and untouched.

## Verification

148 passed, 4 skipped, offline, no API key. New coverage: cap-survival regression (a matching old low-confidence fact beats 69 recent irrelevant ones), no-query fallback preserved byte-for-byte semantics, prefs/loops ranked, strict-mode subset walked recursively over the schema, every array bounded, date rule present.
