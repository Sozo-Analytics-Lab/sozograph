# ADR: A Co-Occurrence Entity Graph for Multi-Hop Retrieval

Date: 2026-09-07
Status: Accepted, implemented on `main`. LoCoMo measurement pending.

## Context

`ADR_Passport3_Shelved_2026-09-07.md` shelved the event-ledger direction and
named the next lean iteration's priorities. Multi-hop was the weakest
category everywhere measured on LoCoMo: 10-22% depending on the run, against
30-34% for full_context. The autopsy's concrete failures were mostly
incomplete lists and missed relationships: "What are Melanie's pets' names?"
(gold: Oliver, Luna, Bailey; answered "Oliver Bailey"), "What activity do
Audrey's dogs do at the park?" answered from one record when several existed.

`rank_expanded` (added earlier, `atomic-observations`/`fix-entity-expansion`)
already lifts every record whose text names an entity the query itself names.
That handles "What did Tim do this summer?" pulling in "Tim kayaked the river
gorge" (no lexical overlap with "summer", but Tim is named in both). It does
not handle a query that names a relationship rather than a name: "What does
Melanie do with her family?" never says "Caroline," so no record about
Caroline alone gets lifted, even if Caroline is Melanie's sister and several
observations establish that by naming them together.

## Decision

Add `EntityGraph`, a co-occurrence graph over names, in `retrieve.py`.

Two names get an edge whenever one record names both of them: an
observation's `participants`, an episode's `participants`. That is data the
extractor already writes on every segment; no new extraction field, no schema
change. Edge weight is how often the pair co-occurs. `neighbors(name, hops=1,
limit=6)` does a capped BFS and returns the strongest-connected names first.

`rank_expanded` takes an optional `graph` parameter. When the query names an
entity directly (the existing path, unchanged), its 1-hop co-occurrence
neighbors also get a lift, smaller than a direct match:

- Direct name match: `_ENTITY_MATCH_BONUS = 10.0`
- Graph-connected neighbor: `_GRAPH_NEIGHBOR_BONUS = 4.0`
- Neither: 0, ordinary BM25 plus prior

A direct match always outranks a graph-connected record, so a spurious edge
(two people who happened to be mentioned in the same segment without being
meaningfully related) can compete with an ordinary lexical match but can
never displace an actual name hit. With no query-named entity, the graph is
never consulted at all: `rank_expanded(graph=None)` is unchanged from before
this ADR, and the graph short-circuits before any lookup when `match_names`
returns nothing.

The graph is schema-agnostic like the rest of `retrieve.py`: `EntityGraph.build()`
takes plain lists of name strings, not `Observation`/`Episode` objects, so
it needs no import from `schema` and is directly testable with bare lists.
`render.py` supplies the groups (`_co_occurrence_groups`) and builds the
graph once per `export_context` call, outside the budget-trim retry loop,
since neither the vocabulary nor the graph depends on the caps being trimmed.

This is the pure-Python form of the frontier literature's graph-propagation
idea for multi-hop retrieval (HippoRAG's Personalized PageRank, Graphiti's
typed-edge graph): who is connected to whom, without a graph database, an
embedding model, or a learned ranker behind it.

## Alternatives considered

- **Full Personalized PageRank.** Rejected for now. The plan's own equation
  is a real random-walk-with-restart over typed edges; this passport's graphs
  are small (hundreds of nodes at most) and shallow (mostly single-hop
  relationships: family, coworkers, pets' owners), where a capped BFS gives
  nearly the same practical result as PPR at a fraction of the code and zero
  tuning surface (no alpha, no convergence tolerance). Revisit if a future
  eval shows 2-hop expansion earning its keep and a denser graph exposes PPR's
  actual advantage over BFS: better handling of hub nodes and weighted paths.
- **Persist the graph as a sidecar.** Rejected. Building it is a single pass
  over already-loaded records, cheap enough to redo on every render; a
  persisted index is one more thing that can drift from the passport it
  describes, which is exactly the failure mode Passport 3's sidecars were
  shelved for.
- **2-hop by default.** Rejected for the initial default. A friend-of-a-friend
  expansion risks pulling in an entire long conversation's cast for one query,
  especially once a "everyone was at the party" episode creates a fully
  connected clique. `neighbors(hops=...)` supports it; `_GRAPH_HOPS = 1` in
  `render.py` is the one place to change to try it once there is eval data to
  judge it against.

## Consequences

- `rank_expanded`'s signature grows two optional parameters (`graph`,
  `graph_hops`), both defaulted so every existing caller is unaffected.
- `_build()` in `render.py` gains two parameters (`vocabulary`, `graph`),
  hoisted out of the function and computed once in `export_context` rather
  than rebuilt on every budget-trim retry (up to 400 in the worst case) --
  incidentally fixing a pre-existing inefficiency in `_entity_vocabulary`,
  which used to be rebuilt on every retry too.
- Only observations are graph-widened for now, since that is where the
  autopsy's multi-hop failures concentrated (`Details recalled` is the
  section a list-completion question actually reads from). Entities and
  episodes could use the same widening later; not done here to keep the
  change scoped to the measured failure.

## Verification

174 tests passed (13 new: 11 for `EntityGraph`/`rank_expanded` in
`test_retrieve.py`, 2 integration tests through `export_context` in
`test_render.py`, including a negative control proving the graph contributes
nothing when the query names no entity), 4 skipped, ruff clean. Manually
verified end to end: a query naming only "Tim" surfaces two observations
about "Jen" that name neither Tim nor any query term, correctly ranked above
40 unrelated records; the same passport under an unrelated query surfaces
none of them.

LoCoMo accuracy impact is not yet measured. That is the next step: a Kaggle
run on the matched conversations, multi-hop category isolated, against the
16% baseline this ADR is meant to move.
