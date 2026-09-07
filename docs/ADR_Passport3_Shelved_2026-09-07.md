# ADR: Passport 3 Shelved, v3.1 Stays on the Lean Line

Date: 2026-09-07
Status: Accepted, implemented on `v3.1-lean` as a revert of `ec69bdb`

## Context

`ec69bdb` (2026-09-05, shipped as 0.3.0) added Passport 3: an append-only
event ledger with bitemporal claims, exact evidence spans, deterministic
replica merge, monotonic tombstones, a pre-retrieval policy gate, field-weighted
BM25F, and HMAC signatures. A follow-up tranche on `bench-passport3` fixed a
regression the first version introduced (`evidence_quote` had become a
required extraction field, degrading the shared v2 path) and added a
disposable semantic sidecar: field vectors bound to record revision and
embedding model, fused with BM25F through weighted reciprocal rank fusion.

Both versions were measured on LoCoMo, same 8B backbone (Llama-3.1-8B Q4_K_M),
matched on the same conversations.

## Measured result

Required-`evidence_quote` regression, all 10 conversations:

| sozograph (v2 path) | Accuracy | temporal |
|---|---:|---:|
| before Passport 3 | 25.4% | 28% |
| with required evidence_quote | 21.0% | 14% |

Ablation ladder after the fix, matched on 4 conversations, same model:

| System | Accuracy | Abstain | multi-hop | temporal |
|---|---:|---:|---:|---:|
| v2.1, lean extraction | 26.1% | 51% | 10% | 28% |
| v3 ledger (BM25F + timeline) | 22.1% | 59% | 16% | 9% |
| v3 + semantic sidecar | 24.7% | 52% | 22% | 13% |

The ledger scored below plain v2.1 even after the regression fix. The semantic
sidecar closed some of that gap but never crossed it, and doing so required an
embedding model, turning "no embedding model" from an unconditional promise
into a conditional one.

## Diagnosis

v3's retrieval was rebuilt from zero inside a heavier data model and did not
inherit what made v2.1 work: per-section budget caps, the entity-lift-above-
cutoff logic in `rank_expanded`, the conjunction-guarded near-duplicate merge,
the timeline-vs-relevance query router. Those were tuned over several
benchmark cycles. v3's flat top-50 BM25F render had none of them. The
architecture got more sophisticated; the retrieval got less tuned. Net loss.

Two mechanisms plausibly compounded this on top of the missing tuning:

1. Evidence spans, valid-time/transaction-time bookkeeping, and policy fields
   add extraction surface a weak model has to fill on every item. That is
   attention spent on ledger bookkeeping instead of writing good atomic facts.
2. A second parallel data model means every future retrieval improvement has
   to be built and tuned twice, once per format, or the two formats drift.

## Decision

Revert `ec69bdb` on `main`. Passport 3's ledger, evidence linking, and
semantic sidecar are removed from the public API. The code is not deleted
from history: it is reachable on `bench-passport3` and in `ec69bdb` itself for
anyone who wants to pick the ledger direction back up later with better
tuning behind it.

v3.1 stays on one canonical passport: the pydantic-only, pure-Python-BM25
Passport 2.1 shape. "No vector database, no embedding model, no local weights"
returns to being unconditional.

What ships as 0.3.1:

- Lean extraction restored. `evidence_quote` is not a required or requested
  extraction field. This alone recovers 21.0% -> 26.8%(ish) on the matched set.
- Everything Passport 2.1 already had: atomic observations, entity-expanded
  retrieval, event-dated timeline rendering, conjunction-guarded near-duplicate
  merge, per-segment ingest isolation.

What was learned and stays as direction for the next lean iteration, not as
shipped code:

- **Evidence, optional and cheap.** `deterministic_quote()`'s exact-match and
  content-overlap scoring (from the shelved `evidence.py`) is worth
  reattaching directly to Fact/Preference/Observation/Entity as an optional
  field, filled with zero extra model calls, no ledger required.
- **A pure-Python entity graph for multi-hop.** Multi-hop is the weakest
  category everywhere measured (10-22%). `Entity.aliases` and
  `Observation.participants` co-occurrence is already-captured data; a plain
  dict-adjacency, one-or-two-hop expansion is the frontier literature's
  graph-propagation idea without a graph database or an embedding model.
- **Push `when` resolution harder.** Too many observations still carry an
  empty `when`. The rule needs to catch more implicit relative-time mentions.
- **Push extraction recall.** Still the largest lever by abstention rate.
  Denser few-shot examples and tighter segmentation, no new dependencies.

## Alternatives considered

- **Keep the ledger, unexported, as an experimental module.** Rejected. It
  would still carry maintenance weight and drift risk with no measured benefit,
  and "remove it, it's in git history" is honestly less work and less
  confusing to a reader of the tree than a half-hidden second data model.
- **Tune v3's retrieval to parity before deciding.** Considered, not chosen
  now. Real effort with no evidence yet that the ledger's added complexity
  would ever pay for itself on this benchmark; the lean line already beats it
  today. Revisit if a concrete need (multi-device sync, audit trail, deletion
  compliance) makes the ledger's actual differentiators load-bearing, not just
  theoretically nice.
- **Keep the semantic sidecar only, drop the ledger.** Rejected for 3.1. It
  still lost to plain v2.1 on this eval, and it break the no-embedding-model
  promise for a net accuracy loss. Worth another look once lean retrieval
  itself is closer to its ceiling and the marginal win from vectors is
  measured against a stronger baseline.

## Consequences

- 0.3.0 (PyPI) shipped with the required-evidence_quote regression. 0.3.1
  supersedes it. 0.3.0 should be yanked so new installs don't land on the
  degraded path.
- The Passport 3 planning document
  (`SozoGraph_Portable_Memory_Passport_Engineering_Plan.docx`) and its
  implementation status doc are removed from the working tree by this revert.
  They remain readable in `ec69bdb` and on `bench-passport3`.
- `PASSPORT_VERSION` in `schema.py` is unaffected: it stayed at `"2.1"`
  throughout the Passport 3 work, since that value is the JSON wire format's
  own version, not the package version. No passport files written under 2.1
  need migration.

## Verification

161 tests passed, 4 skipped, offline, ruff clean. `git revert ec69bdb` applied
without conflict, confirming nothing on `main` was built on top of the
Passport 3 commit before this decision landed.
