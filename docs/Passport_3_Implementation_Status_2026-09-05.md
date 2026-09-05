# Passport 3 Implementation Status

Date: 5 September 2026

The foundation and direct-ingestion tranches are complete. Passport 3 runs beside Passport 2.1. Existing ingest, save, load, render, and provider paths still pass their tests.

## Completed in this session

- [x] Added a governed Passport 3 record model.
- [x] Added fact, preference, observation, episode, entity, open loop, procedure, outcome, and relation record kinds.
- [x] Added deterministic canonical JSON.
- [x] Added SHA-256 logical record IDs.
- [x] Added separate revision hashes for changing record content.
- [x] Added exact evidence spans with character offsets, UTF-8 byte offsets, source hashes, quotes, locators, and verification.
- [x] Added half-open valid-time intervals.
- [x] Added transaction-time queries over event history.
- [x] Added active, disputed, superseded, retracted, and deleted lifecycle states.
- [x] Added authority, sensitivity, and scope fields.
- [x] Added an append-only event ledger with Lamport clocks and causal parents.
- [x] Added deterministic replica merge and event deduplication.
- [x] Added monotonic tombstones that block stale resurrection.
- [x] Added policy filtering before search and prompt rendering.
- [x] Added field-aware BM25F retrieval.
- [x] Added an explicit prompt data boundary.
- [x] Added HMAC-SHA-256 integrity signatures.
- [x] Added atomic Passport 3 save and load.
- [x] Added offline migration from Passport 1 and Passport 2 snapshots.
- [x] Preserved future fields at their original level in Passport 2 and Passport 3 round trips.
- [x] Exported the Passport 3 API from the package root.
- [x] Added direct `SozoGraph.ingest_v3()` projection from validated extraction output.
- [x] Added exact evidence-quote requests to the strict extraction schema.
- [x] Added deterministic exact-span anchoring with coarse evidence fallback.
- [x] Added replay-safe semantic hashes and revision lineage.
- [x] Added collision-resistant source identities for direct ingestion.
- [x] Added source-registry union during replica merge.
- [x] Added 27 Passport 3 invariant and integration tests.

## Verified

- [x] Full test suite: 188 passed.
- [x] Live provider tests: 4 skipped because they require provider credentials.
- [x] New and changed production code passes Ruff.
- [x] Existing Passport 2 behavior remains green.
- [x] Replica merge is commutative at the materialized state level.
- [x] Exact evidence verification handles Unicode byte boundaries.
- [x] Deletion survives stale replica merge.
- [x] Integrity verification fails after payload tampering.
- [x] Direct ingestion is idempotent across later transaction times.
- [x] Valid-time queries return the prior fact revision.
- [x] Direct ingestion failures are isolated and auditable.

## Engineering equations now represented in code

Logical record identity uses the SHA-256 digest of canonical identity fields. Revision identity uses the SHA-256 digest of canonical record content. Event order uses the tuple of Lamport clock, event time, replica ID, and event ID. Materialization is a deterministic fold over that order. BM25F combines weighted field frequencies before term saturation. These equations are documented beside their implementations.

## Remaining work

- [x] Write Passport 3 events directly from validated extractor output.
- [x] Capture exact evidence spans during extraction. Use coarse source locators when a quote cannot be verified.
- [ ] Extract relations, procedures, and outcomes from model output.
- [ ] Add calibrated contradiction and supersession policy.
- [ ] Add asymmetric signatures and envelope encryption.
- [ ] Add consent receipts, purpose binding, retention jobs, and user deletion workflows.
- [ ] Add disposable graph and dense retrieval sidecars.
- [ ] Add temporal query parsing and graph traversal.
- [ ] Add learned reranking and diversity selection.
- [ ] Add importers and exporters for external memory systems.
- [ ] Run LoCoMo, LongMemEval, and conflict-heavy portable memory evaluations.
- [ ] Add property tests, fuzz tests, and long-running replica convergence tests.
- [ ] Publish the versioned Passport 3 JSON Schema and migration guide.

## Current integration point

Use `SozoGraph.ingest_v3()` for direct event-ledger ingestion. Pass an existing `MemoryPassport` back into the method for incremental updates. Use `Passport.to_v3()` to upgrade an existing snapshot. The stable `SozoGraph.ingest()` method still returns Passport 2.1.
