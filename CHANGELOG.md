# Changelog

## 0.4.0 (Visa)

SozoGraph becomes the portable memory layer for agents: one passport, a visa per agent, every write a stamp.

- **Passport 3.1.** An optional stamp log becomes canonical when present. Sections are its deterministic projection, folded through the 2.2 resolver. Genesis stamps migrate any snapshot losslessly.
- **Stamp operations**: assert, supersede, correct, retract, expire, confirm, forget. Content-addressed ids, hybrid logical clocks, a Merkle head.
- **Merge as a CRDT.** Union of stamps with redaction and quarantine on top of the payload lattice. Replicas converge in any order; a forget survives every stale merge.
- **Competence.** A Beta posterior per agent and domain, seeded from Passport Control write scores, moved by holder confirmations and corrections, deciding concurrent writes.
- **Shelf life.** Weibull survival by kind of claim; upcoming, current, stale and ended states in views; censored refits from retractions.
- **Admission gate.** Secrets rejected; protected attributes, instructions in third-party content and lower-trust writes against higher-trust claims quarantined until the holder confirms.
- **Visas.** Domains, write and forget permissions, sensitivity ceiling, expiry, revocation; minimum-necessary views priced by sensitivity; tokens stored as hashes.
- **MCP server** (`sozograph serve`): stdio for local agents; Streamable HTTP with an OAuth 2.1 visa office (PKCE, dynamic registration, resource-bound tokens, refresh rotation, revocation); REST mirror with OpenAPI; ChatGPT-style search and fetch; file and Firestore stores; audit log; HMAC attestation.
- **CLI** (`sozograph`): init, upgrade, visa, stamp, recall, pending, confirm, dismiss, forget, history, merge, sync, handoff, return, import, verify, trust, schema, hook, serve.
- **Agent Skill** `sozograph-passport`, a **Claude Code plugin** and marketplace entry, a **Muse connector pack**, and setup guides for Claude, Grok, Gemini, ChatGPT and NotebookLM.
- **Handoff packets** and return forms for agents that read files; **importers** for Claude, ChatGPT and Gemini memory, CLAUDE.md, Letta Agent File and OpenMemory.
- **Passport Relay**, Passport Control played by several agents on one passport, and a **conformance suite** of golden fixtures.
- **Fix:** timestamps equal to the current time now survive serialization.
- **Fix:** event-time filters ignore discussion time for undated observations.

## 0.3.2

- Passport 2.2 with extensible storage, explicit migration, opaque future-format
  transport, JSON Schema, canonical hashes, and atomic single-writer saves.
- Distinct source identities, overlapping turn splitting, per-turn timestamps,
  bounded repair, ingest reports, completion receipts, and checkpoints.
- Optional source evidence with exact/candidate alignment and full retention.
- Scoped beliefs, equal-time disputes, conservative observation deduplication,
  provenance unions, dated history, and compaction guards.
- Explainable recall, whole-record budgeting, aliases, Unicode/CJK tokenization,
  temporal intervals, optional graph/source expansion, and erasure/integrity APIs.
- Resumable LoCoMo/LongMemEval evaluation, independent models and stage caches,
  source coverage, hypotheses export, paired bootstrap, and offline profiling.
- Historical tests updated for intentional version, date, graph-default, and
  conservative paraphrase behavior changes. No new core dependencies.
- No live 0.3.2 QA accuracy claim; not published to PyPI by this task.
