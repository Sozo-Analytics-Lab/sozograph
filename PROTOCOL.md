# The SozoGraph passport protocol, format 3.1

A passport is one person's memory, shared by every agent they use. The holder owns it. Each agent holds a visa. Every write is a stamp. This document specifies the file, the stamp operations, the projection from stamps to beliefs, the admission gate, visas and merging, so that any implementation in any language reads and writes the same passports. The fixtures in `conformance/fixtures` are normative.

## 1. The file

A passport is a UTF-8 JSON object.

| Field | Meaning |
|---|---|
| `version` | `"3.1"` |
| `holder` | `{"id", "name", "issued_at"}`. A holder record marks the passport as log-backed |
| `stamps` | The stamp log, the source of truth |
| `head` | SHA-256 Merkle root over the sorted stamp ids (section 5) |
| `agents` | Agent registry keyed by agent id: `name`, `vendor`, `model`, `home_domains`, `first_seen`, `last_seen` |
| `visas` | Visas (section 7). Tokens appear only as `token_sha256` |
| `facts`, `prefs`, `entities`, `open_loops`, `episodes`, `observations`, `contradictions` | The projection of the stamp log (section 4), in the Passport 2.2 record shapes plus the governance fields below |
| `sources` | Optional evidence: source segments and retained turns |
| `meta` | Operational metadata. Implementations preserve unknown keys |

Every projected record may carry governance fields: `stamp_ids`, `author`, `origin`, `domains`, `kind`, `valid_from`, `valid_until`, `sensitivity`. Empty values stay off the wire.

A file with stamps or a holder record is **log-backed**: the stamps are canonical and the sections are derived. A 3.1 file with neither is a plain snapshot with 2.2 semantics. Readers migrate 1.0, 2.0, 2.1 and 2.2 files by version bump, and turn them into genesis stamps (section 4.6) on the first write. Version `3.0` names a shelved preview: readers round-trip it unchanged and refuse to mutate it.

## 2. Stamps

```json
{"id": "st_9f2c...", "op": "supersede", "section": "facts",
 "claim": {"subject": "", "key": "home_city", "value": "Mutare", "valid_from": "2027-03-14",
           "kind": "residence", "domains": ["general"], "source": "claude-code", "ts": "2027-01-26T12:00:00Z"},
 "targets": ["st_41aa..."], "author": "claude-code", "visa": "visa_8c1e...", "origin": "user_direct",
 "hlc": "1801137600000-00002-local", "seen": "1801137600000-00001-local",
 "evidence": [{"quote": "we're relocating to Mutare, moving day is 14 March"}],
 "admission": "accepted", "sig": "hmac-sha256:server:..."}
```

| Field | Meaning |
|---|---|
| `op` | `assert`, `supersede`, `correct`, `retract`, `expire`, `confirm`, `forget` |
| `section` | The section the claim belongs to |
| `claim` | The record, in its section's shape |
| `targets` | Stamp ids this stamp acts on |
| `author` | The writer: an agent id, `holder`, `import:<vendor>`, or `sozograph:extractor:<provider>` |
| `origin` | Who the information came from (section 6) |
| `hlc` | Hybrid logical clock reading `<wall ms, 13 digits>-<counter, 5 digits>-<node>`; lexicographic order is clock order |
| `seen` | The latest `hlc` the writer had observed |
| `evidence` | Free-form supporting quotes |
| `reason` | A short note; erasure stamps say only `holder request` |
| `erase_sources` | Source ids a forget stamp removes |
| `admission` | `accepted` or `quarantined`, set by the receiving replica (section 6) |
| `sig` | Attestation: `hmac-sha256:<key id>:<hex>` or `ed25519:<key id>:<base64url>` over the stamp id |
| `redacted` | `true` once a forget stamp erased this stamp's payload |

### Operations

| Op | Claim | Effect |
|---|---|---|
| `assert` | required | A new claim |
| `supersede` | required | The targets stop being current at this claim's time and stay in history |
| `correct` | optional | The targets were false from the start and leave history. A claim, when present, is asserted |
| `retract` | empty | The targets stop being true and leave the slot empty |
| `expire` | `valid_until` | The targets end on that date |
| `confirm` | empty | Holder only. A quarantined target is admitted with origin `user_confirmed`; an accepted target gains competence |
| `forget` | empty | The targets' payloads are erased everywhere (section 5) |

## 3. Identity

`id` is `st_` followed by the first 32 hex digits of SHA-256 over the canonical JSON of the stamp with `id`, `admission`, `admission_reason`, `sig` and `redacted` removed. Canonical JSON sorts keys, uses `,` and `:` separators, keeps non-ASCII characters, and rejects NaN. A redacted stamp keeps its original id.

## 4. Projection

The projection is a pure function of the stamp log and the agent registry. Every replica holding the same stamps computes the same sections.

### 4.1 The live set

A stamp is **admitted** when its admission is `accepted`, or when an accepted holder `confirm` names it. Among admitted stamps in clock order:

- targets of `correct` are **corrected**,
- targets of `retract` are **retracted**,
- targets of `forget`, and every redacted stamp, are **forgotten**,
- targets of `supersede` are **superseded**,
- targets of `expire` take that stamp's `valid_until`.

The **live** stamps are the admitted `assert`, `supersede` and `correct` stamps with a nonempty claim that are neither corrected, retracted nor forgotten. Superseded stamps remain live for history and leave the current belief.

The **claim time** of a stamp is its claim's `valid_from`, else its claim's `ts`, else its clock reading.

### 4.2 Facts and preferences

A slot is `(casefold(subject), normalized key)`. Within a slot, the current stamps (live stamps outside every supersede's targets) group into value classes. A class `B` **dominates** a class `A` when `B`'s latest claim time is strictly later and either `B`'s highest origin rank is at least `A`'s, or `B`'s highest origin rank is at least `imported`. Higher-trust claims therefore outrank newer third-party and tool-result claims.

The **frontier** is the set of undominated classes. One frontier class is the belief. With several, each class scores

```
score = logit(competence(author, first domain of the claim)) + w(origin)
```

taking the best member. Classes within `DELTA = 0.5` of the best are disputed; the rest stay in history. Origin weights: `user_direct` and `user_confirmed` 3, `agent_inferred` 0, `imported` -0.25, `tool_result` -0.5, `third_party` -1.

Current and older classes then fold, in claim-time order, through the Passport 2.2 resolver: equal values merge provenance and keep the latest time; a later value replaces an earlier one and records a contradiction from the old value to the new; equal-time alternatives become disputed. A superseded stamp records a contradiction from its value to its superseder's.

### 4.3 Other sections

- **Open loops**: slot `(casefold(subject), normalized item)`; the latest claim time wins.
- **Entities**: fold in clock order by name and alias, unioning aliases and provenance.
- **Episodes**: by id, the latest stamp in clock order wins.
- **Observations**: fold in claim-time order with 2.2 deduplication: exact text, or a guarded paraphrase with equal participants and event dates.
- **Contradictions**: imported contradiction stamps pass through; projection adds the changes from 4.2. Duplicates collapse.

### 4.4 Governance fields

Each projected record carries the ids of the stamps behind it, the author of the latest, the highest origin among them, the union of their domains, the latest `kind`, `valid_from` and `valid_until`, and the highest sensitivity.

### 4.5 Competence

Each `(agent, domain)` holds a Beta posterior. The prior is `kappa = 4` pseudo-observations at `q`: the agent model's Passport Control write score when the registry names a measured model, else 0.85 in the agent's home domains and 0.6 elsewhere; the holder is 0.99. Each accepted holder `confirm` of an agent's stamp adds a success in the stamp's domains. Each accepted holder `correct` or `retract` of an accepted agent stamp adds a failure. Dismissing a quarantined stamp leaves competence unchanged.

### 4.6 Genesis

Turning a snapshot into a log writes one `assert` stamp per record, in section order `facts, prefs, entities, open_loops, episodes, observations, contradictions`, with author `import:passport`, origin `imported`, the record as its claim (minus `stamp_ids`), and the clock reading `<record time in ms>-<running counter>-genesis`. Projecting the genesis log reproduces the snapshot's sections.

## 5. Merge and erasure

A log maps stamp ids to payloads. Merging two logs takes the union of ids and joins each payload:

- a redacted copy beats an unredacted one,
- `quarantined` beats `accepted`,
- the larger signature string wins.

After the union, every target of an accepted `forget` stamp is redacted, and the sources its `erase_sources` names are removed. Redaction clears `claim`, `evidence` and `reason` and keeps the id, op, targets and clock.

Union and these joins are commutative, associative and idempotent, so replicas converge in any delivery order, and a forget made on one replica stays forgotten after a merge with a stale copy. Agent records merge by id (earliest `first_seen`, latest `last_seen`). Visas merge by id with the earliest `revoked_at`.

The **head** is the SHA-256 Merkle root over the sorted, deduplicated stamp ids: leaves are SHA-256 of each id; an odd level repeats its last node; the root of an empty log is SHA-256 of the empty string.

## 6. Admission

Origins rank `third_party` 0, `tool_result` 1, `imported` 2, `agent_inferred` 3, `user_confirmed` 4, `user_direct` 4. Agents may write `user_direct`, `agent_inferred`, `tool_result` and `third_party`; `user_confirmed` belongs to the holder and `imported` to importers.

A receiving replica decides each incoming stamp, in order:

1. **Rejected** when the claim or evidence holds a secret: a card number that passes Luhn, a key or token format, a private key, a password, or a credential-named attribute. Rejected stamps stay out of the file.
2. **Rejected** when the visa is inactive or read-only, when a `forget` lacks forget permission, when an agent sends `confirm`, when claim domains or a target lie outside the visa, or when a target is unknown.
3. **Quarantined** when a non-holder stamp targets a stamp of higher origin rank.
4. **Quarantined** when a non-holder `forget` carries an origin other than `user_direct`.
5. **Quarantined** when a claim of rank below 4 writes a protected attribute: contact details, postal address, payment and banking, identity numbers, credentials or standing instructions to assistants. Third-party and tool-result values holding an email address, phone number or IBAN count as protected.
6. **Quarantined** when a claim of rank below 4 holds an instruction aimed at assistants, or when a claim of rank 2 or below rests on evidence that does.
7. **Accepted** otherwise.

A hosted replica signs each admitted stamp. `verify` then flags every stamp whose signature fails.

## 7. Visas

```json
{"id": "visa_8c1e...", "agent": "muse", "domains": ["shopping"], "include_general": true,
 "write": true, "forget": false, "sensitivity_ceiling": 1, "purpose": "oauth:<client id>",
 "issued_at": "...", "expires_at": "...", "revoked_at": null, "token_sha256": "..."}
```

A visa's view holds the records whose domains meet the visa's (`general` included unless `include_general` is false; `*` sees every domain), whose sensitivity is at most the ceiling (0 public, 1 personal, 2 sensitive; 3 is rejected at admission), and which are still fresh. Records with empty domains or sensitivity take values inferred from their words. The view ranks records for the query and packs them to the budget, charging 0.05 per sensitivity level against relevance.

**Freshness.** A claim with `valid_from` in the future is upcoming. A claim past its `valid_until` has ended. Otherwise a claim of kind `k` survives `d` days after it became true with `S(d) = exp(-(d / eta_k) ^ gamma_k)`: current while S is at least 0.5, stale down to 0.2, ended below. Defaults (`eta` in days, `gamma`): mood 1, 2; situation 2, 2; symptom 7, 2; visit 10, 2; task 30, 1; project_state 365, 1; preference 1095, 1; employment 1461, 1; residence 1826, 1; relationship 3652, 1; identity and fact permanent. Bookings and trips end at `valid_until`, else after a 30-day scale.

**Tokens.** A bearer token `sgv_...` maps to a visa through its SHA-256. OAuth scopes mirror the visa: `passport:read`, `passport:write`, `passport:forget`, `domain:<name>`.

## 8. Transports

- **MCP**, stdio or Streamable HTTP, tools `passport_recall`, `passport_propose`, `passport_history`, `passport_handoff`, `passport_forget`, `search`, `fetch`; resource `passport://view`; prompt `passport_brief`.
- **OAuth 2.1** per the MCP authorization specification of 2026-07-28: protected resource metadata, authorization server metadata, dynamic client registration, PKCE, resource indicators bound to `/mcp`, refresh rotation and revocation. The consent page issues the visa.
- **REST**: `/v1/recall`, `/v1/stamps`, `/v1/history`, `/v1/handoff`, `/v1/forget`, `/v1/sync`, `/v1/records/{id}`, described at `/openapi.json`.
- **Files**: the visa copy (`sozograph-visa-copy/3.1`) and the return form (`sozograph-return/3.1`).

## 9. Conformance

An implementation conforms when, for every file in `conformance/fixtures`, it computes the expected head from the input stamps and the expected sections from the input stamps and agents, in any delivery order.
