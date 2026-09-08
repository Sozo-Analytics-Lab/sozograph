# SozoGraph

Portable JSON memory for LLM agents. Package **0.3.2**, passport format **2.2**.

One JSON object holds beliefs, preferences, observations, episodes, entities,
open loops, dated changes, and optional source evidence. Copy the file between
machines or put the same object in a database. Offline reading, ranking, and
saving need only pydantic. No required provider SDK, embeddings, weights,
vector database, or service.

```python
from sozograph import SozoGraph, Passport
sg = SozoGraph("openai:gpt-4o-mini")
p = sg.ingest([{"id": "turn-1", "speaker": "Alice", "text": "I moved to Kwekwe.",
                "ts": "2026-09-07T10:00:00Z"}])
print(p.context(query="Where does Alice live?"))
p.save("alice.json")
p = Passport.load("alice.json")
```

## Install and providers

`pip install sozograph` installs the core. Optional extras: `anthropic`, `openai`,
`gemini`, `ollama`, `litellm`, and `langchain`; for example,
`pip install "sozograph[openai]"`. SDKs are lazy-loaded. Native structured output
is used where supported, with provider-specific gateway fallbacks.

```python
SozoGraph()                              # environment selection
SozoGraph("anthropic")
SozoGraph("ollama:llama3.2")
SozoGraph("openai:my-model", base_url="http://localhost:8000/v1")
```

Set SOZOGRAPH_PROVIDER, SOZOGRAPH_MODEL, and the selected provider API key.
Existing provider instances, including the LangChain adapter, also work.

## Ingestion and replay

Chat turns, transcripts, Firestore objects, RTDB envelopes, and Supabase rows
are normalized and segmented. Oversized turns split with parent offsets and
bounded overlap. Saturated output, invalid records, and failures can trigger
smaller repair segments.

```python
forecast = sg.plan(history, retention="full", max_extra_calls=8)
p = sg.ingest(history, passport=p, retention="full", max_extra_calls=8,
              checkpoint=lambda memory: memory.save("checkpoint.json"))
print(p.ingest_report)
```

The return type remains Passport. stats and ingest_report are runtime diagnostics;
completion receipts persist in meta.ingest. The same successful units with the
same extraction configuration skip provider calls on replay. reextract=True
requests fresh extraction; extraction_revision identifies caller configuration
changes. Check ingest_report["complete"] because exhausted repairs leave partial
memory. max_extra_calls bounds extraction invocations, not SDK transport retries.
Planning figures are estimates. Supply stable turn/session IDs and timestamps;
an optional clock enables repeatable tests. Unknown source time is not event time.

## Evidence retention

| Mode | Source content | Tradeoff |
|---|---|---|
| none default | Hashes and turn IDs | Smallest; discarded details cannot be recovered |
| excerpts | Locally aligned excerpts | Partial coverage; overlap matches are candidates |
| full | Complete normalized turns and split parts | Recovery from extraction omissions; grows with history |

Exact evidence identifies a verbatim span, not proof of truth. Candidate alignment
is lexical and can be wrong. No mandatory generated quote fields are added.

## Explainable recall

```python
result = p.recall(query="Alice activities", budget_chars=3000)
print(result.context)
print([(r.score, r.components) for r in result.selected])
print([(r.section, r.reason) for r in result.dropped])
p.context(query="gate code", include_sources=True)
p.context(query="Alice family activities", graph=True)
p.recall(subject="Alice", date_from="2026-02-01", date_to="2026-02-28")
p.recall(budget_tokens=800, token_counter=my_token_counter)
```

BM25, explicit entity identity, and time/confidence priors rank candidates once;
a greedy packer selects complete records within the budget. No-query rendering
prioritizes the profile. Section caps and list-subject quotas bound selection.
Chronological queries display dated observations in order. Graph expansion is
off by default. Weights are engineering defaults, not calibrated confidence or a
demonstrated optimum. Unicode/CJK features improve lexical coverage; multilingual
accuracy remains unmeasured. Headers that cannot fit yield an empty context.
Memory text is marked as data and controls are escaped; this is not a complete
prompt-injection defense.

## Changes and forgetting

Facts/preferences can carry a subject. Newer keyed values update beliefs;
equal-time alternatives stay disputed. Observation dedupe preserves event dates,
negation, quantities, and ordered content. Month/year dates become intervals.

```python
p.set_loop_status("Book flight", "completed", subject="Alice")
p.forget(source_ids=["segment-id"])
p.redact("sensitive phrase")
print(p.audit())
```

Erasure conservatively removes connected source groups and dependent records,
including evidence. It may delete unrelated details in a shared source. Audit
metadata and unknown extensions are cleared. It cannot revoke external copies or
prevent reingestion. Subject filtering is not tenant authorization; use separate
passports per user.

## Portable contract

[JSON Schema](schemas/passport-2.2.schema.json), [migrations](MIGRATING.md), and
[release notes](CHANGELOG.md) describe the contract. Versions 1.0, 2.0, and 2.1
migrate explicitly. Unknown fields remain at their original level. Unsupported
future versions round-trip opaquely; memory operations reject them.

canonical_json() excludes the operational update timestamp and designated failure
metadata; content_hash() hashes that representation. This supports fixed-input
replay, not deterministic LLM output or distributed convergence. Saving uses
unique temporary files and atomic replacement. The file API remains single-writer.

## Evaluation and development

```bash
python -m bench.evaluate --dataset locomo --data data/locomo10.json --dry-run
python -m bench.evaluate --dataset locomo --data data/locomo10.json
python -m bench.evaluate --dataset longmemeval --data data/longmemeval_s_cleaned.json
python -m bench.profile
pip install -e ".[dev]"
pytest
ruff check src tests bench scripts
python scripts/export_schema.py
```

See [benchmark instructions](bench/README.md) for independently cached construction,
recall, predictions, and grading; separate extractor/answerer models; and lexical
and full-context baselines. Gold annotations never enter history. Coverage measures
source membership, not answer presence. Official LongMemEval scoring can consume
the exported hypotheses; the included equivalence judge is separately labeled.

Historical 0.3.1 local 8B LoCoMo results were 25.39% QA accuracy versus 47.66%
full-context accuracy at roughly one tenth of the token cost. These are not 0.3.2
results. Version 0.3.2 has offline regression and timing validation; no new live
QA accuracy or SOTA claim is made. See the implementation report in
docs/releases/0.3.2/ for measurements, mathematics, and limitations.

MIT license.
