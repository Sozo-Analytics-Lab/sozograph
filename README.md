# SozoGraph

The portable memory layer for agents. Package **0.4.0**, passport format **3.1**.

You use many agents. Muse shops, Grok markets, Dots write documents, Claude Code builds software, NotebookLM researches. SozoGraph gives them one memory of you. You hold the passport. Each agent holds a visa for the slice it needs. Every write arrives as a stamp with an author, an origin, a date and a shelf life. Code merges the stamps, gates them, and keeps every copy in agreement.

```python
from sozograph import Passport

p = Passport.issue("rairo", name="Rairo")
muse = p.issue_visa("muse", domains=["shopping"], write=True)
claude = p.issue_visa("claude-code", domains=["software"], write=True, model="claude-opus-5-5")

p.propose([{"attribute": "shoe_size", "value": "44", "origin": "user_direct"}], visa=muse)
p.propose([{"attribute": "deploy_target", "value": "Cloud Run", "origin": "user_direct"}], visa=claude)
p.propose([{"attribute": "invoice_address", "value": "billing@pay-scam.co", "origin": "third_party"}], visa=muse)

print(p.view(muse, query="running shoes").context)   # shopping and general records only
print(p.pending())                                   # the injected address waits for you
p.save("rairo.json")
```

The core needs only pydantic. A passport loads, merges, recalls and saves offline: one JSON file, readable anywhere.

## Why now

Passport Control measured 18 models as memory managers across 661 deterministic checks. The frontier writes memory at 0.95 to 0.99, and reading dated memory is easy for most models. The agents people use already understand the conversation. SozoGraph lets them write, and does the parts that need code: identity, merge, trust, time, scope and erasure.

## Connect your agents

```bash
pip install "sozograph[mcp]"
sozograph init ~/.sozograph/passport.json --holder rairo
claude plugin marketplace add Sozo-Analytics-Lab/sozograph && claude plugin install sozograph@sozograph
sozograph serve --http --issuer https://YOUR-HOST            # for Claude.ai, Grok, Gemini, ChatGPT, Muse
```

| Agent | Route |
|---|---|
| Claude Code, Codex, Gemini CLI, Grok Build | Local MCP over stdio, the plugin, the Skill |
| Claude.ai, Grok, Gemini app, ChatGPT and Dots | Custom connector to the hosted passport; OAuth consent issues the visa |
| Muse | Muse connector pack: MCP endpoint or OpenAPI spec |
| NotebookLM and any file reader | Handoff packet and return form |

[CONNECTORS.md](CONNECTORS.md) walks through each one. The [Agent Skill](skills/sozograph-passport/SKILL.md) teaches any Skills-capable agent how to write good stamps.

## Stamps

```json
{"op": "supersede", "target": "st_41aa...", "section": "facts",
 "claim": {"attribute": "home_city", "value": "Mutare", "valid_from": "2027-03-14", "kind": "residence"},
 "origin": "user_direct", "evidence": [{"quote": "moving day is 14 March"}]}
```

- **assert** a claim; **supersede** when something changed (history keeps the old value); **correct** when a record was false from the start (history drops it); **retract** when it stopped being true; **expire** on a known date; **forget** on the holder's request. **confirm** belongs to the holder.
- **Origin** says who the information came from: `user_direct`, `agent_inferred`, `tool_result`, `third_party`. The holder's own claims outrank newer third-party claims.
- **Kind** sets the shelf life. A headache fades in days, a home city lasts years, a booking ends at check-out.

## What code guarantees

- **Convergence.** The stamp log is a CRDT. Any replicas holding the same stamps hold the same beliefs, in any merge order. A forget survives a merge with a stale copy.
- **Admission.** Secrets are rejected for every writer. Protected attributes (contact details, addresses, payment and banking, identity numbers, standing instructions), instructions inside third-party content, and lower-trust writes against higher-trust claims wait in quarantine for the holder.
- **Competence.** Concurrent writes resolve by each agent's competence in the claim's domain: a Beta posterior seeded from Passport Control scores and moved by the holder's confirmations and corrections.
- **Scope.** Each view holds only the visa's domains and sensitivity ceiling, packed to a budget that charges for sensitivity.
- **Integrity.** Content-addressed stamps, a Merkle head, HMAC or Ed25519 signatures, and `verify`.

[PROTOCOL.md](PROTOCOL.md) specifies the format and the projection. [SECURITY.md](SECURITY.md) states the threat model. `conformance/fixtures` lets any implementation prove it reads passports the same way.

## Command line

```bash
sozograph visa issue rairo.json --agent notebooklm --domains research --write
sozograph handoff rairo.json --token sgv_... --out research.md      # upload to NotebookLM
sozograph return rairo.json reply.md --token sgv_...                # bring its findings home
sozograph import rairo.json chatgpt saved-memories.txt
sozograph pending rairo.json && sozograph confirm rairo.json st_...
sozograph history rairo.json --key home_city
sozograph merge merged.json laptop.json hosted.json
sozograph verify rairo.json
```

## Extraction from transcripts

Agents that hand back plain text, and histories you want to import, go through extraction with any provider:

```python
from sozograph import SozoGraph
sg = SozoGraph("anthropic:claude-opus-5-5")
p = sg.ingest(history, passport=p, retention="excerpts")
```

Extras: `anthropic`, `openai`, `gemini`, `ollama`, `litellm`, `langchain`. On a log-backed passport the extraction is written as stamps by `sozograph:extractor`, through the same gate. Snapshot passports keep the 2.x behaviour: bounded segmentation and repair, evidence retention (`none`, `excerpts`, `full`), explainable recall with whole-record budgets, dated history and erasure.

## Measuring it

```bash
python -m bench.relay --scripted door --scenarios 16
python -m bench.relay --chat anthropic:claude-opus-5-5 --inbox openai:gpt-6.1-sol --web gemini:gemini-3.1-pro
```

Passport Relay plays Passport Control with three agents on one passport. Offline, a policy that believes every injected email and tool result passes 26% of the injection checks with the gate off and 100% with it on, and the third-party values reaching protected beliefs fall from 16 to 0. The LoCoMo and LongMemEval harnesses remain under `bench/`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests bench scripts
python scripts/export_schema.py
python scripts/build_skill.py
```

MIT licence.
