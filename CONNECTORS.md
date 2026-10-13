# Connecting agents to a passport

One passport, one visa per agent. Command-line agents run the passport server locally over stdio. Chat agents reach a hosted passport over Streamable HTTP with OAuth. Agents that only read files take a handoff packet.

| Agent | Home domain | Route |
|---|---|---|
| Claude Code | software | Plugin, or local MCP over stdio |
| Claude.ai and Claude Desktop | any | Custom connector to the hosted passport, plus the Skill |
| Grok | marketing | Custom connector; Remote MCP tools on the xAI API |
| Gemini app | any | Custom app from the MCP URL |
| ChatGPT and Dots | documents | Developer mode app |
| Muse | shopping | Muse connector (MCP or OpenAPI) |
| NotebookLM | research, decks | Handoff packet and return form |
| Codex, Gemini CLI, Grok Build, Muse Code | software | Local MCP over stdio |

## 1. Create the passport

```bash
pip install "sozograph[mcp]"
sozograph init ~/.sozograph/passport.json --holder rairo --name "Rairo"
```

Bring in what other assistants already know about you:

```bash
sozograph import ~/.sozograph/passport.json chatgpt saved-memories.txt
sozograph import ~/.sozograph/passport.json claude claude-memory.txt
sozograph import ~/.sozograph/passport.json claude-md ~/.claude/CLAUDE.md
```

Imports land with origin `imported`. Anything that looks like an instruction or a protected attribute waits in `sozograph pending`.

## 2. Local agents over stdio

**Claude Code, with the plugin** (Skill, session brief and server in one):

```bash
claude plugin marketplace add Sozo-Analytics-Lab/sozograph
claude plugin install sozograph@sozograph
```

**Claude Code, server only:**

```bash
claude mcp add sozograph -- sozograph serve --stdio --agent claude-code --domains software,documents
```

**Any MCP client** takes the same command: `sozograph serve --stdio --passport PATH --agent NAME --domains a,b`. The server issues a standing visa for that agent on first start and reuses it after.

## 3. Host the passport for chat agents

Chat agents reach remote servers over the public internet, so the passport needs an HTTPS address.

**Try it with a tunnel:**

```bash
sozograph serve --http --port 8765 --issuer https://YOUR-TUNNEL-HOST
```

The first start prints a holder secret once. Keep it: it opens the visa office and approves every connector. Set `SOZOGRAPH_HOLDER_SECRET` to choose your own.

**Run it on Cloud Run with Firestore:**

```bash
gcloud run deploy sozograph --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars SOZOGRAPH_ISSUER=https://YOUR-SERVICE-URL,SOZOGRAPH_PASSPORT=firestore://PROJECT/passports/rairo \
  --set-secrets SOZOGRAPH_HOLDER_SECRET=sozograph-holder:latest,SOZOGRAPH_SERVER_KEY=sozograph-key:latest
```

Firestore keeps the passport and the visa office state in two documents. Concurrent writes merge: the stamp log is a CRDT. `--allow-unauthenticated` lets connectors reach the OAuth endpoints; every passport route then checks a visa token.

**Endpoints**

| Path | Purpose |
|---|---|
| `/mcp` | MCP over Streamable HTTP, bearer visa token |
| `/.well-known/oauth-protected-resource/mcp` | Protected resource metadata (RFC 9728) |
| `/.well-known/oauth-authorization-server` | Authorization server metadata (RFC 8414) |
| `/register`, `/authorize`, `/token`, `/revoke` | OAuth 2.1 with PKCE and dynamic registration |
| `/visa-office` | The holder's page: pending stamps, visas, revoke all |
| `/v1/recall`, `/v1/stamps`, `/v1/history`, `/v1/handoff`, `/v1/forget`, `/v1/sync` | REST mirror |
| `/openapi.json` | OpenAPI 3.1 description of the REST mirror |

## 4. Chat agents

Each connector sends you to the visa office consent page. Pick the domains, write and forget permissions, sensitivity ceiling and expiry, enter the holder secret, and the agent receives its own visa.

- **Claude.ai and Claude Desktop.** Settings, Connectors, Add custom connector, paste `https://YOUR-HOST/mcp`. Then upload the Skill: run `python scripts/build_skill.py` and upload `build/sozograph-passport.zip` in Settings, Capabilities.
- **Grok.** grok.com/connectors, New Connector, Custom, paste `https://YOUR-HOST/mcp`. A grokbot on the xAI API attaches the same URL as a Remote MCP tool.
- **Gemini app.** Settings, Connected Apps, Add a custom app, paste `https://YOUR-HOST/mcp`.
- **ChatGPT and Dots.** A workspace admin enables developer mode. Apps and Connectors, Create, paste `https://YOUR-HOST/mcp`, choose OAuth. Plans with read and fetch access use the passport's `search` and `fetch` tools.
- **Muse.** See [connectors/muse](connectors/muse/README.md) for the directory listing pack and the custom connector route.

## 5. NotebookLM and other file readers

```bash
sozograph visa issue ~/.sozograph/passport.json --agent notebooklm --domains research --write
sozograph handoff ~/.sozograph/passport.json --token sgv_... --out research-passport.md
```

Upload `research-passport.md` as a source. When the research is done, ask NotebookLM to fill the SozoGraph return form, save its reply, and bring it home:

```bash
sozograph return ~/.sozograph/passport.json reply.md --token sgv_...
```

A reply in prose works too with `--provider anthropic` (or any provider): the findings are extracted and authored by NotebookLM.

## 6. Keep replicas in step

A laptop passport and the hosted passport merge by union of stamps:

```bash
sozograph visa issue PASSPORT --agent laptop --domains "*" --write --forget   # on the hosted passport
sozograph sync ~/.sozograph/passport.json https://YOUR-HOST --token sgv_...
```

Every order of syncing gives the same beliefs, and a forget made on either side stays forgotten on both.
