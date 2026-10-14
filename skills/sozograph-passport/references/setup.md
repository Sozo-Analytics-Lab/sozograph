# Connecting each agent

The holder runs one passport. Each agent gets its own visa.

| Agent | Route | Setup |
|---|---|---|
| Claude Code | Local MCP over stdio | `claude mcp add sozograph -- sozograph serve --stdio --passport ~/.sozograph/passport.json --agent claude-code --domains software,documents`, or install the SozoGraph plugin |
| Claude.ai, Claude Desktop | Custom connector | Settings, Connectors, Add custom connector, paste `https://<host>/mcp`. Upload this Skill as a ZIP in Settings, Capabilities |
| Grok | Custom connector | grok.com/connectors, New Connector, Custom, paste `https://<host>/mcp` |
| Gemini app | Custom app | Settings, Connected Apps, Add a custom app, paste `https://<host>/mcp` |
| ChatGPT and Dots | Developer mode app | Workspace admin enables developer mode; Apps and Connectors, Create, paste `https://<host>/mcp`, OAuth |
| Muse | Muse connector | Directory listing (MCP URL or OpenAPI at `https://<host>/openapi.json`), or ask Muse to build a custom connector from the OpenAPI spec |
| NotebookLM | Handoff packet | `sozograph handoff passport.json --token $NOTEBOOKLM_VISA --out research.md`, upload as a source, bring the return form back with `sozograph return` |

Every remote connector sends the holder to the visa office to approve the visa's domains and permissions. Revoke any visa there, or all of them at once.
