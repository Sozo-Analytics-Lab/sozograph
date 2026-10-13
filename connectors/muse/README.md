# Muse connector pack

Everything the Muse connector platform asks for, ready to paste. Meta reviews directory connectors for functional, security and legal requirements before listing them. Muse also builds a custom connector for one person, straight from an OpenAPI spec.

## Two routes

**Directory listing** (every Muse user can add it): submit at muse.ai/platform with the fields below. Choose connection type **Existing MCP** and give the hosted endpoint, or **Raw API** with the OpenAPI URL.

**Custom connector** (the holder's own Muse): ask Muse:

> Build a connector for my SozoGraph passport. The OpenAPI spec is at https://YOUR-HOST/openapi.json. It uses a bearer token.

Paste the visa token into Muse's secure credential prompt, and keep it out of the chat. Set the approval policy to "ask per action" for writes.

## Listing fields

| Field | Value |
|---|---|
| Connector name | SozoGraph passport |
| Company | Sozo Analytics Lab |
| Website | https://github.com/Sozo-Analytics-Lab/sozograph |
| Icon | `icon-512.png` (512 x 512) |
| Connection type | Existing MCP: `https://YOUR-HOST/mcp` (or Raw API: `https://YOUR-HOST/openapi.json`) |
| Documentation | https://github.com/Sozo-Analytics-Lab/sozograph/blob/main/CONNECTORS.md |
| Authentication | OAuth 2.1 with PKCE. Discovery at `/.well-known/oauth-protected-resource/mcp`; dynamic client registration at `/register` |
| Access requirements | The holder runs or hosts their own passport. Free to use in every region |
| Rate limits | 50 stamps per write call |
| Support contact | Set before submission |
| Privacy policy | `privacy-policy.md`, published at a public URL |
| Terms | `terms.md`, published at a public URL |

## Description

> Your memory, everywhere you work. SozoGraph keeps one passport of what your agents know about you: sizes, brands, budgets, projects, plans. Muse reads the shopping slice of it under a visa you approve, and writes back what it learns. Claude Code, Grok, Gemini and your other agents read their own slices. You see every write, approve anything sensitive, and revoke any visa in one click.

## Example prompts

- "Check my passport for my shoe size and favourite running brands, then find trail shoes under 120 dollars."
- "Remember that I returned the blue jacket because the sleeves ran short."
- "What did I buy for the house last month?"
- "Save my preference for refurbished electronics with a warranty."

## Scopes Muse should request

`passport:read passport:write domain:shopping domain:home`

The holder chooses the final domains, write permission, sensitivity ceiling and expiry on the consent page. Purchases and payment details stay inside Muse; the passport's gate rejects card numbers and quarantines payment and address writes until the holder confirms them.
