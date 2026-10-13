# Security model

A passport concentrates what a person's agents know about them. This page states what SozoGraph defends, how, and where the boundary sits.

## Assets

The holder's claims; the stamp log and its history; source evidence; visa tokens and refresh tokens; the holder secret; the server signing key.

## Threats and defences

| Threat | Defence |
|---|---|
| Injected instructions in emails, web pages and tool results become memory (memory poisoning) | Every stamp carries an origin. Instructions aimed at assistants inside third-party content, and claims resting on such evidence, wait in quarantine. Protected attributes accept writes only from the holder's words or confirmation. Passport Relay measures it: a policy that believes every injection passes 26% of the injection checks with the gate off and 100% with it on |
| A third-party claim overrides the holder's own | The origin guard keeps newer low-trust claims behind higher-trust ones, and lower-trust stamps that supersede, correct, retract or forget a higher-trust claim wait for the holder |
| Secrets stored in memory | Card numbers (Luhn), API keys, access tokens, private keys, passwords and credential-named attributes are rejected before storage, for every writer including the holder |
| An agent reads beyond its scope | Each agent reads through its own visa: domains, sensitivity ceiling and expiry. Out-of-scope records leave the candidate set before ranking. Agent-initiated forgets match and cascade only inside the visa |
| A stolen token | Tokens are stored as SHA-256 hashes. Visas expire. Refresh rotates the token. The holder revokes one visa, or every visa at once, in the visa office |
| Token reuse across servers | OAuth tokens are bound to the passport's `/mcp` resource (RFC 8707), and the server refuses tokens issued for another resource |
| Stamps entering a hosted log another way | The server signs admitted stamps with HMAC; `sozograph verify --key` flags unsigned or altered stamps. Ed25519 signatures identify individual writers offline |
| Tampering with a saved file | Content-addressed stamp ids, the Merkle head and a projection check in `verify` |
| Erasure undone by a stale replica | Redaction is the top of the merge lattice: a forget stays applied after every merge |
| Consent forgery | The consent page requires the holder secret; visa office actions require a signed session cookie and a CSRF token |
| Memory text forging prompt structure | Rendered rows escape control characters and newlines, and the context header labels memory as data |

## The boundary

- **Vendor copies.** An agent may keep what it read in its own memory. A forget erases the passport and every replica it syncs with; copies inside a vendor's product follow that vendor's deletion controls.
- **A compromised agent with a write visa** can write plausible false claims with origin `user_direct`. Competence, history and the holder's confirmations limit the damage; the holder revokes the visa.
- **The hosted server** runs with the holder's trust. Run it on infrastructure the holder controls, keep `SOZOGRAPH_HOLDER_SECRET` and `SOZOGRAPH_SERVER_KEY` in a secret manager, and serve it over HTTPS only.
- **Formatting is a mitigation.** Labelling memory as data reduces prompt injection through recalled text; the receiving model still decides what it obeys.

## Reporting

Report vulnerabilities privately to the maintainers through GitHub security advisories on github.com/Sozo-Analytics-Lab/sozograph.
