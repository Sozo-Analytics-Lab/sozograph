# SozoGraph passport: privacy policy (template)

Replace the bracketed fields, then publish this page at a public URL before submitting a connector listing.

**Who holds the data.** The passport belongs to its holder, [HOLDER OR OPERATOR NAME]. It lives in storage the holder controls: [a file on the holder's server | a Firestore project the holder owns].

**What the connector reads.** An agent reads only the records inside its visa: the domains, sensitivity ceiling and expiry the holder approved on the consent page.

**What the connector writes.** An agent writes stamps: claims with an author, an origin, a time and evidence. The holder sees every write in the visa office. Writes to contact details, addresses, payment and banking fields, identity numbers and standing instructions wait for the holder's confirmation.

**What the passport refuses.** Card numbers, passwords, keys and tokens are rejected before storage.

**Tokens.** Access tokens are visa tokens. The server stores their SHA-256 hashes only. Refresh tokens rotate on every use.

**Logs.** The server keeps an audit log of each request: time, visa, agent, action and counts. Log rows hold record counts and query text up to 200 characters.

**Deletion.** The holder revokes any visa, or every visa at once, in the visa office. A forget request erases matching records from every replica the passport syncs with. Copies an agent already stored in its own memory follow that vendor's deletion controls.

**Contact.** [EMAIL]
