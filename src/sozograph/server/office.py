"""
The visa office: OAuth 2.1 for the hosted passport, where consent issues a visa.

Every chat agent that adds the passport as a custom connector (Claude, Grok,
Gemini, ChatGPT, Muse) registers dynamically, sends the holder to the consent
page, and receives an access token. The access token is the visa's bearer
token; the refresh token rotates it. The consent page is where the holder
picks the visa's domains, write and forget permissions, sensitivity ceiling
and expiry. Tokens are stored as SHA-256 hashes only.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass
from datetime import timedelta
from html import escape
from pathlib import Path
from typing import Any

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from ..schema import Visa, utcnow
from ..visa import DOMAINS, find_visa, token_hash
from .service import PassportService

CODE_TTL = 300
REQUEST_TTL = 900
SESSION_TTL = 12 * 3600


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scopes_for(visa: Visa) -> list[str]:
    scopes = ["passport:read"]
    if visa.write:
        scopes.append("passport:write")
    if visa.forget:
        scopes.append("passport:forget")
    scopes += [f"domain:{d}" for d in visa.domains]
    return scopes


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:40] or "agent"


@dataclass
class _Pending:
    client: OAuthClientInformationFull
    params: AuthorizationParams
    created: float


@dataclass
class _Code:
    code: AuthorizationCode
    token: str
    visa_id: str


class OfficeState:
    """Clients, refresh tokens, the holder secret hash and keys, beside the passport."""

    def __init__(self, document):
        if isinstance(document, (str, Path)):
            from ..store import JsonDocument
            document = JsonDocument(path=Path(document))
        self.document = document
        self.data: dict[str, Any] = {"clients": {}, "refresh": {}, "holder_secret_sha256": "",
                                     "session_key": "", "server_key": ""}
        self.data.update(document.load())

    def save(self) -> None:
        self.document.save(self.data)

    def ensure_keys(self, holder_secret: str | None = None, server_key: str | None = None) -> str | None:
        """Set keys once. Returns a generated holder secret the first time, for one display."""
        generated = None
        if holder_secret:
            self.data["holder_secret_sha256"] = _sha(holder_secret)
        elif not self.data.get("holder_secret_sha256"):
            generated = secrets.token_urlsafe(18)
            self.data["holder_secret_sha256"] = _sha(generated)
        if not self.data.get("session_key"):
            self.data["session_key"] = secrets.token_hex(32)
        if server_key:
            self.data["server_key"] = server_key
        elif not self.data.get("server_key"):
            self.data["server_key"] = secrets.token_hex(32)
        self.save()
        return generated


class VisaOffice:
    """OAuthAuthorizationServerProvider whose access tokens are visa tokens."""

    def __init__(self, service: PassportService, state: OfficeState, *, issuer: str, resource: str):
        self.service = service
        self.state = state
        self.issuer = issuer.rstrip("/")
        self.resource = resource
        self._pending: dict[str, _Pending] = {}
        self._codes: dict[str, _Code] = {}

    # -- holder secret and session --------------------------------------------

    def check_secret(self, secret: str) -> bool:
        expected = self.state.data.get("holder_secret_sha256", "")
        return bool(expected) and hmac.compare_digest(expected, _sha(secret or ""))

    def session_cookie(self) -> str:
        expiry = str(int(time.time()) + SESSION_TTL)
        mac = hmac.new(bytes.fromhex(self.state.data["session_key"]), expiry.encode(), hashlib.sha256).hexdigest()
        return f"{expiry}.{mac}"

    def valid_session(self, cookie: str | None) -> bool:
        if not cookie or "." not in cookie:
            return False
        expiry, mac = cookie.split(".", 1)
        good = hmac.new(bytes.fromhex(self.state.data["session_key"]), expiry.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(mac, good) and expiry.isdigit() and int(expiry) > time.time()

    def csrf(self, cookie: str) -> str:
        return hmac.new(bytes.fromhex(self.state.data["session_key"]), b"csrf:" + cookie.encode(),
                        hashlib.sha256).hexdigest()[:32]

    # -- clients ----------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        raw = self.state.data["clients"].get(client_id)
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.state.data["clients"][client_info.client_id] = client_info.model_dump(mode="json", exclude_none=True)
        self.state.save()

    # -- authorization ------------------------------------------------------------

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if params.resource and params.resource.rstrip("/") != self.resource.rstrip("/"):
            raise AuthorizeError("invalid_target", "this server issues tokens for its own MCP endpoint only")
        now = time.time()
        self._pending = {k: v for k, v in self._pending.items() if now - v.created < REQUEST_TTL}
        request_id = secrets.token_urlsafe(24)
        self._pending[request_id] = _Pending(client, params, now)
        return f"{self.issuer}/visa-office/consent?request={request_id}"

    def pending_request(self, request_id: str) -> _Pending | None:
        entry = self._pending.get(request_id)
        if entry and time.time() - entry.created < REQUEST_TTL:
            return entry
        return None

    def approve(self, request_id: str, *, domains: list[str], write: bool, forget: bool, ceiling: int,
                days: int) -> str:
        entry = self._pending.pop(request_id, None)
        if entry is None:
            raise AuthorizeError("invalid_request", "consent request expired")
        name = entry.client.client_name or entry.client.client_id
        expires = utcnow() + timedelta(days=days) if days else None
        with self.service.store.transaction(signer=self.service.signer) as p:
            visa = p.issue_visa(_slug(name), domains=domains or ["general"], write=write, forget=forget,
                                sensitivity_ceiling=ceiling, purpose=f"oauth:{entry.client.client_id}",
                                expires_at=expires, name=name)
        self.service._cache = None
        code = secrets.token_urlsafe(32)
        auth_code = AuthorizationCode(
            code=code, scopes=scopes_for(visa), expires_at=time.time() + CODE_TTL,
            client_id=entry.client.client_id, code_challenge=entry.params.code_challenge,
            redirect_uri=entry.params.redirect_uri,
            redirect_uri_provided_explicitly=entry.params.redirect_uri_provided_explicitly,
            resource=entry.params.resource or self.resource, subject="holder")
        self._codes[code] = _Code(auth_code, visa.token, visa.id)
        return construct_redirect_uri(str(entry.params.redirect_uri), code=code, state=entry.params.state,
                                      iss=self.issuer)

    def deny(self, request_id: str) -> str:
        entry = self._pending.pop(request_id, None)
        if entry is None:
            raise AuthorizeError("invalid_request", "consent request expired")
        return construct_redirect_uri(str(entry.params.redirect_uri), error="access_denied",
                                      state=entry.params.state, iss=self.issuer)

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str):
        entry = self._codes.get(authorization_code)
        if entry is None or entry.code.client_id != client.client_id or entry.code.expires_at < time.time():
            return None
        return entry.code

    async def exchange_authorization_code(self, client, authorization_code) -> OAuthToken:
        entry = self._codes.pop(authorization_code.code, None)
        if entry is None:
            raise TokenError("invalid_grant", "authorization code already used or unknown")
        refresh = "sgr_" + secrets.token_urlsafe(32)
        self.state.data["refresh"][_sha(refresh)] = {"visa": entry.visa_id, "client_id": client.client_id,
                                                     "scopes": authorization_code.scopes,
                                                     "resource": authorization_code.resource}
        self.state.save()
        return OAuthToken(access_token=entry.token, expires_in=self._expires_in(entry.visa_id),
                          scope=" ".join(authorization_code.scopes), refresh_token=refresh)

    def _expires_in(self, visa_id: str) -> int | None:
        visa = next((v for v in self.service.read().visas if v.id == visa_id), None)
        if visa is None or visa.expires_at is None:
            return None
        return max(0, int((visa.expires_at - utcnow()).total_seconds()))

    async def load_refresh_token(self, client, refresh_token: str):
        row = self.state.data["refresh"].get(_sha(refresh_token))
        if row is None or row["client_id"] != client.client_id:
            return None
        return RefreshToken(token=refresh_token, client_id=client.client_id, scopes=row["scopes"],
                            resource=row.get("resource"), subject="holder")

    async def exchange_refresh_token(self, client, refresh_token, scopes) -> OAuthToken:
        row = self.state.data["refresh"].pop(_sha(refresh_token.token), None)
        if row is None:
            raise TokenError("invalid_grant", "refresh token unknown")
        token = "sgv_" + secrets.token_urlsafe(32)
        with self.service.store.transaction(signer=self.service.signer) as p:
            visa = next((v for v in p.visas if v.id == row["visa"]), None)
            if visa is None or not visa.active():
                self.state.save()
                raise TokenError("invalid_grant", "visa revoked")
            visa.token_sha256 = token_hash(token)
            granted = scopes_for(visa)
        self.service._cache = None
        refresh = "sgr_" + secrets.token_urlsafe(32)
        self.state.data["refresh"][_sha(refresh)] = {**row, "scopes": granted}
        self.state.save()
        return OAuthToken(access_token=token, expires_in=self._expires_in(row["visa"]),
                          scope=" ".join(granted), refresh_token=refresh)

    async def load_access_token(self, token: str) -> AccessToken | None:
        visa = find_visa(self.service.read(), token)
        if visa is None:
            return None
        expires = int(visa.expires_at.timestamp()) if visa.expires_at else None
        return AccessToken(token=token, client_id=visa.purpose.removeprefix("oauth:") or visa.agent,
                           scopes=scopes_for(visa), expires_at=expires, resource=self.resource,
                           subject="holder", claims={"iss": self.issuer, "visa": visa.id})

    async def revoke_token(self, token) -> None:
        if isinstance(token, AccessToken):
            visa = find_visa(self.service.read(), token.token)
            if visa is not None:
                self.service.decide("revoke", [visa.id])
        elif isinstance(token, RefreshToken):
            row = self.state.data["refresh"].pop(_sha(token.token), None)
            self.state.save()
            if row:
                self.service.decide("revoke", [row["visa"]])

    async def exchange_identity_assertion(self, client, params) -> OAuthToken:
        raise TokenError("unsupported_grant_type", "identity assertions are unsupported")


# -- pages -------------------------------------------------------------------

_STYLE = """
:root{--ink:#1B2A4A;--teal:#127D82;--orange:#BE7C1F;--paper:#FFFFFF;--soft:#F3F6F5;--line:#D5DCDB;--muted:#6B6B6B}
@media (prefers-color-scheme:dark){:root{--ink:#E8EEF3;--paper:#0F1726;--soft:#17223A;--line:#2A3A5A;--muted:#9FB3C8}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Calibri,sans-serif}
main{max-width:720px;margin:0 auto;padding:32px 16px}h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}
.sub{color:var(--muted);margin:0 0 24px}.card{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
label{display:block;margin:6px 0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:4px 12px}
input[type=password],select{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--paper);color:var(--ink);font:inherit}
button{background:var(--teal);color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;cursor:pointer;margin:4px 8px 4px 0}
button.alt{background:transparent;color:var(--ink);border:1px solid var(--line)}button.warn{background:var(--orange)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}
code{font-size:13px}.mark{color:var(--teal);font-weight:600}
"""


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{escape(title)}</title>"
            f"<style>{_STYLE}</style></head><body><main>{body}</main></body></html>")


def consent_page(office: VisaOffice, request_id: str, entry: _Pending, error: str = "") -> str:
    name = escape(entry.client.client_name or entry.client.client_id)
    requested = {s.split(":", 1)[1] for s in (entry.params.scopes or []) if s.startswith("domain:")}
    boxes = "".join(
        f"<label><input type=checkbox name=domains value={d} {'checked' if d in requested or d == 'general' else ''}>"
        f" {d}</label>" for d in DOMAINS)
    wants_write = not entry.params.scopes or "passport:write" in (entry.params.scopes or [])
    note = f"<p class=mark>{escape(error)}</p>" if error else ""
    return page("Visa office", f"""
<h1>Visa office</h1><p class=sub><b>{name}</b> asks to open your passport.</p>{note}
<form method=post action="{office.issuer}/visa-office/consent">
<input type=hidden name=request value="{escape(request_id)}">
<div class=card><h2>Domains this agent may see</h2><div class=grid>{boxes}</div></div>
<div class=card><h2>Permissions</h2>
<label><input type=checkbox name=write value=1 {'checked' if wants_write else ''}> Write stamps</label>
<label><input type=checkbox name=forget value=1> Erase on your request</label>
<label>Sensitivity ceiling <select name=ceiling><option value=0>Public only</option>
<option value=1 selected>Personal</option><option value=2>Sensitive (health, money)</option></select></label>
<label>Expires after <select name=days><option value=0>No expiry</option><option value=7>7 days</option>
<option value=30 selected>30 days</option><option value=90>90 days</option></select></label></div>
<div class=card><label>Holder secret <input type=password name=secret autocomplete=current-password required></label></div>
<button name=decision value=approve>Issue visa</button><button class=alt name=decision value=deny>Decline</button>
</form>""")


def login_page(office: VisaOffice, error: str = "") -> str:
    note = f"<p class=mark>{escape(error)}</p>" if error else ""
    return page("Visa office", f"""<h1>Visa office</h1><p class=sub>Your passport, your visas.</p>{note}
<form method=post action="{office.issuer}/visa-office/login"><div class=card>
<label>Holder secret <input type=password name=secret autocomplete=current-password required></label></div>
<button>Open</button></form>""")


def office_page(office: VisaOffice, cookie: str) -> str:
    p = office.service.read()
    token = office.csrf(cookie)
    rows = "".join(
        f"<tr><td><b>{escape(v.agent)}</b><br><code>{escape(v.id)}</code></td><td>{escape(', '.join(v.domains))}</td>"
        f"<td>{'write' if v.write else 'read'}{', forget' if v.forget else ''}</td>"
        f"<td>{'active' if v.active() else 'ended'}</td><td>"
        + (f"<form method=post action='{office.issuer}/visa-office/action'><input type=hidden name=csrf value={token}>"
           f"<input type=hidden name=ids value='{escape(v.id)}'><button class=alt name=action value=revoke>Revoke</button>"
           f"</form>" if v.active() else "") + "</td></tr>" for v in p.visas)
    pending = "".join(
        f"<tr><td>{escape(x['author'])}</td><td>{escape(x['summary'])}</td><td>{escape(x['reason'])}</td><td>"
        f"<form method=post action='{office.issuer}/visa-office/action'><input type=hidden name=csrf value={token}>"
        f"<input type=hidden name=ids value='{escape(x['id'])}'><button name=action value=confirm>Confirm</button>"
        f"<button class=alt name=action value=dismiss>Dismiss</button></form></td></tr>" for x in p.pending())
    holder = escape(p.holder.get("name") or p.holder.get("id") or "holder")
    return page("Visa office", f"""<h1>Visa office</h1><p class=sub>Passport of {holder}. {len(p.stamps)} stamps.</p>
<h2>Waiting for you</h2><div class=card><table><tr><th>Agent</th><th>Stamp</th><th>Why it waits</th><th></th></tr>
{pending or '<tr><td colspan=4>Nothing waits.</td></tr>'}</table></div>
<h2>Visas</h2><div class=card><table><tr><th>Agent</th><th>Domains</th><th>Rights</th><th>State</th><th></th></tr>
{rows or '<tr><td colspan=5>No visas yet.</td></tr>'}</table></div>
<form method=post action='{office.issuer}/visa-office/action'><input type=hidden name=csrf value={token}>
<input type=hidden name=ids value=all><button class=warn name=action value=revoke_all>Revoke every visa</button></form>""")
