"""
Visas: what each agent may see and write, and the minimum-necessary view it gets.

A visa names domains, write and forget permissions, a sensitivity ceiling and
an expiry. Its view of the passport is the eligible set

    E_a = { c : D(c) meets D_a, s(c) <= ceiling_a, the claim is still fresh }

packed to the budget with a selection that charges `price` per sensitivity
level, so a sensitive record needs more relevance to earn its place.
Out-of-scope claims are absent from the candidate set by construction.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime
from typing import Any

from .schema import AgentRecord, Passport, Visa, utcnow
from .shelf import freshness

DOMAINS = ("general", "identity", "shopping", "marketing", "documents", "software", "research",
           "health", "finance", "travel", "home", "work", "social")

_DOMAIN_WORDS = [
    ("software", r"\b(code|repo|github|gitlab|python|typescript|javascript|rust|golang|deploy|stack|framework|"
                 r"database|postgres|sql|docker|kubernetes|cloud run|firestore|commit|branch|pull request|"
                 r"ci|lint|pytest|api|sdk|backend|frontend|server)\b"),
    ("health", r"\b(health|allerg\w*|medic\w*|doctor|symptom|diagnos\w*|prescription|therapy|clinic|pregnan\w*|"
               r"headache|migraine|injur\w*|blood type|diet|vegetarian|vegan)\b"),
    ("finance", r"\b(bank|salary|income|budget|loan|tax|invoice|payment|credit|debt|invest\w*|pension|mortgage)\b"),
    ("shopping", r"\b(size|shoe\w*|brand|purchase\w*|bought|order|cart|shop\w*|store|discount|delivery|"
                 r"retailer|wishlist|price)\b"),
    ("marketing", r"\b(brand voice|campaign|audience|seo|newsletter|social media|followers|marketing|"
                  r"launch|ad spend|tagline|press)\b"),
    ("documents", r"\b(document|docs|report|template|style guide|house style|memo|letter|slides|deck|"
                  r"formatting|citation style)\b"),
    ("research", r"\b(research|paper|study|literature|dataset|hypothesis|experiment|notebook|arxiv)\b"),
    ("travel", r"\b(flight|hotel|trip|travel|airport|booking|check-in|check-out|itinerary|honeymoon)\b"),
]
_SENSITIVE = re.compile(r"\b(health|allerg\w*|medic\w*|doctor|symptom|diagnos\w*|prescription|therapy|"
                        r"pregnan\w*|bank|salary|income|loan|debt|tax|mortgage|divorc\w*|dating|sexual\w*|"
                        r"religio\w*|politic\w*|address|blood type)\b")


def _text(record: dict[str, Any]) -> str:
    parts = [record.get(k) for k in ("key", "value", "text", "item", "summary", "name", "old", "new")]
    return " ".join(str(p) for p in parts if p not in (None, "")).replace("_", " ").lower()


def effective_domains(record: dict[str, Any]) -> list[str]:
    if record.get("domains"):
        return list(record["domains"])
    text = _text(record)
    for domain, pattern in _DOMAIN_WORDS:
        if re.search(pattern, text):
            return [domain]
    return ["general"]


def effective_sensitivity(record: dict[str, Any]) -> int:
    level = record.get("sensitivity")
    if level is not None:
        return int(level)
    return 2 if _SENSITIVE.search(_text(record)) else 1


# -- issuing ---------------------------------------------------------------

def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_visa(passport: Passport, *, agent: str, domains=None, write=False, forget=False,
               sensitivity_ceiling=1, purpose="", expires_at=None, name="", vendor="", model="",
               include_general=True, now=None) -> Visa:
    """Register the agent, mint a visa and its one-time bearer token."""
    passport.assert_supported()
    moment = now or utcnow()
    agent = agent.strip()
    if not agent or agent == "holder" or agent.startswith(("import:", "sozograph:")):
        raise ValueError("agent names 'holder', 'import:*' and 'sozograph:*' are reserved")
    token = "sgv_" + secrets.token_urlsafe(32)
    visa = Visa(id="visa_" + secrets.token_hex(8), agent=agent,
                domains=sorted({d.strip().lower() for d in (domains or ["general"]) if d.strip()}),
                include_general=include_general, write=write, forget=forget,
                sensitivity_ceiling=sensitivity_ceiling, purpose=purpose, issued_at=moment,
                expires_at=expires_at, token_sha256=token_hash(token))
    visa._token = token
    record = passport.agents.get(agent) or AgentRecord(id=agent, first_seen=moment)
    record.name = name or record.name or agent
    record.vendor = vendor or record.vendor
    record.model = model or record.model
    if not record.home_domains:
        record.home_domains = [d for d in visa.domains if d != "*"]
    passport.agents[agent] = record
    passport.visas.append(visa)
    passport.touch(moment)
    return visa


def find_visa(passport: Passport, token: str, *, now: datetime | None = None) -> Visa | None:
    digest = token_hash(token)
    for visa in passport.visas:
        if secrets.compare_digest(visa.token_sha256, digest) and visa.active(now):
            return visa
    return None


def revoke(passport: Passport, visa_id: str | None = None, *, everything: bool = False, now=None) -> int:
    """Revoke one visa, or every visa at once (the holder's kill switch)."""
    moment = now or utcnow()
    count = 0
    for visa in passport.visas:
        if (everything or visa.id == visa_id) and visa.revoked_at is None:
            visa.revoked_at = moment
            count += 1
    if count:
        passport.touch(moment)
    return count


def holder_visa() -> Visa:
    """The holder's own view: every domain, every sensitivity level the file holds."""
    return Visa(id="holder", agent="holder", domains=["*"], write=True, forget=True, sensitivity_ceiling=2)


# -- viewing -----------------------------------------------------------------

_VIEW_SECTIONS = ("facts", "prefs", "entities", "open_loops", "contradictions", "episodes", "observations")
_FRESH_SECTIONS = {"facts", "prefs", "observations", "open_loops"}


def eligible(record: dict[str, Any], visa: Visa, section: str, now: datetime, *, include_expired=False,
             overrides=None) -> tuple[bool, str]:
    if not visa.sees_domain(effective_domains(record)):
        return False, "outside visa domains"
    if effective_sensitivity(record) > visa.sensitivity_ceiling:
        return False, "above sensitivity ceiling"
    if section in _FRESH_SECTIONS:
        state, _ = freshness(record, now, overrides=overrides)
        if state == "expired" and not include_expired:
            return False, "expired"
        return True, state
    return True, "current"


def scoped_copy(passport: Passport, visa: Visa, *, now=None, include_expired=False) -> Passport:
    """A passport holding only what this visa may see, freshness marked on each record."""
    moment = now or utcnow()
    overrides = passport.meta.get("shelf")
    copy = passport.model_copy(deep=True)
    copy.sources = []
    copy.stamps = []
    copy.visas = []
    copy.agents = {}
    copy.meta = {}
    for section in _VIEW_SECTIONS:
        kept = []
        for record in getattr(copy, section):
            data = record.to_compact()
            if hasattr(record, "ts"):
                data["ts"] = record.ts.isoformat()
            ok, state = eligible(data, visa, section, moment, include_expired=include_expired,
                                 overrides=overrides)
            if not ok:
                continue
            if state in {"stale", "upcoming", "expired"}:
                record.freshness = state
            kept.append(record)
        setattr(copy, section, kept)
    return copy


def view(passport: Passport, visa: Visa, *, query=None, budget_chars=6000, budget_tokens=None,
         token_counter=None, now=None, price=0.05, include_expired=False, caps=None,
         header="SOZOGRAPH PASSPORT"):
    """The visa's minimum-necessary context for a query, as a RecallResult."""
    from .recall import recall
    moment = now or utcnow()
    if not visa.active(moment):
        raise PermissionError("visa is inactive")
    scoped = scoped_copy(passport, visa, now=moment, include_expired=include_expired)
    return recall(scoped, query=query, budget_chars=budget_chars, budget_tokens=budget_tokens,
                  token_counter=token_counter, caps=caps, header=f"{header} (visa: {visa.agent})",
                  penalty=lambda section, data: price * effective_sensitivity(data))
