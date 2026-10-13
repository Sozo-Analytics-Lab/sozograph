"""
The admission gate: what a stamp must satisfy before it can shape beliefs.

Every stamp is scanned before it enters a log-backed passport:

1. Secrets never enter the file. Card numbers that pass Luhn, key and token
   formats, private keys and passwords are rejected outright.
2. A visa bounds its agent: write and forget permissions, and the domains the
   agent may write into.
3. Origins form a lattice. A lower-trust stamp that corrects, retracts,
   supersedes or forgets a higher-trust claim waits for the holder.
4. Protected attributes (identity, contact details, address, payment and
   banking, standing instructions to assistants) accept writes only from the
   holder's own words or confirmations. Everything else waits in quarantine.
5. Imperatives aimed at assistants inside third-party content are quarantined.

The verdict is a pure function of the stamp, the visa and the stamps it targets.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .schema import ORIGIN_RANK, Passport, Stamp, Visa

ACCEPTED, QUARANTINED, REJECTED = "accepted", "quarantined", "rejected"
TRUSTED = ORIGIN_RANK["user_confirmed"]

_SECRET_PATTERNS = [
    ("api key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}")),
    ("api key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}")),
    ("cloud key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("cloud key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("access token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("access token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("access token", re.compile(r"\bsgv_[A-Za-z0-9_-]{20,}")),
    ("access token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("password", re.compile(r"\b(?:password|passcode|passwd|pin|cvv|cvc)\b\s*(?:is|:|=)\s*\S+", re.I)),
]
_SECRET_KEY = re.compile(r"\b(?:password|passcode|passwd|pin|cvv|cvc|secret|api key|apikey|access token|"
                         r"auth token|private key|seed phrase|mnemonic)\b")
_DIGITS = re.compile(r"(?<![\d-])(?:\d[ -]?){13,19}(?![\d])")

_PROTECTED_KEY = re.compile(
    r"\b(?:e ?mail|phone|mobile|cell|whatsapp|telegram|contact|address|postcode|zip ?code|postal|"
    r"iban|swift|bic|routing|account (?:number|no)|bank|banking|card|payment|billing|invoice|wallet|"
    r"passport (?:number|no)|national id|id number|ssn|social security|tax id|date of birth|dob|"
    r"credential|login|username|instruction|instructions|standing order|assistant rule)\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

_INSTRUCTION = [
    re.compile(r"\b(?:ai|assistant|assistants|agent|agents|bot|bots|chatbot|llm|copilot)\b[^.\n]{0,80}?"
               r"\b(?:must|should|need to|are required to|always|record|save|store|remember|update|"
               r"forward|send|set|treat|ignore|disregard|add)\b", re.I),
    re.compile(r"\b(?:ignore|disregard|forget)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all)\b"
               r"[^.\n]{0,20}\binstructions?\b", re.I),
    re.compile(r"\bnote (?:to|for) (?:the )?(?:ai|assistants?|agents?|bots?)\b", re.I),
    re.compile(r"<!--.*?-->", re.S),
]

_CLAIM_TEXT_FIELDS = ("key", "value", "text", "item", "summary", "name", "attribute", "old", "new")


def _words(text: str) -> str:
    return re.sub(r"[_\-./]+", " ", str(text).lower())


def _luhn(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def claim_text(claim: dict[str, Any]) -> str:
    parts = []
    for field in _CLAIM_TEXT_FIELDS:
        value = claim.get(field)
        if value is None or value == "":
            continue
        parts.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str))
    return "\n".join(parts)


def scan_secret(claim: dict[str, Any], evidence: list[dict[str, Any]] | None = None) -> str | None:
    """The kind of secret a claim carries, or None."""
    texts = [claim_text(claim)] + [str(e.get("quote", "")) for e in evidence or []]
    for text in texts:
        for kind, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                return kind
        for match in _DIGITS.finditer(text):
            if _luhn(match.group(0)):
                return "card number"
    key = _words(claim.get("key") or claim.get("attribute") or "")
    if key and _SECRET_KEY.search(key):
        return "credential"
    return None


def protected(claim: dict[str, Any], origin: str) -> bool:
    key = _words(claim.get("key") or claim.get("attribute") or "")
    if key and _PROTECTED_KEY.search(key):
        return True
    if ORIGIN_RANK.get(origin, 0) <= ORIGIN_RANK["tool_result"]:
        text = claim_text({k: v for k, v in claim.items() if k != "key"})
        return bool(_EMAIL.search(text) or _IBAN.search(text) or _PHONE.search(text))
    return False


def instruction(claim: dict[str, Any]) -> bool:
    text = claim_text({k: v for k, v in claim.items() if k != "key"})
    return any(p.search(text) for p in _INSTRUCTION)


def admit(passport: Passport, stamp: Stamp, visa: Visa | None, *, now=None) -> tuple[str, str]:
    """Return (verdict, reason) for a stamp about to enter the log."""
    secret = scan_secret(stamp.claim, stamp.evidence)
    if secret:
        return REJECTED, f"secret ({secret}); credentials stay with the vendor's own vault"
    holder = stamp.author == "holder"
    if visa is not None:
        if not visa.active(now):
            return REJECTED, "visa is inactive"
        if not visa.write:
            return REJECTED, "visa is read-only"
        if stamp.op == "forget" and not visa.forget:
            return REJECTED, "visa lacks forget permission"
        if stamp.op == "confirm":
            return REJECTED, "confirmation is a holder action"
        domains = stamp.claim.get("domains") or ["general"]
        if stamp.claim and not visa.sees_domain(domains):
            return REJECTED, "claim domains lie outside the visa"
    by_id = {s.id: s for s in passport.stamps}
    rank = ORIGIN_RANK.get(stamp.origin, 0)
    for target_id in stamp.targets:
        target = by_id.get(target_id)
        if target is None:
            return REJECTED, f"unknown target {target_id}"
        if visa is not None and target.claim and not visa.sees_domain(target.claim.get("domains") or ["general"]):
            return REJECTED, "target lies outside the visa"
        if not holder and stamp.op != "confirm" and ORIGIN_RANK.get(target.origin, 0) > rank:
            return QUARANTINED, "a lower-trust write against a higher-trust claim waits for the holder"
    if holder:
        return ACCEPTED, ""
    if stamp.op == "forget" and stamp.origin != "user_direct":
        return QUARANTINED, "forgetting needs the holder's request"
    if stamp.claim:
        if rank < TRUSTED and stamp.origin != "user_direct" and protected(stamp.claim, stamp.origin):
            return QUARANTINED, "protected attribute needs the holder's words or confirmation"
        if rank < TRUSTED and instruction(stamp.claim):
            return QUARANTINED, "an instruction inside third-party content"
        if rank <= ORIGIN_RANK["imported"] and any(instruction({"text": str(e.get("quote", ""))})
                                                   for e in stamp.evidence):
            return QUARANTINED, "the evidence is an instruction inside third-party content"
    return ACCEPTED, ""
