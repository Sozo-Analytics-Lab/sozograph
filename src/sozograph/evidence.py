from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .prompts import (
    EVIDENCE_LINK_SCHEMA,
    EVIDENCE_LINK_SYSTEM_PROMPT,
    EVIDENCE_LINK_USER_PROMPT_TEMPLATE,
)
from .providers.base import LLMProvider

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SPAN_RE = re.compile(r"[^\n.!?;]+(?:[.!?;]+|$)")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "ours", "she", "that", "the", "their", "theirs", "them", "they", "this", "to",
    "us", "was", "we", "were", "with", "you", "your", "yours",
}


def _exact_substring(source: str, candidate: str | None) -> str | None:
    text = str(candidate or "").strip()
    if not text:
        return None
    match = re.search(re.escape(text), source, flags=re.IGNORECASE)
    return source[match.start() : match.end()] if match else None


def _claim(bucket: str, item: Any) -> str:
    if bucket in {"facts", "prefs"}:
        return f"{item.key}: {item.value}"
    if bucket == "entities":
        return str(item.name)
    if bucket == "open_loops":
        return str(item.item)
    return str(item.text)


def _fallbacks(bucket: str, item: Any) -> list[str]:
    supplied = getattr(item, "evidence_quote", None)
    values = [str(supplied or "")]
    if bucket in {"facts", "prefs"}:
        values.append(str(item.value))
    elif bucket == "entities":
        values.extend([str(item.name), *(str(alias) for alias in item.aliases)])
    elif bucket == "open_loops":
        values.append(str(item.item))
    elif bucket == "observations":
        values.append(str(item.text))
    return values


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(text)
        if len(token) > 1 and token.casefold() not in _STOP
    }


def deterministic_quote(bucket: str, item: Any, source_text: str) -> str | None:
    """Find a verified source span without another model call.

    Stable scalar values and names use exact matching. Paraphrased observations
    may use a source clause when most of the claim's content words occur in it.
    The returned text is always copied from the source.
    """
    for fallback in _fallbacks(bucket, item):
        quote = _exact_substring(source_text, fallback)
        if quote:
            return quote
    if bucket != "observations":
        return None

    claim_tokens = _tokens(_claim(bucket, item))
    if len(claim_tokens) < 2:
        return None
    numeric = {token for token in claim_tokens if any(char.isdigit() for char in token)}
    best: tuple[float, int, str] | None = None
    for match in _SPAN_RE.finditer(source_text):
        span = match.group(0).strip()
        span_tokens = _tokens(span)
        overlap = claim_tokens & span_tokens
        if len(overlap) < 2 or not numeric.issubset(span_tokens):
            continue
        coverage = len(overlap) / len(claim_tokens)
        precision = len(overlap) / max(1, len(span_tokens))
        score = 0.75 * coverage + 0.25 * precision
        candidate = (score, -len(span), span)
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] < 0.52:
        return None
    return best[2]


def unresolved_candidates(update: dict[str, Any], source_text: str) -> list[tuple[str, str, Any]]:
    unresolved: list[tuple[str, str, Any]] = []
    for bucket in ("facts", "prefs", "entities", "open_loops", "observations"):
        for index, item in enumerate(update.get(bucket) or []):
            quote = deterministic_quote(bucket, item, source_text)
            if quote:
                item.evidence_quote = quote
            else:
                unresolved.append((f"{bucket}:{index}", _claim(bucket, item), item))
    return unresolved


@dataclass(frozen=True)
class EvidenceLinkStats:
    deterministic: int = 0
    model_linked: int = 0
    unresolved: int = 0


def link_evidence(
    provider: LLMProvider,
    update: dict[str, Any],
    source_text: str,
) -> EvidenceLinkStats:
    """Link only unresolved candidates, then verify every returned quote."""
    total = sum(
        len(update.get(bucket) or [])
        for bucket in ("facts", "prefs", "entities", "open_loops", "observations")
    )
    unresolved = unresolved_candidates(update, source_text)
    deterministic = total - len(unresolved)
    if not unresolved:
        return EvidenceLinkStats(deterministic=deterministic)

    lookup = {candidate_id: item for candidate_id, _claim_text, item in unresolved}
    candidates = [
        {"candidate_id": candidate_id, "claim": claim_text}
        for candidate_id, claim_text, _item in unresolved
    ]
    payload = provider.complete_json(
        system=EVIDENCE_LINK_SYSTEM_PROMPT,
        user=EVIDENCE_LINK_USER_PROMPT_TEMPLATE.format(
            source_text=source_text,
            candidates_json=json.dumps(candidates, ensure_ascii=False),
        ),
        schema=EVIDENCE_LINK_SCHEMA,
        temperature=0.0,
    )
    linked = 0
    seen: set[str] = set()
    for row in payload.get("links") or []:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in seen or candidate_id not in lookup:
            continue
        seen.add(candidate_id)
        quote = str(row.get("quote") or "")
        exact = _exact_substring(source_text, quote)
        if exact:
            lookup[candidate_id].evidence_quote = exact
            linked += 1
    return EvidenceLinkStats(
        deterministic=deterministic,
        model_linked=linked,
        unresolved=len(unresolved) - linked,
    )
