"""Context rendering facade and deterministic section priors."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .schema import Episode, Passport

_DAY = 86_400.0

@dataclass(frozen=True)
class Caps:
    """How many of each section to include."""

    facts: int = 60
    prefs: int = 30
    entities: int = 15
    open_loops: int = 12
    contradictions: int = 8
    episodes: int = 12
    observations: int = 40



def _kv_prior(now: float, oldest: float):
    """
    Score a fact or preference by recency and confidence.

    The previous formula was `ts.timestamp() / 1e9 + confidence * 0.5`, which
    put recency on a ~1.77 scale and confidence on a 0.5 scale. A 0.1 gap in
    confidence outranked roughly three years of recency, so the weighting was
    effectively confidence-only and by accident. Both terms are normalized to
    0..1 here and the weights are stated.
    """
    span = max(now - oldest, _DAY)

    def score(item: Any) -> float:
        recency = 1.0 - min(1.0, (now - item.ts.timestamp()) / span)
        return 0.6 * recency + 0.4 * float(item.confidence)

    return score



def _time_prior(now: float, oldest: float):
    span = max(now - oldest, _DAY)

    def score(item: Any) -> float:
        stamp = getattr(item, "ts", None) or getattr(item, "ts_new", None)
        return 1.0 - min(1.0, (now - stamp.timestamp()) / span)

    return score



def _episode_prior(now: float, oldest: float):
    span = max(now - oldest, _DAY)

    def score(ep: Episode) -> float:
        recency = 1.0 - min(1.0, (now - ep.ts.timestamp()) / span)
        return 0.5 * recency + 0.5 * float(ep.salience)

    return score



def _bounds(passport: Passport) -> tuple:
    stamps = [f.ts.timestamp() for f in passport.facts]
    stamps += [p.ts.timestamp() for p in passport.prefs]
    stamps += [o.ts.timestamp() for o in passport.open_loops]
    stamps += [e.ts.timestamp() for e in passport.episodes]
    stamps += [o.ts.timestamp() for o in passport.observations]
    stamps += [c.ts_new.timestamp() for c in passport.contradictions]
    stamps.append(passport.updated_at.timestamp())
    return max(stamps), min(stamps)



_TIME_QUERY_RE = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|\bwhen\b|\bhow long\b|\bhow old\b|what day|which day|what time|what year",
    re.IGNORECASE,
)



def _wants_chronology(query: str | None) -> bool:
    return bool(query) and bool(_TIME_QUERY_RE.search(query))



def export_context(passport: Passport, *, query=None, budget_chars=6000,
                   header="SOZOGRAPH PASSPORT", caps=None, **kwargs) -> str:
    from .recall import recall
    return recall(passport, query=query, budget_chars=budget_chars, header=header,
                  caps=caps, **kwargs).context
