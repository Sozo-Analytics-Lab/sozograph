"""
Shelf life: how long a claim stays true without anyone saying so.

A claim of kind k survives d days after it became true with a Weibull survival
function S(d) = exp(-(d / eta)^gamma). An explicit end date overrides it. The
scales below are engineering defaults; `fit_scale` refits one from the
holder's own retractions with the closed-form censored maximum likelihood
estimate for a known shape.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from typing import Any

INF = math.inf

#: kind -> (scale in days, shape). None means the claim needs an explicit end.
KINDS: dict[str, tuple[float, float] | None] = {
    "mood": (1, 2),
    "situation": (2, 2),
    "symptom": (7, 2),
    "visit": (10, 2),
    "booking": None,
    "trip": None,
    "task": (30, 1),
    "project_state": (365, 1),
    "preference": (1095, 1),
    "employment": (1461, 1),
    "residence": (1826, 1),
    "relationship": (3652, 1),
    "identity": (INF, 1),
    "fact": (INF, 1),
}
#: Bookings and trips without an end fall back to a month.
_UNBOUNDED = (30, 2)

CURRENT, STALE = 0.5, 0.2

_KIND_WORDS = [
    ("symptom", r"headache|migraine|fever|flu|cold|cough|sprain|twisted|injur|sore|pain|sick|ill\b|nausea|ankle"),
    ("situation", r"delay|outage|power cut|load.?shedding|stuck|traffic|queue|offline|down\b"),
    ("mood", r"\bmood\b|feeling|tired|stressed|anxious|excited|bored"),
    ("visit", r"visit|staying with|in town|guest"),
    ("booking", r"booking|reservation|check.?in|check.?out|hotel|flight|ticket"),
    ("trip", r"\btrip\b|travel|vacation|holiday|honeymoon"),
    ("employment", r"employer|job|role|works at|work_at|company|position|title"),
    ("residence", r"city|home|lives|live_in|residence|address|country|moved"),
    ("relationship", r"spouse|married|partner|wife|husband|girlfriend|boyfriend|engaged|divorc"),
    ("identity", r"birthday|birth|blood|name|nationality|allerg|height"),
    ("project_state", r"project|repo|stack|framework|deploy|sprint|branch|release|version"),
    ("preference", r"prefer|favourite|favorite|likes?|dislikes?|style|tone|brand|size"),
]


def infer_kind(key: str, value: Any = "") -> str:
    """A conservative guess from words in the attribute and value."""
    text = f"{key} {value}".lower()
    for kind, pattern in _KIND_WORDS:
        if re.search(pattern, text):
            return kind
    return "fact"


def parse_moment(value: str | datetime | date | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        try:
            moment = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def scale(kind: str, overrides: dict[str, Any] | None = None) -> tuple[float, float]:
    if overrides and kind in overrides:
        eta, gamma = overrides[kind]
        return float(eta), float(gamma)
    entry = KINDS.get(kind or "fact", KINDS["fact"])
    return entry if entry is not None else _UNBOUNDED


def survival(kind: str, age_days: float, *, overrides=None) -> float:
    if age_days <= 0:
        return 1.0
    eta, gamma = scale(kind, overrides)
    if eta == INF:
        return 1.0
    return math.exp(-((age_days / eta) ** gamma))


def median_life(kind: str, overrides=None) -> float:
    eta, gamma = scale(kind, overrides)
    return INF if eta == INF else eta * math.log(2) ** (1 / gamma)


def freshness(record: dict[str, Any], now: datetime, *, overrides=None) -> tuple[str, float]:
    """
    ("upcoming" | "current" | "stale" | "expired", survival) at time `now`.

    The clock starts at valid_from, the event date, or the discussion time, in
    that order. An explicit valid_until ends the claim on that day.
    """
    begins = parse_moment(record.get("valid_from"))
    if begins is not None and begins > now:
        return "upcoming", 1.0
    until = parse_moment(record.get("valid_until"))
    if until is not None:
        if len(str(record.get("valid_until"))) <= 10:
            until = until.replace(hour=23, minute=59, second=59)
        return ("expired", 0.0) if now > until else ("current", 1.0)
    kind = record.get("kind") or "fact"
    start = (parse_moment(record.get("valid_from")) or parse_moment(record.get("when"))
             or parse_moment(record.get("ts")))
    if start is None:
        return "current", 1.0
    s = survival(kind, (now - start).total_seconds() / 86_400, overrides=overrides)
    if s >= CURRENT:
        return "current", s
    return ("stale", s) if s >= STALE else ("expired", s)


def fit_scale(observed_days: list[float], censored_days: list[float], shape: float) -> float | None:
    """
    Censored Weibull scale for a known shape:

        eta = ( sum over all claims of d_i^gamma / r ) ^ (1 / gamma)

    where r counts the observed ends and the sum also runs over claims still open.
    """
    r = len(observed_days)
    if r == 0:
        return None
    total = sum(max(d, 0.0) ** shape for d in observed_days) + sum(max(d, 0.0) ** shape for d in censored_days)
    return (total / r) ** (1 / shape)


def fit_from_passport(passport, now: datetime | None = None) -> dict[str, tuple[float, float]]:
    """Refit scales per kind from holder retractions in a log-backed passport."""
    from .schema import utcnow
    moment = now or utcnow()
    by_id = {s.id: s for s in passport.stamps}
    ended: dict[str, float] = {}
    for s in passport.stamps:
        if s.op == "retract" and s.admission == "accepted":
            for t in s.targets:
                target = by_id.get(t)
                if target and target.claim:
                    start = parse_moment(target.claim.get("valid_from") or target.claim.get("ts"))
                    end = parse_moment(s.claim.get("valid_until")) or _hlc_moment(s.hlc)
                    if start and end:
                        ended[t] = (end - start).total_seconds() / 86_400
    observed: dict[str, list[float]] = {}
    open_: dict[str, list[float]] = {}
    for s in passport.stamps:
        if s.op not in {"assert", "supersede", "correct"} or not s.claim or s.section not in {"facts", "prefs"}:
            continue
        kind = s.claim.get("kind") or "fact"
        if KINDS.get(kind, KINDS["fact"]) is None or scale(kind)[0] == INF:
            continue
        if s.id in ended:
            observed.setdefault(kind, []).append(ended[s.id])
        else:
            start = parse_moment(s.claim.get("valid_from") or s.claim.get("ts"))
            if start:
                open_.setdefault(kind, []).append((moment - start).total_seconds() / 86_400)
    fitted = {}
    for kind, days in observed.items():
        shape = scale(kind)[1]
        eta = fit_scale(days, open_.get(kind, []), shape)
        if eta:
            fitted[kind] = (round(eta, 3), shape)
    return fitted


def _hlc_moment(hlc: str) -> datetime | None:
    from .stamps import hlc_datetime
    return hlc_datetime(hlc) if hlc else None
