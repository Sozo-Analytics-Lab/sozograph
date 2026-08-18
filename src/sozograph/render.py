"""
Render a passport into the text an agent actually reads.

One build path, parameterized by section caps. The previous version duplicated
its entire body inside a nested closure so that budget trimming could re-run
it, which meant every format change had to be made twice by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .retrieve import rank
from .schema import Contradiction, Entity, Episode, Fact, OpenLoop, Passport, Preference
from .utils import normalize_key

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


#: Trimmed in this order when over budget. Episodes go first because the belief
#: state is the part that must survive: losing a fact loses knowledge, while
#: losing an episode loses only detail. Facts have a floor and are trimmed last.
_TRIM_ORDER = (
    ("episodes", 0),
    ("contradictions", 0),
    ("entities", 3),
    ("open_loops", 2),
    ("prefs", 5),
    ("facts", 8),
)


def _val_to_str(value: Any, max_len: int = 220) -> str:
    if value is None:
        s = "null"
    elif isinstance(value, bool):
        s = "true" if value else "false"
    elif isinstance(value, (int, float)):
        s = str(value)
    elif isinstance(value, str):
        s = value.strip()
    else:
        s = str(value)
    return s[: max_len - 1] + "…" if len(s) > max_len else s


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
    stamps += [c.ts_new.timestamp() for c in passport.contradictions]
    stamps.append(passport.updated_at.timestamp())
    return max(stamps), min(stamps)


def _select(items: list[Any], query: str | None, prior, limit: int,
            text_of=None) -> list[Any]:
    if limit <= 0 or not items:
        return []
    if text_of is None:
        # Facts, prefs and loops are not query-ranked: the belief state goes in
        # whole. Only ordering changes, so that a trim keeps the best ones.
        scored = rank(items, None, text_of=lambda x: "", limit=limit, prior=prior)
    else:
        scored = rank(items, query, text_of=text_of, limit=limit, prior=prior)
    return [s.item for s in scored]


def _build(passport: Passport, caps: Caps, query: str | None, header: str) -> list[str]:
    now, oldest = _bounds(passport)
    kv_prior = _kv_prior(now, oldest)
    t_prior = _time_prior(now, oldest)

    facts: list[Fact] = _select(passport.facts, query, kv_prior, caps.facts)
    prefs: list[Preference] = _select(passport.prefs, query, kv_prior, caps.prefs)
    loops: list[OpenLoop] = _select(passport.open_loops, query, t_prior, caps.open_loops)
    changes: list[Contradiction] = _select(
        passport.contradictions, query, t_prior, caps.contradictions
    )
    entities: list[Entity] = list(passport.entities)[: max(0, caps.entities)]
    episodes: list[Episode] = _select(
        passport.episodes,
        query,
        _episode_prior(now, oldest),
        caps.episodes,
        text_of=lambda e: e.search_text(),
    )
    if episodes:
        episodes = sorted(episodes, key=lambda e: e.ts)

    lines: list[str] = [header]
    if passport.user_key:
        lines.append(f"User: {passport.user_key}")
    lines.append(f"Updated: {passport.updated_at.isoformat()}")

    def section(title: str, rows: list[str]) -> None:
        if not rows:
            return
        lines.append("")
        lines.append(title)
        lines.extend(rows)

    section("Facts (current beliefs):",
            [f"- {normalize_key(f.key)}: {_val_to_str(f.value)}" for f in facts])
    section("Preferences:",
            [f"- {normalize_key(p.key)}: {_val_to_str(p.value)}" for p in prefs])
    section("Key entities:",
            [f"- {e.name} ({e.type})" if e.type and e.type != "other" else f"- {e.name}"
             for e in entities])
    section("Open loops:",
            [f"- {_val_to_str(o.item, max_len=240)}" for o in loops])
    section("Recent updates (contradictions resolved by time):",
            [f"- {normalize_key(c.key)} changed: {_val_to_str(c.old)} -> {_val_to_str(c.new)}"
             for c in changes])
    section("What happened:",
            [f"- [{e.ts.date().isoformat()}] {_val_to_str(e.summary, max_len=400)}"
             for e in episodes])
    return lines


def export_context(
    passport: Passport,
    *,
    query: str | None = None,
    budget_chars: int = 3000,
    header: str = "SOZOGRAPH PASSPORT",
    caps: Caps | None = None,
) -> str:
    """
    Render the passport as a context block.

    With a `query`, episodes are ranked against it lexically. Without one they
    are ordered by recency and salience. Facts and preferences are always
    included in full while the budget allows, so a retrieval miss can never
    hide a known fact.
    """
    budget_chars = max(400, int(budget_chars or 3000))
    current = caps or Caps()

    lines = _build(passport, current, query, header)
    if len("\n".join(lines)) <= budget_chars:
        return "\n".join(lines)

    # Shrink the least load-bearing section that still has room to give.
    for _ in range(400):
        for name, floor in _TRIM_ORDER:
            value = getattr(current, name)
            if value > floor:
                step = max(1, value // 4)
                current = replace(current, **{name: max(floor, value - step)})
                break
        else:
            text = "\n".join(lines)
            return text[: budget_chars - 1] + "…"

        lines = _build(passport, current, query, header)
        if len("\n".join(lines)) <= budget_chars:
            return "\n".join(lines)

    text = "\n".join(lines)
    return text[: budget_chars - 1] + "…" if len(text) > budget_chars else text
