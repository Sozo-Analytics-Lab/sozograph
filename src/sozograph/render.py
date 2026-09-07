"""
Render a passport into the text an agent actually reads.

One build path, parameterized by section caps. The previous version duplicated
its entire body inside a nested closure so that budget trimming could re-run
it, which meant every format change had to be made twice by hand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from .retrieve import EntityGraph, rank, rank_expanded
from .schema import (
    Contradiction,
    Entity,
    Episode,
    Fact,
    Observation,
    OpenLoop,
    Passport,
    Preference,
)
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
    observations: int = 40


#: Trimmed in this order when over budget. Episodes go first because the belief
#: state is the part that must survive: losing a fact loses knowledge, while
#: losing an episode loses only detail. Facts have a floor and are trimmed last.
#: With a query, every cut is relevance-aware, so the trim order matters less
#: than it used to; without one, this ordering still decides.
#: Observations are the recall layer a single-hop question reads from, so they
#: are trimmed late and keep a high floor: episodes (the coarse narrative they
#: supersede) go first, and the belief-state facts are trimmed last of all.
_TRIM_ORDER = (
    ("episodes", 0),
    ("contradictions", 0),
    ("entities", 3),
    ("open_loops", 2),
    ("prefs", 5),
    ("observations", 12),
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
    stamps += [o.ts.timestamp() for o in passport.observations]
    stamps += [c.ts_new.timestamp() for c in passport.contradictions]
    stamps.append(passport.updated_at.timestamp())
    return max(stamps), min(stamps)


#: Queries whose intent is temporal ("when did the hike happen", "what happened
#: in July 2023"). When one is detected, the recalled details are rendered in
#: event-date order, so the model reads a timeline instead of a relevance list.
_TIME_QUERY_RE = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|\bwhen\b|\bhow long\b|\bhow old\b|what day|which day|what time|what year",
    re.IGNORECASE,
)


def _wants_chronology(query: str | None) -> bool:
    return bool(query) and bool(_TIME_QUERY_RE.search(query))


def _obs_date_key(o: Observation) -> str:
    """Event date when known, else discussion time."""
    return o.when or o.ts.date().isoformat()


def _entity_vocabulary(passport: Passport) -> list[str]:
    """
    Every name a query might mention, for entity-expanded retrieval.

    Entity names and aliases plus observation participants. Plain strings,
    matched on word boundaries; no model call, no dependency.
    """
    vocab: list[str] = []
    for e in passport.entities:
        vocab.append(e.name)
        vocab.extend(e.aliases)
    for o in passport.observations:
        vocab.extend(o.participants)
    return vocab


#: Direct hops only. Two names co-occurring in one record is real relational
#: signal; a friend-of-a-friend chain risks pulling in an entire long
#: conversation's cast for one query. Start conservative, widen later if an
#: eval shows it helps.
_GRAPH_HOPS = 1


def _co_occurrence_groups(passport: Passport) -> list[list[str]]:
    """
    Every record's own participant list, for the co-occurrence graph.

    An observation or episode naming two people together is the actual
    relational data multi-hop questions need ("Melanie and Caroline went
    hiking"); it costs nothing extra to capture, since the extractor already
    writes `participants` on both record kinds.
    """
    groups: list[list[str]] = []
    for o in passport.observations:
        if len(o.participants) >= 2:
            groups.append(o.participants)
    for e in passport.episodes:
        if len(e.participants) >= 2:
            groups.append(e.participants)
    return groups


def _select(items: list[Any], query: str | None, prior, limit: int,
            text_of) -> list[Any]:
    """
    Pick a section's entries, relevance-aware when a query is present.

    Below the caps and the budget the whole section renders regardless, so
    ranking only decides what survives a cut. With no query the prior alone
    orders the section, which is the pre-query behaviour exactly.
    """
    if limit <= 0 or not items:
        return []
    scored = rank(items, query, text_of=text_of, limit=limit, prior=prior)
    return [s.item for s in scored]


def _build(
    passport: Passport,
    caps: Caps,
    query: str | None,
    header: str,
    vocabulary: list[str],
    graph: EntityGraph,
) -> list[str]:
    now, oldest = _bounds(passport)
    kv_prior = _kv_prior(now, oldest)
    t_prior = _time_prior(now, oldest)

    facts: list[Fact] = _select(
        passport.facts, query, kv_prior, caps.facts, text_of=lambda f: f.search_text()
    )
    prefs: list[Preference] = _select(
        passport.prefs, query, kv_prior, caps.prefs, text_of=lambda p: p.search_text()
    )
    loops: list[OpenLoop] = _select(
        passport.open_loops, query, t_prior, caps.open_loops, text_of=lambda o: o.search_text()
    )
    changes: list[Contradiction] = _select(
        passport.contradictions,
        query,
        t_prior,
        caps.contradictions,
        text_of=lambda c: c.search_text(),
    )
    entities: list[Entity] = _select(
        passport.entities,
        query,
        None,
        caps.entities,
        text_of=lambda e: e.search_text(),
    )
    observations: list[Observation] = [
        s.item
        for s in rank_expanded(
            passport.observations,
            query,
            text_of=lambda o: o.search_text(),
            limit=caps.observations,
            prior=t_prior,
            vocabulary=vocabulary,
            graph=graph,
            graph_hops=_GRAPH_HOPS,
        )
    ]
    if _wants_chronology(query):
        observations = sorted(observations, key=_obs_date_key)
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
    section("Details recalled:",
            [f"- [{_obs_date_key(o)}] {_val_to_str(o.text, max_len=300)}"
             if o.when else f"- {_val_to_str(o.text, max_len=300)}"
             for o in observations])
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
    budget_chars: int = 6000,
    header: str = "SOZOGRAPH PASSPORT",
    caps: Caps | None = None,
) -> str:
    """
    Render the passport as a context block.

    With a `query`, every section is ranked against it lexically, blended with
    recency and confidence as a prior. Below the caps and the budget each
    section renders in full; ranking only decides what survives a cut, so an
    old but relevant fact now outranks a recent irrelevant one at the edge of
    the cap. Without a query the prior alone orders everything.

    The `Details recalled` section is the observation layer: atomic statements
    of what was said or happened. A single-hop question ("where did the dog
    hide the bone") is usually answered from here, not from the belief-state
    facts, so it is ranked against the query and trimmed late.
    """
    budget_chars = max(400, int(budget_chars or 6000))
    current = caps or Caps()
    # Built once: neither depends on caps, so rebuilding them on every trim
    # retry (up to 400 below) would be pure waste on a passport of any size.
    vocabulary = _entity_vocabulary(passport)
    graph = EntityGraph.build(_co_occurrence_groups(passport))

    lines = _build(passport, current, query, header, vocabulary, graph)
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

        lines = _build(passport, current, query, header, vocabulary, graph)
        if len("\n".join(lines)) <= budget_chars:
            return "\n".join(lines)

    text = "\n".join(lines)
    return text[: budget_chars - 1] + "…" if len(text) > budget_chars else text
