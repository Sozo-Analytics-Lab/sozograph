"""
The truth layer: deterministic, local, and never calls a model.

Everything here runs in memory before any network request. Given the same
inputs it produces the same passport, which is what makes the state portable
and the behaviour auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dedupe import DedupeReport, Verdict, find_match
from .schema import (
    Contradiction,
    Entity,
    Episode,
    Fact,
    OpenLoop,
    Passport,
    Preference,
)
from .utils import normalize_key


@dataclass
class ResolveStats:
    facts_upserted: int = 0
    prefs_upserted: int = 0
    entities_merged: int = 0
    open_loops_added: int = 0
    episodes_added: int = 0
    contradictions_added: int = 0
    keys_deduped: int = 0
    dedupe: DedupeReport = field(default_factory=DedupeReport)

    def to_dict(self) -> dict[str, int]:
        return {
            "facts_upserted": self.facts_upserted,
            "prefs_upserted": self.prefs_upserted,
            "entities_merged": self.entities_merged,
            "open_loops_added": self.open_loops_added,
            "episodes_added": self.episodes_added,
            "contradictions_added": self.contradictions_added,
            "keys_deduped": self.keys_deduped,
        }


def _value_equal(a: Any, b: Any) -> bool:
    """
    Compare two values the way a person would.

    The old comparison was `str.strip()` equality, which meant "Direct" and
    "direct" were recorded as a contradiction and the value flip-flopped on
    every ingest. Numbers written as text ("7" and 7) had the same problem.
    """
    if a is b or a == b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    if isinstance(a, str) and isinstance(b, str):
        return " ".join(a.lower().split()) == " ".join(b.lower().split())
    if isinstance(a, str) or isinstance(b, str):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return False


def _entity_key(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def _loop_key(item: str) -> str:
    return " ".join((item or "").strip().lower().split())


def _merge_entity(existing: Entity, incoming: Entity) -> Entity:
    aliases = list(existing.aliases)
    seen = {a.lower() for a in aliases}

    def add(value: str) -> None:
        value = (value or "").strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            aliases.append(value)

    if _entity_key(existing.name) != _entity_key(incoming.name):
        add(incoming.name)
    for alias in incoming.aliases:
        add(alias)

    typ = existing.type
    if typ == "other" and incoming.type != "other":
        typ = incoming.type
    return Entity(name=existing.name, type=typ, aliases=aliases)


def _record_contradiction(
    contradictions: list[Contradiction], candidate: Contradiction
) -> bool:
    """
    Append a contradiction unless the same change is already recorded.

    These were append-only. Re-ingesting the same conversation re-appended the
    identical entry every time, so the section grew without bound and the
    "recent updates" block filled with the same line repeated.
    """
    for existing in contradictions:
        if (
            existing.key == candidate.key
            and _value_equal(existing.old, candidate.old)
            and _value_equal(existing.new, candidate.new)
        ):
            if candidate.ts_new > existing.ts_new:
                existing.ts_new = candidate.ts_new
                existing.source_new = candidate.source_new
            return False
    contradictions.append(candidate)
    return True


def _upsert_kv(
    *,
    items: list[Any],
    incoming: Any,
    contradictions: list[Contradiction],
    stats: ResolveStats,
) -> tuple[bool, Contradiction | None]:
    """
    Insert or update one fact or preference, resolving conflicts by time.

    Key identity runs through the dedupe tiers: exact match first, then a
    guarded fuzzy match. A pair the polarity guard blocks becomes a new key
    rather than silently overwriting the belief it resembles.
    """
    incoming.key = normalize_key(incoming.key)
    match = find_match(incoming.key, [it.key for it in items])
    stats.dedupe.record(match)

    if not match.is_merge:
        items.append(incoming)
        return True, None

    if match.verdict is Verdict.MERGE:
        stats.keys_deduped += 1

    target = match.existing
    idx = next(i for i, it in enumerate(items) if it.key == target)
    current = items[idx]
    # Canonicalize the stored key on every match. A passport built before the
    # normalizer was unified can hold "Tone"; without this it lingers forever
    # and renders differently from the key it merges under.
    current.key = normalize_key(current.key)
    incoming.key = current.key

    if _value_equal(current.value, incoming.value):
        if incoming.ts > current.ts:
            current.ts = incoming.ts
            current.source = incoming.source
        current.confidence = max(float(current.confidence), float(incoming.confidence))
        items[idx] = current
        return False, None

    if incoming.ts >= current.ts:
        change = Contradiction(
            key=current.key,
            old=current.value,
            new=incoming.value,
            ts_old=current.ts,
            ts_new=incoming.ts,
            source_old=current.source,
            source_new=incoming.source,
        )
        added = _record_contradiction(contradictions, change)
        items[idx] = incoming
        return True, (change if added else None)

    # The incoming value is older than what is stored, so it does not win.
    change = Contradiction(
        key=current.key,
        old=incoming.value,
        new=current.value,
        ts_old=incoming.ts,
        ts_new=current.ts,
        source_old=incoming.source,
        source_new=current.source,
    )
    added = _record_contradiction(contradictions, change)
    return False, (change if added else None)


def _upsert_open_loop(existing: list[OpenLoop], incoming: OpenLoop) -> bool:
    key = _loop_key(incoming.item)
    if not key:
        return False
    for i, loop in enumerate(existing):
        if _loop_key(loop.item) == key:
            if incoming.ts > loop.ts:
                existing[i] = incoming
                return True
            return False
    existing.append(incoming)
    return True


def _upsert_episode(existing: list[Episode], incoming: Episode) -> bool:
    for i, ep in enumerate(existing):
        if ep.id == incoming.id:
            existing[i] = incoming
            return False
    existing.append(incoming)
    return True


def merge_passport_update(
    base: Passport,
    *,
    facts: list[Fact] | None = None,
    prefs: list[Preference] | None = None,
    entities: list[Entity] | None = None,
    open_loops: list[OpenLoop] | None = None,
    episodes: list[Episode] | None = None,
) -> tuple[Passport, ResolveStats]:
    """Merge an extraction update into a passport. Deterministic throughout."""
    stats = ResolveStats()

    for fact in facts or []:
        updated, change = _upsert_kv(
            items=base.facts, incoming=fact,
            contradictions=base.contradictions, stats=stats,
        )
        stats.facts_upserted += int(updated)
        stats.contradictions_added += int(change is not None)

    for pref in prefs or []:
        updated, change = _upsert_kv(
            items=base.prefs, incoming=pref,
            contradictions=base.contradictions, stats=stats,
        )
        stats.prefs_upserted += int(updated)
        stats.contradictions_added += int(change is not None)

    _merge_entities(base, entities or [], stats)

    for loop in open_loops or []:
        stats.open_loops_added += int(_upsert_open_loop(base.open_loops, loop))

    for episode in episodes or []:
        stats.episodes_added += int(_upsert_episode(base.episodes, episode))

    if stats.dedupe:
        audit = base.meta.setdefault("dedupe", {})
        for bucket, rows in stats.dedupe.to_dict().items():
            audit.setdefault(bucket, []).extend(rows)

    _sort(base)
    base.touch()
    return base, stats


def _merge_entities(base: Passport, entities: list[Entity], stats: ResolveStats) -> None:
    by_key: dict[str, Entity] = {_entity_key(e.name): e for e in base.entities}
    alias_index: dict[str, str] = {}
    for entity in base.entities:
        key = _entity_key(entity.name)
        for alias in entity.aliases:
            alias_index[_entity_key(alias)] = key

    for incoming in entities:
        incoming_key = _entity_key(incoming.name)
        target = None
        if incoming_key in by_key:
            target = incoming_key
        elif incoming_key in alias_index:
            target = alias_index[incoming_key]
        else:
            for alias in incoming.aliases:
                alias_key = _entity_key(alias)
                if alias_key in by_key:
                    target = alias_key
                    break
                if alias_key in alias_index:
                    target = alias_index[alias_key]
                    break

        if target is None:
            base.entities.append(incoming)
            by_key[incoming_key] = incoming
            for alias in incoming.aliases:
                alias_index[_entity_key(alias)] = incoming_key
        else:
            merged = _merge_entity(by_key[target], incoming)
            by_key[target] = merged
            for i, entity in enumerate(base.entities):
                if _entity_key(entity.name) == target:
                    base.entities[i] = merged
                    break
            for alias in merged.aliases:
                alias_index[_entity_key(alias)] = target
        stats.entities_merged += 1


def _sort(base: Passport) -> None:
    """Stable ordering, so the serialized passport is byte-comparable."""
    base.facts.sort(key=lambda x: (normalize_key(x.key), -x.ts.timestamp()))
    base.prefs.sort(key=lambda x: (normalize_key(x.key), -x.ts.timestamp()))
    base.entities.sort(key=lambda x: (_entity_key(x.name), x.type))
    base.open_loops.sort(key=lambda x: (-x.ts.timestamp(), _loop_key(x.item)))
    base.contradictions.sort(key=lambda x: (normalize_key(x.key), -x.ts_new.timestamp()))
    base.episodes.sort(key=lambda x: (x.ts.timestamp(), x.id))
