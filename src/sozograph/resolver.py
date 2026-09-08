"""
The truth layer: deterministic, local, and never calls a model.

Everything here runs in memory before any network request. Given the same
inputs it produces the same passport, which is what makes the state portable
and the behaviour auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

from .dedupe import DedupeReport, Verdict, find_match
from .retrieve import tokenize
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
from .utils import normalize_key, sha256_json


@dataclass
class ResolveStats:
    facts_upserted: int = 0
    prefs_upserted: int = 0
    entities_merged: int = 0
    open_loops_added: int = 0
    episodes_added: int = 0
    observations_added: int = 0
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
            "observations_added": self.observations_added,
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
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if a is b or a == b:
        return True
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
    merged = existing.model_copy(deep=True)
    merged.type, merged.aliases = typ, aliases
    _merge_provenance(merged, incoming)
    return merged


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
            and existing.subject == candidate.subject
            and existing.ts_old == candidate.ts_old
            and existing.ts_new == candidate.ts_new
            and existing.status == candidate.status
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
    incoming = incoming.model_copy(deep=True)
    incoming.key = normalize_key(incoming.key)
    scoped = [it for it in items if it.subject.casefold() == incoming.subject.casefold()]
    match = find_match(incoming.key, [it.key for it in scoped])
    stats.dedupe.record(match)
    if not match.is_merge:
        items.append(incoming)
        return True, None
    if match.verdict is Verdict.MERGE:
        stats.keys_deduped += 1
    group = [it for it in scoped if it.key == match.existing]
    incoming.key = normalize_key(match.existing)
    latest = max(it.ts for it in group)
    for current in group:
        current.key = incoming.key
        if _value_equal(current.value, incoming.value):
            _merge_provenance(current, incoming)
            if incoming.ts > latest:
                for other in group:
                    if other is not current:
                        _record_contradiction(contradictions, _change(other, incoming))
                        items.remove(other)
                current.status = "active"
            if incoming.ts > current.ts:
                current.ts, current.source = incoming.ts, incoming.source
            current.confidence = max(current.confidence, incoming.confidence)
            return False, None
    changes = []
    for current in group:
        if incoming.ts == latest:
            current.status = incoming.status = "disputed"
            # A canonical display order does not select a winner.
            left, right = sorted([current, incoming], key=lambda x: sha256_json(x.value))
            change = _change(left, right, disputed=True)
        elif incoming.ts > latest:
            change = _change(current, incoming)
        else:
            change = _change(incoming, current)
        if _record_contradiction(contradictions, change):
            changes.append(change)
    if incoming.ts > latest:
        for current in group:
            items.remove(current)
        incoming.status = "active"
        items.append(incoming)
    elif incoming.ts == latest:
        items.append(incoming)
    return incoming.ts >= latest, changes[0] if changes else None


def _change(old, new, *, disputed=False):
    return Contradiction(key=old.key, subject=old.subject, old=old.value, new=new.value,
                         ts_old=old.ts, ts_new=new.ts, source_old=old.source,
                         source_new=new.source, status="disputed" if disputed else "resolved")


def _merge_provenance(existing, incoming):
    existing.source_ids = sorted(set(existing.source_ids + incoming.source_ids
                                      + [s for s in [getattr(existing, "source", None), getattr(incoming, "source", None)] if s]))
    refs = {sha256_json(e.to_compact()): e for e in existing.evidence + incoming.evidence}
    existing.evidence = [refs[k] for k in sorted(refs)]


def _upsert_open_loop(existing: list[OpenLoop], incoming: OpenLoop) -> bool:
    key = _loop_key(incoming.item)
    if not key:
        return False
    for i, loop in enumerate(existing):
        if _loop_key(loop.item) == key and loop.subject.casefold() == incoming.subject.casefold():
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


def _observation_key(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@lru_cache(maxsize=8192)
def _content_tokens(text: str) -> frozenset[str]:
    """Content tokens of a statement, cached: the near-dup scan is O(n) per
    incoming observation over the whole existing set."""
    return frozenset(tokenize(text))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _shares_participant(a: Observation, b: Observation) -> bool:
    left = {p.lower() for p in a.participants}
    right = {p.lower() for p in b.participants}
    if not left and not right:
        return True
    return bool(left & right)


#: A paraphrase duplicate is only ever skipped on a conjunction of evidence.
#: Token overlap alone would fold "Melanie hiked Kalambo Falls" into "Melanie
#: hiked Nyika plateau" at 0.75; the participant and date guards are what make
#: the skip safe. Carrying a duplicate stays the cheaper error.
_NEAR_DUP_JACCARD = 0.85


def _add_observation(existing: list[Observation], incoming: Observation) -> bool:
    """
    Append an observation unless the same statement is already held.

    Observations are append-only: there is no belief to overwrite, only a
    record of what was seen. Exact-text duplicates merge outright, so
    re-ingesting the same conversation does not double the recall layer.
    A near-duplicate (near-identical token set) merges only with the same
    participants and the same event date; anything less keeps both records,
    since a false skip deletes real recall and an extra record costs one
    ranking slot. Participants union when a statement recurs with new names
    attached.
    """
    key = _observation_key(incoming.text)
    if not key:
        return False
    for obs in existing:
        # Explicit dates must agree. Undated copies are only equivalent on
        # their discussion date; otherwise recurrence is ambiguous.
        if (obs.when, obs.when_end, obs.precision) != (incoming.when, incoming.when_end, incoming.precision):
            continue
        if not obs.when and obs.ts.date() != incoming.ts.date():
            continue
        exact = _observation_key(obs.text) == key
        near = (safe_observation_paraphrase(obs.text, incoming.text)
                and set(p.casefold() for p in obs.participants) == set(p.casefold() for p in incoming.participants)
                and bool(obs.participants))
        if exact or near:
            _merge_provenance(obs, incoming)
            _union_participants(obs, incoming)
            if incoming.ts < obs.ts:
                obs.ts, obs.source = incoming.ts, incoming.source
            return False
    existing.append(incoming.model_copy(deep=True))
    return True


def safe_observation_paraphrase(a: str, b: str) -> bool:
    # Conservative: no changed content token, number, polarity, directional
    # relation, or named-person order may be discarded as a paraphrase.
    if tokenize(a) != tokenize(b):
        return False
    guards = {"not", "no", "never", "without", "before", "after", "from", "until"}
    if set(tokenize(a, keep_stopwords=True)) & guards != set(tokenize(b, keep_stopwords=True)) & guards:
        return False
    def names(text):
        return re.findall(r"\b[A-Z][a-z]+\b", text)
    return names(a) == names(b)


def _union_participants(obs: Observation, incoming: Observation) -> None:
    merged = list(obs.participants)
    seen = {p.lower() for p in merged}
    for p in incoming.participants:
        if p.lower() not in seen:
            seen.add(p.lower())
            merged.append(p)
    obs.participants = merged


def merge_passport_update(
    base: Passport,
    *,
    facts: list[Fact] | None = None,
    prefs: list[Preference] | None = None,
    entities: list[Entity] | None = None,
    open_loops: list[OpenLoop] | None = None,
    episodes: list[Episode] | None = None,
    observations: list[Observation] | None = None,
    now: datetime | None = None,
) -> tuple[Passport, ResolveStats]:
    """Merge an extraction update into a passport. Deterministic throughout."""
    base.assert_supported()
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

    for observation in observations or []:
        stats.observations_added += int(_add_observation(base.observations, observation))

    if stats.dedupe:
        audit = base.meta.setdefault("dedupe", {})
        for bucket, rows in stats.dedupe.to_dict().items():
            audit.setdefault(bucket, []).extend(rows)

    _sort(base)
    base.touch(now)
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
    base.facts.sort(key=lambda x: (x.subject.casefold(), normalize_key(x.key), -x.ts.timestamp(), sha256_json(x.value)))
    base.prefs.sort(key=lambda x: (x.subject.casefold(), normalize_key(x.key), -x.ts.timestamp(), sha256_json(x.value)))
    base.entities.sort(key=lambda x: (_entity_key(x.name), x.type))
    base.open_loops.sort(key=lambda x: (-x.ts.timestamp(), _loop_key(x.item)))
    base.contradictions.sort(key=lambda x: (x.subject.casefold(), normalize_key(x.key), -x.ts_new.timestamp(), sha256_json([x.old, x.new])))
    base.episodes.sort(key=lambda x: (x.ts.timestamp(), x.id))
    base.observations.sort(key=lambda x: (-x.ts.timestamp(), _observation_key(x.text)))
