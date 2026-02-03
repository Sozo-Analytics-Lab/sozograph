from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    Passport,
    Fact,
    Preference,
    Entity,
    OpenLoop,
    Contradiction,
)
from .utils import normalize_key


def _norm_key(key: str) -> str:
    # Canonical key identity for truth-layer merges
    return (key or "").strip().lower()


@dataclass
class ResolveStats:
    facts_upserted: int = 0
    prefs_upserted: int = 0
    entities_merged: int = 0
    open_loops_added: int = 0
    contradictions_added: int = 0


def _value_equal(a: Any, b: Any) -> bool:
    # JSON-ish equality with simple normalization
    if a is b:
        return True
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


def _entity_key(name: str) -> str:
    return (name or "").strip().lower()


def _merge_entity(existing: Entity, incoming: Entity) -> Entity:
    # Prefer existing name/type; merge aliases + include the other name as alias if different
    aliases = list(existing.aliases)
    seen = {a.lower(): True for a in aliases}

    def add_alias(x: str) -> None:
        x = (x or "").strip()
        if not x:
            return
        k = x.lower()
        if k in seen:
            return
        seen[k] = True
        aliases.append(x)

    # cross-add names as aliases if different
    if existing.name.strip().lower() != incoming.name.strip().lower():
        add_alias(incoming.name)

    for a in incoming.aliases:
        add_alias(a)

    # If types differ and existing is "other", upgrade to incoming type
    typ = existing.type
    if typ == "other" and incoming.type != "other":
        typ = incoming.type

    return Entity(name=existing.name, type=typ, aliases=aliases)


def _upsert_kv_with_temporal_priority(
    *,
    items: List[Any],  # list[Fact] or list[Preference]
    incoming: Any,  # Fact or Preference
    contradictions: List[Contradiction],
    is_fact: bool,
) -> Tuple[bool, Optional[Contradiction]]:
    """
    Upsert by key. If value changes, latest ts wins and we record contradiction.
    Returns (updated, contradiction_or_none).
    """
    # IMPORTANT: normalize incoming key to prevent "Tone" vs "tone" duplication
    key = _norm_key(incoming.key)
    incoming.key = key

    idx = None
    for i, it in enumerate(items):
        if _norm_key(it.key) == key:
            idx = i
            break

    if idx is None:
        items.append(incoming)
        return True, None

    current = items[idx]

    # Always canonicalize stored key once matched (fixes "Tone" lingering forever)
    current.key = key

    # If value same, keep the most recent ts/confidence optionally
    if _value_equal(current.value, incoming.value):
        # Keep the latest ts (if incoming is newer) and max confidence
        if incoming.ts > current.ts:
            current.ts = incoming.ts
            current.source = incoming.source
        if float(incoming.confidence) > float(current.confidence):
            current.confidence = float(incoming.confidence)
        items[idx] = current
        return False, None

    # Value differs: temporal priority
    if incoming.ts >= current.ts:
        # record contradiction old -> new
        c = Contradiction(
            key=key,
            old=current.value,
            new=incoming.value,
            ts_old=current.ts,
            ts_new=incoming.ts,
            source_old=current.source,
            source_new=incoming.source,
        )
        contradictions.append(c)
        items[idx] = incoming
        return True, c

    # Incoming is older: still record contradiction, but do not replace current
    c = Contradiction(
        key=key,
        old=incoming.value,
        new=current.value,
        ts_old=incoming.ts,
        ts_new=current.ts,
        source_old=incoming.source,
        source_new=current.source,
    )
    contradictions.append(c)
    items[idx] = current
    return False, c


def _dedupe_open_loops(existing: List[OpenLoop], incoming: OpenLoop) -> bool:
    """
    Light dedupe: same normalized text -> keep newest.
    Returns True if added/updated.
    """
    norm = " ".join((incoming.item or "").strip().lower().split())
    if not norm:
        return False

    for i, loop in enumerate(existing):
        norm2 = " ".join((loop.item or "").strip().lower().split())
        if norm2 == norm:
            # keep the newest ts
            if incoming.ts > loop.ts:
                existing[i] = incoming
                return True
            return False

    existing.append(incoming)
    return True


def merge_passport_update(
    base: Passport,
    *,
    facts: List[Fact],
    prefs: List[Preference],
    entities: List[Entity],
    open_loops: List[OpenLoop],
) -> Tuple[Passport, ResolveStats]:
    """
    Deterministically merge an extractor update into a passport.
    """
    stats = ResolveStats()

    # Facts
    for f in facts:
        updated, c = _upsert_kv_with_temporal_priority(
            items=base.facts,
            incoming=f,
            contradictions=base.contradictions,
            is_fact=True,
        )
        if updated:
            stats.facts_upserted += 1
        if c is not None:
            stats.contradictions_added += 1

    # Preferences
    for p in prefs:
        updated, c = _upsert_kv_with_temporal_priority(
            items=base.prefs,
            incoming=p,
            contradictions=base.contradictions,
            is_fact=False,
        )
        if updated:
            stats.prefs_upserted += 1
        if c is not None:
            stats.contradictions_added += 1

    # Entities (merge by name key, also check aliases overlap)
    entity_map: Dict[str, Entity] = {_entity_key(e.name): e for e in base.entities}
    # Build alias index to catch "Sozo Graph" vs "SozoGraph"
    alias_index: Dict[str, str] = {}
    for e in base.entities:
        k = _entity_key(e.name)
        for a in e.aliases:
            alias_index[_entity_key(a)] = k

    for inc in entities:
        inc_name_k = _entity_key(inc.name)
        target_k = None

        if inc_name_k in entity_map:
            target_k = inc_name_k
        elif inc_name_k in alias_index:
            target_k = alias_index[inc_name_k]
        else:
            # Try matching by any incoming alias
            for a in inc.aliases:
                ak = _entity_key(a)
                if ak in entity_map:
                    target_k = ak
                    break
                if ak in alias_index:
                    target_k = alias_index[ak]
                    break

        if target_k is None:
            base.entities.append(inc)
            entity_map[inc_name_k] = inc
            # add aliases to index
            for a in inc.aliases:
                alias_index[_entity_key(a)] = inc_name_k
            stats.entities_merged += 1
        else:
            merged = _merge_entity(entity_map[target_k], inc)
            entity_map[target_k] = merged
            # rehydrate base.entities list item
            for i, e in enumerate(base.entities):
                if _entity_key(e.name) == target_k:
                    base.entities[i] = merged
                    break
            # refresh alias index with merged aliases
            for a in merged.aliases:
                alias_index[_entity_key(a)] = target_k
            stats.entities_merged += 1

    # Open loops
    for o in open_loops:
        if _dedupe_open_loops(base.open_loops, o):
            stats.open_loops_added += 1

    # Keep deterministic ordering: sort facts/prefs by key, then ts desc
    base.facts.sort(key=lambda x: (_norm_key(x.key), -x.ts.timestamp()))
    base.prefs.sort(key=lambda x: (_norm_key(x.key), -x.ts.timestamp()))
    base.entities.sort(key=lambda x: (_entity_key(x.name), x.type))
    base.open_loops.sort(key=lambda x: (-x.ts.timestamp(), (x.item or "").lower()))
    base.contradictions.sort(key=lambda x: (_norm_key(x.key), -x.ts_new.timestamp()))

    base.touch()
    return base, stats
