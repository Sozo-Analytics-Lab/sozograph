"""
Tier 3: semantic reconciliation of the key vocabulary.

Tiers 0 through 2 are local and free. This one costs a single model call and is
the only tier that can catch the case the others structurally cannot:

    code_style:              "minimal"
    boilerplate_preference:  "low"

Same belief, different key, different value, and a string-distance score of
about 0.51. No amount of threshold tuning reaches it. Semantic reconciliation
is therefore not a fallback for when fuzzy matching fails; it is the only thing
that handles the whitepaper's own motivating example.

It runs offline over the passport's key list rather than per mutation. That is
one request against roughly sixty short strings, not a request per interaction,
and it never touches a vector store.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .providers import LLMProvider, from_env
from .resolver import _record_contradiction, _sort, _value_equal
from .schema import Passport
from .utils import normalize_key

RECONCILE_SYSTEM_PROMPT = """
You are the SozoGraph reconciler.

You are given the key vocabulary of a memory store, with a sample value for
each key. Some keys are different names for the same underlying belief and
should be merged under one canonical key.

Merge two keys only when they record the same property of the same subject.

Do NOT merge:
- Opposites or bounds: budget_min and budget_max, start_date and end_date.
- Different subjects: user_location and office_location.
- Different granularity: city and full_address.
- A general key and a qualified one: budget and travel_budget.
- Anything you are not confident about. A wrong merge destroys a real belief
  and cannot be undone. Leaving a duplicate is the safer error.

Prefer the shorter, more general, more conventional name as the canonical key.
Return an empty list when nothing should be merged. That is a common and
correct answer.
""".strip()

RECONCILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "merges": {
            "type": "array",
            "description": "Groups of keys that record the same belief.",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {
                        "type": "string",
                        "description": "The key to keep. Must be one of the given keys.",
                    },
                    "aliases": {
                        "type": "array",
                        "description": "Keys to fold into the canonical key.",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["canonical", "aliases", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["merges"],
    "additionalProperties": False,
}

RECONCILE_USER_TEMPLATE = """
KEY VOCABULARY:
{vocabulary}
{hints}
Identify only the groups of keys that record the same belief.
""".strip()


@dataclass
class CompactionResult:
    merged: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    keys_before: int = 0
    keys_after: int = 0

    @property
    def keys_removed(self) -> int:
        return self.keys_before - self.keys_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged": self.merged,
            "rejected": self.rejected,
            "keys_before": self.keys_before,
            "keys_after": self.keys_after,
            "keys_removed": self.keys_removed,
        }

    def __repr__(self) -> str:
        return (
            f"<CompactionResult merged={len(self.merged)} "
            f"rejected={len(self.rejected)} keys {self.keys_before}->{self.keys_after}>"
        )


def _vocabulary(passport: Passport, limit: int) -> list[tuple[str, str, str]]:
    """(bucket, key, sample value) for every fact and preference key."""
    rows: list[tuple[str, str, str]] = []
    for bucket, items in (("fact", passport.facts), ("pref", passport.prefs)):
        for item in items:
            sample = str(item.value)
            rows.append((bucket, item.key, sample[:80]))
    return rows[:limit]


def _pending_hints(passport: Passport) -> str:
    """Surface the pairs Tier 2 deliberately deferred."""
    pending = (passport.meta.get("dedupe") or {}).get("pending_review") or []
    if not pending:
        return ""
    seen, lines = set(), []
    for row in pending[-25:]:
        pair = tuple(sorted((str(row.get("incoming")), str(row.get("existing")))))
        if pair in seen or None in pair:
            continue
        seen.add(pair)
        lines.append(f"- {pair[0]} / {pair[1]}")
    if not lines:
        return ""
    return (
        "\nPAIRS FLAGGED AS POSSIBLY EQUIVALENT (judge each on its merits; "
        "several will be genuinely different):\n" + "\n".join(lines) + "\n"
    )


def compact(
    passport: Passport,
    provider: Any | None = None,
    *,
    apply: bool = True,
    max_keys: int = 200,
) -> CompactionResult:
    """
    Reconcile semantically duplicated keys in a passport.

    Set `apply=False` to see what would merge without changing anything.

        result = sozograph.compact(passport)
        print(result.merged)
    """
    result = CompactionResult()
    vocabulary = _vocabulary(passport, max_keys)
    result.keys_before = len({key for _, key, _ in vocabulary})
    result.keys_after = result.keys_before
    if len(vocabulary) < 2:
        return result

    engine: LLMProvider = (
        provider if isinstance(provider, LLMProvider)
        else _resolve_provider(provider)
    )

    payload = engine.complete_json(
        system=RECONCILE_SYSTEM_PROMPT,
        user=RECONCILE_USER_TEMPLATE.format(
            vocabulary="\n".join(f"- {key} ({bucket}) = {value}"
                                 for bucket, key, value in vocabulary),
            hints=_pending_hints(passport),
        ),
        schema=RECONCILE_SCHEMA,
        temperature=0.0,
    )

    known = {key for _, key, _ in vocabulary}
    for group in payload.get("merges") or []:
        if not isinstance(group, dict):
            continue
        canonical = normalize_key(str(group.get("canonical") or ""))
        aliases = [
            normalize_key(str(a))
            for a in (group.get("aliases") or [])
            if isinstance(a, str)
        ]
        aliases = [a for a in aliases if a and a != canonical and a in known]
        reason = str(group.get("reason") or "")

        # The model may only merge keys that exist. A hallucinated canonical
        # would otherwise silently invent a key nothing wrote.
        if not canonical or canonical not in known or not aliases:
            result.rejected.append(
                {"canonical": canonical, "aliases": aliases,
                 "reason": reason, "rejected_because": "key not in vocabulary"}
            )
            continue

        result.merged.append({"canonical": canonical, "aliases": aliases, "reason": reason})

    if apply and result.merged:
        for group in result.merged:
            _apply_merge(passport, group["canonical"], group["aliases"])
        _sort(passport)
        passport.touch()
        audit = passport.meta.setdefault("dedupe", {})
        audit.setdefault("reconciled", []).extend(result.merged)
        # The deferred pairs have now been judged.
        audit.pop("pending_review", None)

    result.keys_after = len({f.key for f in passport.facts} | {p.key for p in passport.prefs})
    return result


def _apply_merge(passport: Passport, canonical: str, aliases: Sequence[str]) -> None:
    """Fold alias keys into the canonical key, newest value winning."""
    for items in (passport.facts, passport.prefs):
        targets = [i for i in items if i.key == canonical]
        if not targets:
            continue
        winner = max(targets, key=lambda i: i.ts)

        for alias in aliases:
            for item in [i for i in items if i.key == alias]:
                if item.ts > winner.ts and not _value_equal(item.value, winner.value):
                    _record_contradiction(
                        passport.contradictions,
                        _contradiction(canonical, winner, item),
                    )
                    winner.value = item.value
                    winner.ts = item.ts
                    winner.source = item.source
                winner.confidence = max(
                    float(winner.confidence), float(item.confidence)
                )
                items.remove(item)

        items[:] = [i for i in items if i.key != canonical or i is winner]


def _contradiction(key: str, old_item: Any, new_item: Any):
    from .schema import Contradiction

    return Contradiction(
        key=key,
        old=old_item.value,
        new=new_item.value,
        ts_old=old_item.ts,
        ts_new=new_item.ts,
        source_old=old_item.source,
        source_new=new_item.source,
    )


def _resolve_provider(spec: Any | None) -> LLMProvider:
    if spec is None:
        return from_env()
    if isinstance(spec, str):
        from .providers import get_provider

        return get_provider(spec)
    raise TypeError(f"Cannot use {type(spec).__name__} as a provider")
