from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from .batching import Segment
from .interaction import Interaction
from .prompts import (
    EXTRACTION_SCHEMA,
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACTOR_USER_PROMPT_TEMPLATE,
    format_known_keys,
)
from .providers.base import LLMProvider
from .retrieve import keywords_from
from .schema import Entity, Episode, Fact, OpenLoop, Preference
from .utils import normalize_key, stable_id

_BARE_LITERALS = {"true": True, "false": False, "null": None, "none": None}


def coerce_value(raw: Any) -> Any:
    """
    Turn a wire string back into a JSON scalar where that is unambiguous.

    The wire schema types `value` as a string so it stays valid under OpenAI
    strict mode, which rejects untyped or any-of properties. Numbers and
    booleans survive the round trip through here; everything else stays text.
    """
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return raw
    low = s.lower()
    if low in _BARE_LITERALS:
        return _BARE_LITERALS[low]
    if s[0] in "-0123456789" or s[0] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.7) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


class Extractor:
    """
    Turns interactions into candidate memory updates.

    Provider-agnostic by construction: it holds an LLMProvider and never names
    a vendor, a transport, or an SDK. Swapping engines is a constructor
    argument, not a code change.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def extract(
        self,
        interaction: Interaction,
        *,
        source_id: str,
        known_keys: Iterable[str] | None = None,
    ) -> dict[str, list]:
        """
        Extract facts, prefs, entities and open loops from one interaction.

        `known_keys` is the passport's current key vocabulary. Passing it is
        what stops the model inventing a synonym for a key that already exists.
        """
        prompt = EXTRACTOR_USER_PROMPT_TEMPLATE.format(
            known_keys_block=format_known_keys(list(known_keys or [])),
            interaction_type=interaction.type,
            ts_iso=interaction.ts.isoformat(),
            interaction_text=interaction.short_text(),
        )
        payload = self.provider.complete_json(
            system=EXTRACTOR_SYSTEM_PROMPT,
            user=prompt,
            schema=EXTRACTION_SCHEMA,
            temperature=0.2,
        )
        return self.validate(payload, source_id=source_id, ts=interaction.ts)

    def extract_segment(
        self,
        segment: Segment,
        *,
        known_keys: Iterable[str] | None = None,
        max_chars: int = 12_000,
    ) -> dict[str, list]:
        """
        Extract from a whole segment in one call.

        This is the batching win: a 600-turn conversation costs about 18 calls
        instead of 600, and the model sees a coherent stretch of dialogue
        rather than a single isolated line.
        """
        prompt = EXTRACTOR_USER_PROMPT_TEMPLATE.format(
            known_keys_block=format_known_keys(list(known_keys or [])),
            interaction_type=segment.type,
            ts_iso=segment.ts.isoformat(),
            interaction_text=segment.text(max_chars=max_chars),
        )
        payload = self.provider.complete_json(
            system=EXTRACTOR_SYSTEM_PROMPT,
            user=prompt,
            schema=EXTRACTION_SCHEMA,
            temperature=0.2,
        )
        source_id = stable_id("seg_", segment.id)
        update = self.validate(payload, source_id=source_id, ts=segment.ts)
        update["episodes"] = self._episode(payload, segment, source_id)
        return update

    def _episode(self, payload: dict, segment: Segment, source_id: str) -> list[Episode]:
        raw = payload.get("episode")
        if not isinstance(raw, dict):
            return []
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            return []

        keywords = [k for k in (raw.get("keywords") or []) if isinstance(k, str) and k.strip()]
        if not keywords:
            # Always leave something to match a query against, even when the
            # model returns none.
            keywords = keywords_from(summary)

        participants = [
            p for p in (raw.get("participants") or []) if isinstance(p, str) and p.strip()
        ] or segment.participants

        try:
            return [
                Episode(
                    id=segment.id,
                    ts=segment.ts,
                    summary=summary,
                    participants=participants[:12],
                    keywords=keywords[:12],
                    salience=_clamp(raw.get("salience", 0.5), default=0.5),
                    source=source_id,
                )
            ]
        except (ValidationError, TypeError, ValueError):
            return []

    def validate(self, data: dict, *, source_id: str, ts: Any = None) -> dict[str, list]:
        """
        Validate and normalize a raw extraction payload.

        Timestamps come from the interaction, never from the model. The model
        cannot know when something happened outside the text it was given, and
        asking it to guess produced items the schema then rejected.
        """
        out: dict[str, list] = {
            "facts": [], "prefs": [], "entities": [],
            "open_loops": [], "episodes": [],
        }
        if not isinstance(data, dict):
            return out

        stamp = {"ts": ts} if ts is not None else {}

        for bucket, model in (("facts", Fact), ("prefs", Preference)):
            for item in data.get(bucket) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    out[bucket].append(
                        model(
                            key=normalize_key(item["key"]),
                            value=coerce_value(item.get("value")),
                            confidence=_clamp(item.get("confidence", 0.7)),
                            source=source_id,
                            **stamp,
                        )
                    )
                except (ValidationError, KeyError, TypeError, ValueError):
                    continue

        for item in data.get("entities") or []:
            if not isinstance(item, dict):
                continue
            try:
                out["entities"].append(
                    Entity(
                        name=item["name"],
                        type=item.get("type") or "other",
                        aliases=[a for a in (item.get("aliases") or []) if isinstance(a, str)],
                    )
                )
            except (ValidationError, KeyError, TypeError, ValueError):
                continue

        for item in data.get("open_loops") or []:
            if not isinstance(item, dict):
                continue
            try:
                out["open_loops"].append(OpenLoop(item=item["item"], source=source_id, **stamp))
            except (ValidationError, KeyError, TypeError, ValueError):
                continue

        return out
