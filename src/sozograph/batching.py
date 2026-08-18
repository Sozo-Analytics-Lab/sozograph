"""
Group interactions into token-bounded segments, one extraction call each.

Extracting per interaction is the single largest cost in a naive memory
pipeline. A LoCoMo conversation runs to roughly 600 turns, so per-turn
extraction means 600 API calls where LightMem needs about 30. Batching turns
into ~1500-token segments brings that to under 20 for the same conversation,
and the model reasons better with a coherent stretch of dialogue than with one
line at a time.

This is what LightMem's topic-segmentation stage buys, minus LLMLingua, minus a
BERT compressor, minus the model weights either would need you to download.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .interaction import Interaction
from .utils import stable_id

#: Characters per token. Deliberately conservative so a segment lands under the
#: budget rather than over it.
CHARS_PER_TOKEN = 3.6

DEFAULT_MAX_TOKENS = 1500
#: A gap this large between interactions almost certainly separates two topics.
DEFAULT_GAP = timedelta(hours=6)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass
class Segment:
    """A contiguous run of interactions extracted in one call."""

    id: str
    interactions: list[Interaction] = field(default_factory=list)
    boundary_reason: str = ""

    @property
    def ts(self) -> datetime:
        return min(i.ts for i in self.interactions)

    @property
    def end_ts(self) -> datetime:
        return max(i.ts for i in self.interactions)

    @property
    def participants(self) -> list[str]:
        seen: dict[str, None] = {}
        for it in self.interactions:
            speaker = it.meta.get("speaker") if it.meta else None
            if isinstance(speaker, str) and speaker.strip():
                seen.setdefault(speaker.strip(), None)
        return list(seen)

    @property
    def source(self) -> str | None:
        for it in self.interactions:
            if it.source:
                return it.source
        return None

    @property
    def type(self) -> str:
        return self.interactions[0].type if self.interactions else "unknown"

    def text(self, *, max_chars: int | None = None) -> str:
        """The segment as one block, speaker-prefixed where known."""
        lines = []
        for it in self.interactions:
            speaker = (it.meta or {}).get("speaker")
            body = it.text.strip()
            lines.append(f"{speaker}: {body}" if isinstance(speaker, str) and speaker else body)
        joined = "\n".join(lines)
        if max_chars is not None and len(joined) > max_chars:
            return joined[: max_chars - 1] + "…"
        return joined

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.text())

    def __len__(self) -> int:
        return len(self.interactions)


def _rendered_size(it: Interaction) -> int:
    """
    How many characters this interaction contributes to the segment text.

    Counting only `len(it.text)` under-measures: the rendered block adds a
    speaker prefix and a newline per turn, so segments overran the budget by
    the width of the prefixes.
    """
    speaker = (it.meta or {}).get("speaker")
    prefix = len(speaker) + 2 if isinstance(speaker, str) and speaker else 0
    return len(it.text.strip()) + prefix + 1


def _session_of(it: Interaction) -> str | None:
    meta = it.meta or {}
    for key in ("session", "session_id", "segment", "thread_id"):
        value = meta.get(key)
        if value is not None:
            return str(value)
    return None


def segment_interactions(
    interactions: list[Interaction],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    respect_sessions: bool = True,
    gap: timedelta | None = DEFAULT_GAP,
) -> list[Segment]:
    """
    Split interactions into extraction segments.

    A segment closes when the token budget is reached, when the source's own
    session marker changes, or when the time gap between consecutive
    interactions is large enough that they are almost certainly about different
    things. Splitting on a real boundary beats splitting mid-topic, which is
    why the source's own markers win over the token budget.
    """
    if not interactions:
        return []

    ordered = sorted(interactions, key=lambda i: i.ts)
    budget_chars = max(1, int(max_tokens * CHARS_PER_TOKEN))

    segments: list[Segment] = []
    current: list[Interaction] = []
    current_chars = 0
    reason = "start"

    def close(next_reason: str) -> None:
        nonlocal current, current_chars, reason
        if current:
            segments.append(
                Segment(
                    id=stable_id("seg_", [i.text for i in current]),
                    interactions=current,
                    boundary_reason=reason,
                )
            )
        current, current_chars, reason = [], 0, next_reason

    previous: Interaction | None = None
    for it in ordered:
        size = _rendered_size(it)

        if previous is not None:
            if respect_sessions and _session_of(it) != _session_of(previous):
                close("session boundary")
            elif gap is not None and (it.ts - previous.ts) > gap:
                close("time gap")
            elif current_chars + size > budget_chars and current:
                close("token budget")

        current.append(it)
        current_chars += size
        previous = it

        # A single oversized interaction becomes its own segment.
        if current_chars > budget_chars and len(current) == 1:
            close("oversized interaction")
            previous = it

    close("end")
    return segments


def plan(
    interactions: list[Interaction],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, float]:
    """
    Report what batching will cost before running it.

    Useful for a sanity check before spending money on a long history.
    """
    segments = segment_interactions(interactions, max_tokens=max_tokens)
    tokens = [s.estimated_tokens() for s in segments]
    return {
        "interactions": len(interactions),
        "segments": len(segments),
        "api_calls": len(segments),
        "calls_saved_vs_per_interaction": max(0, len(interactions) - len(segments)),
        "estimated_input_tokens": sum(tokens),
        "mean_segment_tokens": (sum(tokens) / len(tokens)) if tokens else 0.0,
        "max_segment_tokens": max(tokens) if tokens else 0,
    }
