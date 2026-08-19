"""
Load the LoCoMo benchmark.

LoCoMo is the benchmark LightMem headlines, so it is the one that makes the
comparison meaningful. The dataset is not vendored here: it is Snap's, and
shipping a copy would let this repository's numbers drift from the real thing.
Fetch `locomo10.json` from https://github.com/snap-research/locomo and point
`--data` at it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: LoCoMo's question categories. LightMem evaluates the first four and excludes
#: adversarial, so that is the default here too.
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
DEFAULT_CATEGORIES = (1, 2, 3, 4)

_SESSION_RE = re.compile(r"^session_(\d+)$")
_DATE_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
    "%d %B, %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


@dataclass
class QA:
    question: str
    answer: str
    category: int
    evidence: list[str] = field(default_factory=list)

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"category_{self.category}")


@dataclass
class Conversation:
    sample_id: str
    turns: list[dict[str, Any]]
    qa: list[QA]
    speakers: list[str] = field(default_factory=list)

    @property
    def sessions(self) -> int:
        return len({t.get("session") for t in self.turns})

    def as_text(self) -> str:
        """The whole conversation as one block, for the full-context baseline."""
        lines, current = [], None
        for turn in self.turns:
            if turn.get("session") != current:
                current = turn.get("session")
                lines.append(f"\n--- session {current} ({turn.get('date', '')}) ---")
            lines.append(f"{turn.get('speaker')}: {turn.get('text')}")
        return "\n".join(lines).strip()

    def estimated_tokens(self) -> int:
        return max(1, len(self.as_text()) // 4)


def _parse_date(raw: Any, fallback: datetime) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        return fallback
    text = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return fallback


def _turn_text(turn: dict[str, Any]) -> str:
    """
    Turn text, with the image caption folded in when there is one.

    LoCoMo turns can carry a shared image described by `blip_caption`. The
    images themselves are not released, so the caption is the only signal, and
    dropping it loses questions that depend on what was shown.
    """
    text = str(turn.get("text") or turn.get("clean_text") or "").strip()
    caption = turn.get("blip_caption")
    if isinstance(caption, str) and caption.strip():
        text = f"{text} [shared an image: {caption.strip()}]".strip()
    return text


def load_conversations(
    path: str | Path,
    *,
    categories: tuple[int, ...] = DEFAULT_CATEGORIES,
    limit: int | None = None,
    offset: int = 0,
) -> list[Conversation]:
    """
    Parse locomo10.json into flat turn lists and QA pairs.

    `offset` skips the first N valid conversations before `limit` applies, so
    a quota-constrained run can be split into daily batches (e.g. offset=0
    limit=3, then offset=3 limit=3, ...) without reprocessing what already ran.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo data not found at {path}.\n"
            "Download locomo10.json from https://github.com/snap-research/locomo "
            "(data/locomo10.json) and pass --data <path>."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]

    conversations: list[Conversation] = []
    skipped = 0
    for index, sample in enumerate(raw):
        conv = sample.get("conversation") or {}
        speakers = [s for s in (conv.get("speaker_a"), conv.get("speaker_b")) if s]

        session_numbers = sorted(
            int(m.group(1)) for key in conv if (m := _SESSION_RE.match(str(key)))
        )

        turns: list[dict[str, Any]] = []
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        for order, number in enumerate(session_numbers):
            session = conv.get(f"session_{number}") or []
            when = _parse_date(
                conv.get(f"session_{number}_date_time"), base + timedelta(days=30 * order)
            )
            for position, turn in enumerate(session):
                if not isinstance(turn, dict):
                    continue
                text = _turn_text(turn)
                if not text:
                    continue
                turns.append({
                    "speaker": turn.get("speaker") or "unknown",
                    "text": text,
                    # Spread turns a minute apart so ordering is preserved and
                    # the batcher's time-gap rule still sees session breaks.
                    "ts": (when + timedelta(minutes=position)).isoformat(),
                    "session": str(number),
                    "date": conv.get(f"session_{number}_date_time", ""),
                    "id": turn.get("dia_id") or f"s{number}:{position}",
                })

        questions = []
        for item in sample.get("qa") or []:
            category = int(item.get("category", 0) or 0)
            if categories and category not in categories:
                continue
            answer = item.get("answer", item.get("adversarial_answer", ""))
            if answer is None:
                continue
            questions.append(
                QA(
                    question=str(item.get("question", "")).strip(),
                    answer=str(answer).strip(),
                    category=category,
                    evidence=[str(e) for e in (item.get("evidence") or [])],
                )
            )

        if turns and questions:
            if skipped < offset:
                skipped += 1
                continue
            conversations.append(
                Conversation(
                    sample_id=str(sample.get("sample_id") or f"conv_{index}"),
                    turns=turns,
                    qa=questions,
                    speakers=speakers,
                )
            )
        if limit and len(conversations) >= limit:
            break

    return conversations


def describe(conversations: list[Conversation]) -> dict[str, Any]:
    from collections import Counter

    categories: Counter = Counter()
    for conv in conversations:
        categories.update(q.category_name for q in conv.qa)
    return {
        "conversations": len(conversations),
        "turns": sum(len(c.turns) for c in conversations),
        "sessions": sum(c.sessions for c in conversations),
        "questions": sum(len(c.qa) for c in conversations),
        "mean_tokens_per_conversation": (
            sum(c.estimated_tokens() for c in conversations) / len(conversations)
            if conversations else 0
        ),
        "by_category": dict(categories),
    }
