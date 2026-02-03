from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


InteractionType = str
# Examples (not enforced as enum to keep v1 flexible):
# "transcript", "chat", "firestore", "rtdb", "supabase", "form", "note", "unknown"


class Interaction(BaseModel):
    """
    Canonical internal representation of any ingested input.

    The extractor ONLY sees Interaction.text (+ minimal metadata).
    Raw objects never go directly to the LLM.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(
        None,
        description="Optional stable identifier for this interaction (doc id, hash, etc.)",
    )

    ts: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of the interaction/event (best-effort)",
    )

    type: InteractionType = Field(
        "unknown",
        description="Origin/type of interaction (transcript, firestore, rtdb, supabase, etc.)",
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Human-readable summary or transcript text used for reasoning",
    )

    source: Optional[str] = Field(
        None,
        description="Human-readable source pointer, e.g. 'firestore:/applications/abc'",
    )

    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw input payload (kept for hashing / evidence, not for LLM)",
    )

    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra metadata (non-memory, non-LLM)",
    )

    def short_text(self, max_chars: int = 4000) -> str:
        """
        Return a truncated version of text safe for prompt inclusion.
        """
        if len(self.text) <= max_chars:
            return self.text
        return self.text[: max_chars - 1] + "…"
