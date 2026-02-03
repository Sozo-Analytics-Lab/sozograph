from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator


JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    # Always serialize as ISO-8601 with timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, description="Normalized fact key, e.g. 'role', 'location_city'")
    value: JSONValue = Field(..., description="Current value (JSON-safe). Prefer scalars in v1.")
    ts: datetime = Field(default_factory=utcnow, description="Timestamp when this fact became true")
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="0..1 confidence of extraction")
    source: str = Field(..., min_length=1, description="Source reference id, e.g. 't1'")

    @field_validator("key")
    @classmethod
    def _strip_key(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("key cannot be empty")
        return v

    def to_compact(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "ts": _iso(self.ts),
            "confidence": float(self.confidence),
            "source": self.source,
        }


class Preference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, description="Normalized preference key, e.g. 'tone', 'language'")
    value: JSONValue = Field(..., description="Preferred value (JSON-safe). Prefer scalars in v1.")
    ts: datetime = Field(default_factory=utcnow)
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    source: str = Field(..., min_length=1)

    @field_validator("key")
    @classmethod
    def _strip_key(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("key cannot be empty")
        return v

    def to_compact(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "ts": _iso(self.ts),
            "confidence": float(self.confidence),
            "source": self.source,
        }


EntityType = Literal[
    "person",
    "organization",
    "project",
    "product",
    "place",
    "tool",
    "skill",
    "concept",
    "other",
]


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    type: EntityType = Field("other")
    aliases: List[str] = Field(default_factory=list)

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, v: List[str]) -> List[str]:
        # de-dupe while preserving order
        seen = set()
        out: List[str] = []
        for a in v or []:
            a2 = (a or "").strip()
            if not a2:
                continue
            key = a2.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(a2)
        return out

    def to_compact(self) -> Dict[str, Any]:
        d = {"name": self.name, "type": self.type}
        if self.aliases:
            d["aliases"] = list(self.aliases)
        return d


class OpenLoop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(..., min_length=1, description="Unresolved question / TODO / missing detail")
    ts: datetime = Field(default_factory=utcnow)
    source: str = Field(..., min_length=1)

    def to_compact(self) -> Dict[str, Any]:
        return {"item": self.item, "ts": _iso(self.ts), "source": self.source}


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, description="Fact/preference key that changed")
    old: JSONValue = Field(..., description="Previous value")
    new: JSONValue = Field(..., description="New value")
    ts_old: datetime = Field(..., description="Timestamp of old value")
    ts_new: datetime = Field(..., description="Timestamp of new value")
    source_old: str = Field(..., min_length=1)
    source_new: str = Field(..., min_length=1)

    def to_compact(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "old": self.old,
            "new": self.new,
            "ts_old": _iso(self.ts_old),
            "ts_new": _iso(self.ts_new),
            "source_old": self.source_old,
            "source_new": self.source_new,
        }


SourceKind = Literal["transcript", "firestore", "rtdb", "supabase", "chat", "form", "unknown"]


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Short id used in Fact.source, e.g. 't1'")
    kind: SourceKind = Field("unknown")
    ts: datetime = Field(default_factory=utcnow)
    hash: Optional[str] = Field(None, description="sha256 hash of canonicalized input payload")
    source: Optional[str] = Field(None, description="Human-readable source pointer, e.g. 'firestore:/apps/abc'")

    def to_compact(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "kind": self.kind, "ts": _iso(self.ts)}
        if self.hash:
            d["hash"] = self.hash
        if self.source:
            d["source"] = self.source
        return d


class Passport(BaseModel):
    """
    Portable cognitive snapshot (SozoGraph v1).
    This object is the stable contract between ingestion + any agent context injection.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field("1.0", description="SozoGraph passport schema version")
    updated_at: datetime = Field(default_factory=utcnow)
    user_key: Optional[str] = Field(None, description="Optional stable user identifier")

    facts: List[Fact] = Field(default_factory=list)
    prefs: List[Preference] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    open_loops: List[OpenLoop] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)

    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata (non-memory)")

    def to_compact_dict(self) -> Dict[str, Any]:
        """
        JSON-serializable dict with ISO timestamps (safe for storage / transport).
        """
        return {
            "version": self.version,
            "updated_at": _iso(self.updated_at),
            **({"user_key": self.user_key} if self.user_key else {}),
            "facts": [f.to_compact() for f in self.facts],
            "prefs": [p.to_compact() for p in self.prefs],
            "entities": [e.to_compact() for e in self.entities],
            "open_loops": [o.to_compact() for o in self.open_loops],
            "contradictions": [c.to_compact() for c in self.contradictions],
            "sources": [s.to_compact() for s in self.sources],
            **({"meta": self.meta} if self.meta else {}),
        }

    def upsert_source(self, src: SourceRef) -> None:
        """
        Add/replace a source by id (helps keep sources list unique).
        """
        for i, existing in enumerate(self.sources):
            if existing.id == src.id:
                self.sources[i] = src
                return
        self.sources.append(src)

    def touch(self) -> None:
        self.updated_at = utcnow()
