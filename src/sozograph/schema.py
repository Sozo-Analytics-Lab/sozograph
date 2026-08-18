from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JSONValue = str | int | float | bool | None | dict[str, Any] | list[Any]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    # Always serialize as ISO-8601 with timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1)
    value: JSONValue
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

    def to_compact(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "ts": _iso(self.ts),
            "confidence": float(self.confidence),
            "source": self.source,
        }


class Preference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1)
    value: JSONValue
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

    def to_compact(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "ts": _iso(self.ts),
            "confidence": float(self.confidence),
            "source": self.source,
        }


#: Single source of truth for entity types. prompts.py builds the JSON Schema
#: enum from this tuple so the wire contract cannot drift from the model.
ENTITY_TYPES = (
    "person",
    "organization",
    "project",
    "product",
    "place",
    "tool",
    "skill",
    "concept",
    "other",
)

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
    aliases: list[str] = Field(default_factory=list)

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, v: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for a in v or []:
            a2 = (a or "").strip()
            if not a2:
                continue
            k = a2.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(a2)
        return out

    def to_compact(self) -> dict[str, Any]:
        d = {"name": self.name, "type": self.type}
        if self.aliases:
            d["aliases"] = list(self.aliases)
        return d


class OpenLoop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(..., min_length=1)
    ts: datetime = Field(default_factory=utcnow)
    source: str = Field(..., min_length=1)

    def to_compact(self) -> dict[str, Any]:
        return {"item": self.item, "ts": _iso(self.ts), "source": self.source}


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    old: JSONValue
    new: JSONValue
    ts_old: datetime
    ts_new: datetime
    source_old: str
    source_new: str

    def to_compact(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "old": self.old,
            "new": self.new,
            "ts_old": _iso(self.ts_old),
            "ts_new": _iso(self.ts_new),
            "source_old": self.source_old,
            "source_new": self.source_new,
        }


class Episode(BaseModel):
    """
    What happened, and when.

    Facts answer "what is true now". A flat key-value belief state throws away
    everything else by construction, which is fatal on multi-hop and temporal
    questions ("what did she say about the painting in session 4"). Episodes
    are compact per-segment summaries produced by the same extraction call, so
    they cost no extra API request.

    They are also what keeps retrieval honest: the belief state is small enough
    to inject in full, so only episodes are ever ranked, and a retrieval miss
    degrades episodic recall rather than losing a fact outright.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    ts: datetime = Field(default_factory=utcnow)
    summary: str = Field(..., min_length=1)
    participants: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    salience: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field(..., min_length=1)

    def to_compact(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "ts": _iso(self.ts),
            "summary": self.summary,
            "salience": round(float(self.salience), 3),
            "source": self.source,
        }
        if self.participants:
            d["participants"] = list(self.participants)
        if self.keywords:
            d["keywords"] = list(self.keywords)
        return d

    def search_text(self) -> str:
        """Everything worth matching a query against."""
        parts = [self.summary, " ".join(self.participants), " ".join(self.keywords)]
        return " ".join(p for p in parts if p)


SourceKind = Literal[
    "transcript",
    "firestore",
    "rtdb",
    "supabase",
    "chat",
    "form",
    "unknown",
]


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: SourceKind = Field("unknown")
    ts: datetime = Field(default_factory=utcnow)
    hash: str | None = None
    source: str | None = None

    def to_compact(self) -> dict[str, Any]:
        d = {"id": self.id, "kind": self.kind, "ts": _iso(self.ts)}
        if self.hash:
            d["hash"] = self.hash
        if self.source:
            d["source"] = self.source
        return d


PASSPORT_VERSION = "2.0"


class Passport(BaseModel):
    """
    A portable memory snapshot.

    This is the whole product: a small JSON object holding what is true
    (`facts`), what is wanted (`prefs`), who and what is involved (`entities`),
    what is unfinished (`open_loops`), what changed (`contradictions`), and
    what happened (`episodes`). It moves between runtimes, databases, and
    client applications as plain JSON, with no vector store to migrate and no
    embedding model to match.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(PASSPORT_VERSION)
    updated_at: datetime = Field(default_factory=utcnow)
    user_key: str | None = None

    facts: list[Fact] = Field(default_factory=list)
    prefs: list[Preference] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)

    meta: dict[str, Any] = Field(default_factory=dict)

    #: Per-interaction merge statistics from the most recent ingest.
    #: Runtime-only: excluded from serialization so it never lands on disk.
    stats: list[Any] = Field(default_factory=list, exclude=True, repr=False)

    @classmethod
    def new(cls) -> Passport:
        """Create an empty passport."""
        return cls()

    # -- key vocabulary ----------------------------------------------------

    def known_keys(self) -> list[str]:
        """
        Every fact and preference key currently held, most useful first.

        Fed back into the extraction prompt so the model reuses an existing key
        instead of coining a synonym for it.
        """
        seen: dict[str, float] = {}
        for item in list(self.facts) + list(self.prefs):
            score = item.ts.timestamp() + float(item.confidence) * 86_400
            if item.key not in seen or score > seen[item.key]:
                seen[item.key] = score
        return [k for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])]

    # -- serialization -----------------------------------------------------

    def to_compact_dict(self) -> dict[str, Any]:
        """The portable form. Empty sections are omitted to keep it small."""
        d: dict[str, Any] = {
            "version": self.version,
            "updated_at": _iso(self.updated_at),
        }
        if self.user_key:
            d["user_key"] = self.user_key
        d["facts"] = [f.to_compact() for f in self.facts]
        d["prefs"] = [p.to_compact() for p in self.prefs]
        d["entities"] = [e.to_compact() for e in self.entities]
        d["open_loops"] = [o.to_compact() for o in self.open_loops]
        d["contradictions"] = [c.to_compact() for c in self.contradictions]
        if self.episodes:
            d["episodes"] = [e.to_compact() for e in self.episodes]
        d["sources"] = [s.to_compact() for s in self.sources]
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Passport:
        """
        Rebuild a passport from `to_compact_dict()` output.

        Tolerant by design: absent sections, a 1.0 passport with no episodes,
        and unknown future keys all load rather than raising. Portability that
        only works one way is not portability.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Passport.from_dict expects a dict, got {type(data).__name__}")

        known = set(cls.model_fields) - {"stats"}
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("version", PASSPORT_VERSION)
        for section in ("facts", "prefs", "entities", "open_loops",
                        "contradictions", "episodes", "sources"):
            payload.setdefault(section, [])
        payload.setdefault("meta", {})

        extra = {k: v for k, v in data.items() if k not in known}
        if extra:
            # Keep anything a newer writer added so a round trip is lossless.
            payload["meta"] = {**payload["meta"], "_unknown": extra}

        passport = cls(**payload)
        passport.version = PASSPORT_VERSION
        return passport

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_compact_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Passport:
        return cls.from_dict(json.loads(text))

    def save(self, path: Any) -> None:
        """
        Write the passport to disk atomically.

        Writes to a sibling temp file and replaces, so an interrupted save
        cannot leave a half-written memory behind.
        """
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        os.replace(tmp, target)

    @classmethod
    def load(cls, path: Any) -> Passport:
        """Read a passport from disk. Accepts 1.0 and 2.0 files."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- convenience -------------------------------------------------------

    def context(
        self,
        *,
        query: str | None = None,
        budget_chars: int = 3000,
        header: str = "SOZOGRAPH PASSPORT",
    ) -> str:
        """
        Render this passport as a context block for a prompt.

        Lives on the model itself so reading your own memory needs no engine,
        no key, and no network.
        """
        from .render import export_context

        return export_context(self, query=query, budget_chars=budget_chars, header=header)

    def token_estimate(self) -> int:
        """Rough token count of the serialized passport (~4 chars per token)."""
        return max(1, len(self.to_json(indent=None)) // 4)

    def is_empty(self) -> bool:
        return not (self.facts or self.prefs or self.entities
                    or self.open_loops or self.episodes)

    def upsert_source(self, src: SourceRef) -> None:
        for i, existing in enumerate(self.sources):
            if existing.id == src.id:
                self.sources[i] = src
                return
        self.sources.append(src)

    def touch(self) -> None:
        self.updated_at = utcnow()
