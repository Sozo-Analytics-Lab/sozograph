from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

JSONValue = str | int | float | bool | None | dict[str, Any] | list[Any]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    # Always serialize as ISO-8601 with timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class PortableModel(BaseModel):
    """Storage is extensible; extraction uses its own strict wire schema."""
    model_config = ConfigDict(extra="allow")

    @field_validator("*", mode="after")
    @classmethod
    def _aware(cls, value):
        if isinstance(value, datetime):
            return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)
        return value

    def to_compact(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_defaults=True)


class EvidenceRef(PortableModel):
    source: str
    turn_id: str
    part_id: str = ""
    quote: str = ""
    start: int | None = None
    end: int | None = None
    match: Literal["exact", "candidate"] = "candidate"
    score: float = Field(0.0, ge=0, le=1)


class MemoryRecord(PortableModel):
    evidence: list[EvidenceRef] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class Fact(MemoryRecord):
    model_config = ConfigDict(extra="allow")

    subject: str = ""
    status: Literal["active", "disputed"] = "active"

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

    def search_text(self) -> str:
        """Everything worth matching a query against."""
        return f"{self.subject} {self.key} {self.value}"



class Preference(MemoryRecord):
    model_config = ConfigDict(extra="allow")

    subject: str = ""
    status: Literal["active", "disputed"] = "active"

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

    def search_text(self) -> str:
        """Everything worth matching a query against."""
        return f"{self.subject} {self.key} {self.value}"



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


class Entity(MemoryRecord):
    model_config = ConfigDict(extra="allow")

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

    def search_text(self) -> str:
        """Everything worth matching a query against."""
        parts = [self.name, self.type, " ".join(self.aliases)]
        return " ".join(p for p in parts if p)



class Observation(MemoryRecord):
    """
    One atomic thing that was said or happened, kept verbatim in meaning.

    Facts hold the belief state: what is true now, deduplicated under a key.
    That layer deliberately discards incidental detail ("the hike took two
    hours", "she hid the bone in a slipper") because it is neither stable nor
    keyable. But that incidental detail is exactly what a single-hop recall
    question asks for. An observation is a self-contained, third-person
    statement of one such detail, with relative dates already resolved, so it
    can be matched by a later query and injected on its own.

    It is append-only and never keyed: there is no belief to overwrite, only a
    record of what was observed. Retrieval, not reconciliation, decides what
    surfaces. This is the portable form of the "atomic fact" memory that wins
    on long-conversation recall benchmarks, carrying no embedding and no vector
    store: a plain string ranked by the same pure-Python BM25 as everything
    else.
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., min_length=1)
    ts: datetime = Field(default_factory=utcnow)
    #: When the event happened, as an ISO date, resolved from the text by the
    #: extractor. Empty when the statement carries no date. This is event time,
    #: which is what a temporal question asks about; `ts` is discussion time,
    #: the segment the statement was extracted from.
    when: str = ""
    when_end: str = ""
    precision: Literal["unknown", "day", "month", "year", "range"] = "unknown"
    source: str = Field(..., min_length=1)
    participants: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _date_precision(cls, data):
        from .temporal import event_interval
        if isinstance(data, dict) and data.get("when"):
            interval = event_interval(data["when"], data.get("ts"))
            if interval:
                start, end, precision = interval
                data = {**data, "when": start, "when_end": end, "precision": precision}
        return data

    @field_validator("when", "when_end")
    @classmethod
    def _norm_when(cls, v: str) -> str:
        """Normalize to an ISO date string, or drop whatever cannot be one."""
        s = (v or "").strip()
        if not s:
            return ""
        candidates = [s.replace("Z", "+00:00")]
        # Weaker models write prose dates; accept the common shapes before
        # giving up, since an unparseable date silently loses event time.
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(candidates[0]).date().isoformat()
        except ValueError:
            return ""

    @model_validator(mode="after")
    def _interval(self):
        if self.when_end and (not self.when or self.when_end < self.when):
            raise ValueError("when_end requires an ordered event interval")
        if self.when and self.precision == "unknown":
            self.precision = "range" if self.when_end else "day"
        return self

    @field_validator("participants")
    @classmethod
    def _clean_participants(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        seen = set()
        for p in v or []:
            p2 = (p or "").strip()
            if not p2 or p2.lower() in seen:
                continue
            seen.add(p2.lower())
            out.append(p2)
        return out

    def search_text(self) -> str:
        """Everything worth matching a query against."""
        parts = [self.text, self.when, " ".join(self.participants)]
        return " ".join(p for p in parts if p)



class OpenLoop(MemoryRecord):
    model_config = ConfigDict(extra="allow")

    subject: str = ""
    status: Literal["open", "completed", "cancelled"] = "open"

    item: str = Field(..., min_length=1)
    ts: datetime = Field(default_factory=utcnow)
    source: str = Field(..., min_length=1)

    def search_text(self) -> str:
        """Everything worth matching a query against."""
        return self.item



class Contradiction(MemoryRecord):
    model_config = ConfigDict(extra="allow")

    subject: str = ""
    status: Literal["resolved", "disputed"] = "resolved"

    key: str
    old: JSONValue
    new: JSONValue
    ts_old: datetime
    ts_new: datetime
    source_old: str
    source_new: str

    def search_text(self) -> str:
        """Everything worth matching a query against."""
        return f"{self.key} {self.old} {self.new}"



class Episode(MemoryRecord):
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

    model_config = ConfigDict(extra="allow")

    id: str
    ts: datetime = Field(default_factory=utcnow)
    summary: str = Field(..., min_length=1)
    participants: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    salience: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field(..., min_length=1)


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


class SourceRef(PortableModel):
    model_config = ConfigDict(extra="allow")

    text: str = ""
    turns: list[dict[str, Any]] = Field(default_factory=list)
    turn_ids: list[str] = Field(default_factory=list)
    retention: Literal["none", "excerpts", "full"] = "none"
    coverage: Literal["none", "partial", "full"] = "none"
    parent: str | None = None

    id: str
    kind: SourceKind = Field("unknown")
    ts: datetime = Field(default_factory=utcnow)
    hash: str | None = None
    source: str | None = None



PASSPORT_VERSION = "2.2"
SUPPORTED_VERSIONS = {"1.0", "2.0", "2.1", PASSPORT_VERSION}


class Passport(PortableModel):
    """
    A portable memory snapshot.

    This is the whole product: a small JSON object holding what is true
    (`facts`), what is wanted (`prefs`), who and what is involved (`entities`),
    what is unfinished (`open_loops`), what changed (`contradictions`), what
    happened (`episodes`), and the atomic details worth recalling
    (`observations`). It moves between runtimes, databases, and client
    applications as plain JSON, with no vector store to migrate and no
    embedding model to match.
    """

    model_config = ConfigDict(extra="allow")

    _future_raw: dict[str, Any] | None = PrivateAttr(default=None)
    _future_state: dict[str, Any] | None = PrivateAttr(default=None)
    ingest_report: dict[str, Any] = Field(default_factory=dict, exclude=True)

    version: str = Field(PASSPORT_VERSION)
    updated_at: datetime = Field(default_factory=utcnow)
    user_key: str | None = None

    facts: list[Fact] = Field(default_factory=list)
    prefs: list[Preference] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
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
        if self._future_raw is not None:
            if self.model_dump(mode="json") != self._future_state:
                raise ValueError("Cannot mutate an unsupported passport version")
            return deepcopy(self._future_raw)
        d: dict[str, Any] = {
            **deepcopy(self.model_extra or {}),
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
        if self.observations:
            d["observations"] = [o.to_compact() for o in self.observations]
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

        payload = deepcopy(data)
        version = str(payload.get("version", "1.0"))
        if version not in SUPPORTED_VERSIONS:
            passport = cls(version=version)
            passport._future_raw = payload
            passport._future_state = passport.model_dump(mode="json")
            return passport
        # Earlier writers stashed extension fields under meta._unknown.
        unknown = payload.get("meta", {}).get("_unknown", {})
        if isinstance(unknown, dict):
            for key, value in unknown.items():
                if key not in cls.model_fields:
                    payload.setdefault(key, deepcopy(value))
        payload.pop("stats", None)
        payload.pop("ingest_report", None)
        payload["version"] = PASSPORT_VERSION
        return cls(**payload)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_compact_dict(), indent=indent, ensure_ascii=False, allow_nan=False, separators=(",", ":") if indent is None else None)

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
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                             prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
                name = stream.name
                stream.write(self.to_json())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path: Any) -> Passport:
        """Read a passport from disk. Accepts 1.0 and 2.0 files."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- convenience -------------------------------------------------------

    def context(
        self,
        *,
        query: str | None = None,
        budget_chars: int = 6000,
        header: str = "SOZOGRAPH PASSPORT",
        **kwargs: Any,
    ) -> str:
        """
        Render this passport as a context block for a prompt.

        Lives on the model itself so reading your own memory needs no engine,
        no key, and no network.
        """
        from .render import export_context

        return export_context(self, query=query, budget_chars=budget_chars, header=header, **kwargs)

    def token_estimate(self) -> int:
        """Rough token count of the serialized passport (~4 chars per token)."""
        return max(1, len(self.to_json(indent=None)) // 4)

    def is_empty(self) -> bool:
        return not (self.facts or self.prefs or self.entities
                    or self.open_loops or self.episodes or self.observations)

    def upsert_source(self, src: SourceRef) -> None:
        self.assert_supported()
        for i, existing in enumerate(self.sources):
            if existing.id == src.id:
                self.sources[i] = src
                return
        self.sources.append(src)

    def assert_supported(self) -> None:
        if self.version != PASSPORT_VERSION or self._future_raw is not None:
            raise ValueError(f"Unsupported passport version {self.version}; round-trip only")

    def touch(self, now: datetime | None = None) -> None:
        self.assert_supported()
        self.updated_at = now or utcnow()

    def canonical_json(self) -> str:
        payload = self.to_compact_dict()
        payload.pop("updated_at", None)
        # Operational diagnostics are excluded, never evidence or extension data.
        meta = payload.get("meta", {}).copy()
        for key in ("ingest_failures", "ingest_report"):
            meta.pop(key, None)
        if meta:
            payload["meta"] = meta
        else:
            payload.pop("meta", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def audit(self):
        from .audit import audit
        return audit(self)

    def recall(self, **kwargs):
        from .recall import recall
        return recall(self, **kwargs)

    def forget(self, *, source_ids=None, subject=None, contains=None):
        from .lifecycle import forget
        return forget(self, source_ids=source_ids, subject=subject, contains=contains)

    def redact(self, text: str):
        """Forget records and source groups containing the supplied sensitive text."""
        return self.forget(contains=text)

    def set_loop_status(self, item: str, status: str, *, subject: str = "", now=None):
        self.assert_supported()
        if status not in {"open", "completed", "cancelled"}:
            raise ValueError("Invalid loop status")
        found = False
        for loop in self.open_loops:
            if loop.item.casefold().strip() == item.casefold().strip() and loop.subject.casefold() == subject.casefold():
                loop.status = status
                loop.ts = now or utcnow()
                found = True
        if not found:
            raise KeyError(item)
        self.touch(now)
