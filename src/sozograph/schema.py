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
        data = self.model_dump(mode="json", exclude_defaults=True)
        # exclude_defaults compares a default_factory field against a fresh call.
        # A timestamp equal to "now" at dump time (common on coarse clocks)
        # would vanish and reload as a different time, so timestamps always stay.
        for name, field in type(self).model_fields.items():
            if field.default_factory is utcnow and name not in data and getattr(self, name) is not None:
                data[name] = self.model_dump(mode="json", include={name})[name]
        return data


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
    # Passport 3.1 governance. A log-backed passport fills these from the
    # stamps that produced the record; empty values stay off disk.
    stamp_ids: list[str] = Field(default_factory=list)
    author: str = ""
    origin: str = ""
    domains: list[str] = Field(default_factory=list)
    kind: str = ""
    valid_from: str = ""
    valid_until: str = ""
    sensitivity: int | None = Field(None, ge=0, le=3)


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



#: Who a claim came from, lowest trust first. user_confirmed is reserved for
#: the holder's own confirmations; imported for vendor and 2.x imports.
ORIGINS = ("third_party", "tool_result", "imported", "agent_inferred", "user_confirmed", "user_direct")
ORIGIN_RANK = {"third_party": 0, "tool_result": 1, "imported": 2, "agent_inferred": 3,
               "user_confirmed": 4, "user_direct": 4}
OPS = ("assert", "supersede", "correct", "retract", "confirm", "forget", "expire")
STAMP_SECTIONS = ("facts", "prefs", "entities", "open_loops", "episodes", "observations", "contradictions")

Origin = Literal["third_party", "tool_result", "imported", "agent_inferred", "user_confirmed", "user_direct"]
Op = Literal["assert", "supersede", "correct", "retract", "confirm", "forget", "expire"]


class Stamp(PortableModel):
    """
    One write to a log-backed passport.

    The stamp log is the source of truth of a Passport 3.1 file. Every section
    is a deterministic projection of it, so two replicas holding the same
    stamps hold the same beliefs. `id` is a content address; `admission`,
    `sig` and `redacted` sit outside it, because they are the receiving
    replica's verdict, its attestation and the holder's erasure, joined on merge.
    """

    id: str = ""
    op: Op = "assert"
    section: str = ""
    claim: dict[str, Any] = Field(default_factory=dict)
    targets: list[str] = Field(default_factory=list)
    author: str = Field(..., min_length=1)
    visa: str = ""
    origin: Origin = "agent_inferred"
    hlc: str = ""
    seen: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
    erase_sources: list[str] = Field(default_factory=list)
    admission: Literal["accepted", "quarantined"] = "accepted"
    admission_reason: str = ""
    sig: str = ""
    redacted: bool = False


class AgentRecord(PortableModel):
    """An agent that has held a visa: who it is and what it is good at."""

    id: str = Field(..., min_length=1)
    name: str = ""
    vendor: str = ""
    model: str = ""
    home_domains: list[str] = Field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None


class Visa(PortableModel):
    """
    One agent's permission to read and write part of a passport.

    The secret token never touches the file; only its SHA-256 does. The token
    is available once, on the object `issue()` returns.
    """

    id: str = Field(..., min_length=1)
    agent: str = Field(..., min_length=1)
    domains: list[str] = Field(default_factory=lambda: ["general"])
    include_general: bool = True
    write: bool = False
    forget: bool = False
    sensitivity_ceiling: int = Field(1, ge=0, le=2)
    purpose: str = ""
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    token_sha256: str = ""
    _token: str | None = PrivateAttr(default=None)

    @property
    def token(self) -> str | None:
        """The bearer secret, present only on the object issue() returned."""
        return self._token

    def active(self, now: datetime | None = None) -> bool:
        moment = now or utcnow()
        if self.revoked_at is not None and self.revoked_at <= moment:
            return False
        return self.expires_at is None or moment < self.expires_at

    def sees_domain(self, domains: list[str]) -> bool:
        wanted = set(domains or ["general"])
        allowed = set(self.domains)
        if "*" in allowed:
            return True
        if self.include_general:
            allowed.add("general")
        return bool(wanted & allowed)

    @classmethod
    def issue(cls, passport: Passport, *, agent: str, domains: list[str] | None = None,
              write: bool = False, forget: bool = False, sensitivity_ceiling: int = 1,
              purpose: str = "", expires_at: datetime | None = None, name: str = "",
              vendor: str = "", model: str = "", include_general: bool = True,
              now: datetime | None = None) -> Visa:
        from .visa import issue_visa
        return issue_visa(passport, agent=agent, domains=domains, write=write, forget=forget,
                          sensitivity_ceiling=sensitivity_ceiling, purpose=purpose,
                          expires_at=expires_at, name=name, vendor=vendor, model=model,
                          include_general=include_general, now=now)


PASSPORT_VERSION = "3.1"
SUPPORTED_VERSIONS = {"1.0", "2.0", "2.1", "2.2", PASSPORT_VERSION}
#: The shelved ledger preview. Its files round-trip opaquely; nothing mutates them.
RESERVED_VERSIONS = {"3.0"}


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
    _loaded_head: str | None = PrivateAttr(default=None)
    _signer: Any = PrivateAttr(default=None)
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

    #: Passport 3.1. When `stamps` is nonempty it is canonical and every
    #: section above is its projection. An empty log is a plain snapshot.
    holder: dict[str, Any] = Field(default_factory=dict)
    agents: dict[str, AgentRecord] = Field(default_factory=dict)
    visas: list[Visa] = Field(default_factory=list)
    stamps: list[Stamp] = Field(default_factory=list)

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
        if self.holder:
            d["holder"] = deepcopy(self.holder)
        if self.agents:
            d["agents"] = {k: self.agents[k].to_compact() for k in sorted(self.agents)}
        if self.visas:
            d["visas"] = [v.to_compact() for v in self.visas]
        if self.stamps:
            from .stamps import merkle_root
            d["stamps"] = [s.to_compact() for s in self.stamps]
            d["head"] = merkle_root([s.id for s in self.stamps])
        if self.meta:
            d["meta"] = self.meta
        return d

    @property
    def log_backed(self) -> bool:
        """
        True when the stamp log is canonical and sections are its projection.

        A passport becomes log-backed when it has a holder record or any stamp:
        `Passport.issue()` for a new one, `ensure_log()` for an existing snapshot.
        """
        return bool(self.stamps) or bool(self.holder.get("id"))

    @classmethod
    def issue(cls, holder: str = "holder", *, name: str = "", now: datetime | None = None) -> Passport:
        """A new, empty, log-backed passport for one holder."""
        moment = now or utcnow()
        return cls(updated_at=moment, holder={"id": holder, "name": name or holder,
                                              "issued_at": _iso(moment)})

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
        loaded_head = payload.pop("head", None)
        payload["version"] = PASSPORT_VERSION
        passport = cls(**payload)
        passport._loaded_head = loaded_head
        return passport

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
        if self.version in RESERVED_VERSIONS:
            raise ValueError(f"Passport {self.version} is the shelved ledger preview; "
                             "this release reads 1.0 to 2.2 and 3.1. Round-trip only.")
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

    def forget(self, *, source_ids=None, subject=None, contains=None, stamp_ids=None):
        """Erase matching records. A log-backed passport records the erasure as a forget stamp."""
        if self.log_backed or stamp_ids:
            from .protocol import forget
            return forget(self, source_ids=source_ids, subject=subject, contains=contains,
                          stamp_ids=stamp_ids)
        from .lifecycle import forget
        return forget(self, source_ids=source_ids, subject=subject, contains=contains)

    def redact(self, text: str):
        """Forget records and source groups containing the supplied sensitive text."""
        return self.forget(contains=text)

    def set_loop_status(self, item: str, status: str, *, subject: str = "", now=None):
        self.assert_supported()
        if status not in {"open", "completed", "cancelled"}:
            raise ValueError("Invalid loop status")
        if self.log_backed:
            from .protocol import set_loop_status
            set_loop_status(self, item, status, subject=subject, now=now)
            return
        found = False
        for loop in self.open_loops:
            if loop.item.casefold().strip() == item.casefold().strip() and loop.subject.casefold() == subject.casefold():
                loop.status = status
                loop.ts = now or utcnow()
                found = True
        if not found:
            raise KeyError(item)
        self.touch(now)

    # -- Passport 3.1: the stamp protocol -------------------------------------

    def ensure_log(self) -> Passport:
        """Make the stamp log canonical, turning a snapshot into genesis stamps once."""
        from .protocol import ensure_log
        ensure_log(self)
        return self

    def propose(self, stamps, *, visa: Visa | None = None, author: str | None = None,
                origin: str | None = None, now: datetime | None = None, node: str = "local"):
        """Write stamps. Returns accepted, quarantined and rejected items with reasons."""
        from .protocol import propose
        return propose(self, stamps, visa=visa, author=author, origin=origin, now=now, node=node)

    def pending(self) -> list[dict[str, Any]]:
        """Quarantined stamps waiting for the holder."""
        from .protocol import pending
        return pending(self)

    def confirm(self, stamp_ids: list[str], *, now: datetime | None = None):
        """Holder confirms stamps: quarantined ones enter beliefs, accepted ones gain trust."""
        from .protocol import confirm
        return confirm(self, list(stamp_ids), now=now)

    def dismiss(self, stamp_ids: list[str], *, now: datetime | None = None):
        """Holder marks stamps false from the start."""
        from .protocol import dismiss
        return dismiss(self, list(stamp_ids), now=now)

    def history(self, *, subject: str = "", key: str | None = None, item: str | None = None,
                section: str | None = None) -> list[dict[str, Any]]:
        """Every write to one slot, oldest first, with what became of it."""
        from .protocol import history
        return history(self, subject=subject, key=key, item=item, section=section)

    def view(self, visa: Visa, **kwargs):
        """The visa's minimum-necessary context (a RecallResult)."""
        from .visa import view
        return view(self, visa, **kwargs)

    def issue_visa(self, agent: str, **kwargs) -> Visa:
        """Mint a visa for an agent. Its token is on the returned object, once."""
        from .visa import issue_visa
        return issue_visa(self, agent=agent, **kwargs)

    def revoke(self, visa_id: str | None = None, *, everything: bool = False,
               now: datetime | None = None) -> int:
        """Revoke one visa, or every visa at once."""
        from .visa import revoke
        return revoke(self, visa_id, everything=everything, now=now)

    def verify(self, *, signer=None) -> dict[str, Any]:
        """Offline integrity of the log: addresses, targets, head, signatures, projection."""
        from .protocol import verify
        return verify(self, signer=signer)

    @classmethod
    def merge(cls, left: Passport, right: Passport) -> Passport:
        """Union two replicas of one passport. Every order of merging gives the same beliefs."""
        from .protocol import merge
        return merge(left, right)
