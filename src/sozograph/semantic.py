"""Optional semantic retrieval for Passport 3.

The passport remains the canonical, portable event ledger. This module builds
a disposable vector sidecar whose entries are valid only for one record
revision and one embedding model.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .passport3 import AccessPolicy, MemoryPassport, MemoryRecord

SIDECAR_VERSION = "1.0"
_FIELD_WEIGHTS = {"key": 1.15, "value": 1.0, "text": 1.0, "evidence": 0.9}


class EmbeddingBackend(Protocol):
    """Minimal adapter for local or hosted embedding models."""

    model_id: str

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not values or not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding vectors must be finite and non-zero")
    result = [value / norm for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("embedding vectors must contain finite values")
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimension mismatch")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _record_fields(record: MemoryRecord) -> dict[str, str]:
    fields = record.search_fields()
    return {
        name: value.strip()
        for name, value in fields.items()
        if name in _FIELD_WEIGHTS and value and value.strip() and value.strip() not in {"null", '""'}
    }


@dataclass
class SemanticEntry:
    record_id: str
    revision_hash: str
    vectors: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "revision_hash": self.revision_hash,
            "vectors": self.vectors,
        }


@dataclass(frozen=True)
class SyncStats:
    embedded_records: int = 0
    embedded_fields: int = 0
    reused_records: int = 0
    removed_records: int = 0


@dataclass
class SemanticSidecar:
    model_id: str
    dimensions: int | None = None
    entries: dict[str, SemanticEntry] = field(default_factory=dict)
    version: str = SIDECAR_VERSION

    @classmethod
    def build(
        cls,
        passport: MemoryPassport,
        embedder: EmbeddingBackend,
    ) -> SemanticSidecar:
        sidecar = cls(model_id=str(embedder.model_id))
        sidecar.sync(passport, embedder)
        return sidecar

    def _check_model(self, embedder: EmbeddingBackend) -> None:
        if str(embedder.model_id) != self.model_id:
            raise ValueError(
                f"sidecar model is {self.model_id!r}; embedder is {embedder.model_id!r}"
            )

    def sync(
        self,
        passport: MemoryPassport,
        embedder: EmbeddingBackend,
    ) -> SyncStats:
        """Incrementally rebuild changed revisions and discard stale vectors."""
        self._check_model(embedder)
        current = {record.id: record for record in passport.materialize() if record.id}
        stale = {
            record_id
            for record_id, entry in self.entries.items()
            if record_id not in current
            or current[record_id].revision_hash != entry.revision_hash
        }
        for record_id in stale:
            del self.entries[record_id]

        pending = [record for record_id, record in current.items() if record_id not in self.entries]
        field_jobs: list[tuple[MemoryRecord, str, str]] = []
        for record in pending:
            for name, text in _record_fields(record).items():
                field_jobs.append((record, name, text))

        raw_vectors = embedder.embed_documents([job[2] for job in field_jobs]) if field_jobs else []
        if len(raw_vectors) != len(field_jobs):
            raise ValueError("embedder returned the wrong number of document vectors")

        grouped: dict[str, dict[str, list[float]]] = {}
        for (record, name, _text), raw in zip(field_jobs, raw_vectors, strict=True):
            vector = _normalize(raw)
            if self.dimensions is None:
                self.dimensions = len(vector)
            if len(vector) != self.dimensions:
                raise ValueError("embedder returned inconsistent dimensions")
            grouped.setdefault(str(record.id), {})[name] = vector

        for record in pending:
            vectors = grouped.get(str(record.id), {})
            if vectors:
                self.entries[str(record.id)] = SemanticEntry(
                    record_id=str(record.id),
                    revision_hash=record.revision_hash,
                    vectors=vectors,
                )
        return SyncStats(
            embedded_records=len(grouped),
            embedded_fields=len(field_jobs),
            reused_records=len(current) - len(pending),
            removed_records=len(stale),
        )

    def dense_search(
        self,
        passport: MemoryPassport,
        query: str,
        embedder: EmbeddingBackend,
        *,
        limit: int = 50,
        policy: AccessPolicy | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Rank current, policy-allowed revisions by weighted field MaxSim."""
        self._check_model(embedder)
        query_vector = _normalize(embedder.embed_query(query))
        if self.dimensions is not None and len(query_vector) != self.dimensions:
            raise ValueError("query embedding dimension does not match the sidecar")
        records = passport.materialize(policy=policy)
        ranked: list[tuple[MemoryRecord, float]] = []
        for record in records:
            entry = self.entries.get(str(record.id))
            if not entry or entry.revision_hash != record.revision_hash:
                continue
            score = max(
                _FIELD_WEIGHTS.get(name, 1.0) * _dot(query_vector, vector)
                for name, vector in entry.vectors.items()
            )
            ranked.append((record, score))
        ranked.sort(key=lambda pair: (-pair[1], -pair[0].confidence, pair[0].id or ""))
        return ranked[: max(0, limit)]

    def search_hybrid(
        self,
        passport: MemoryPassport,
        query: str,
        embedder: EmbeddingBackend,
        *,
        limit: int = 12,
        candidate_limit: int = 50,
        lexical_weight: float = 1.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 60,
        policy: AccessPolicy | None = None,
    ) -> list[MemoryRecord]:
        """Fuse BM25F and semantic ranks with weighted reciprocal rank fusion."""
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        lexical = passport.search(query, limit=candidate_limit, policy=policy)
        semantic = [
            record
            for record, _score in self.dense_search(
                passport, query, embedder, limit=candidate_limit, policy=policy
            )
        ]
        records = {record.id: record for record in [*lexical, *semantic] if record.id}
        scores = {record_id: 0.0 for record_id in records}
        for rank, record in enumerate(lexical, start=1):
            if record.id in scores:
                scores[record.id] += lexical_weight / (rrf_k + rank)
        for rank, record in enumerate(semantic, start=1):
            if record.id in scores:
                scores[record.id] += semantic_weight / (rrf_k + rank)
        ordered = sorted(
            records.values(),
            key=lambda record: (-scores[str(record.id)], -record.confidence, record.id or ""),
        )
        return ordered[: max(0, limit)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "dimensions": self.dimensions,
            "entries": {
                record_id: entry.to_dict()
                for record_id, entry in sorted(self.entries.items())
            },
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, target)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> SemanticSidecar:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != SIDECAR_VERSION:
            raise ValueError(f"unsupported semantic sidecar version: {data.get('version')!r}")
        entries = {
            record_id: SemanticEntry(
                record_id=str(raw["record_id"]),
                revision_hash=str(raw["revision_hash"]),
                vectors={
                    str(name): [float(value) for value in vector]
                    for name, vector in raw.get("vectors", {}).items()
                },
            )
            for record_id, raw in data.get("entries", {}).items()
        }
        return cls(
            version=SIDECAR_VERSION,
            model_id=str(data["model_id"]),
            dimensions=int(data["dimensions"]) if data.get("dimensions") is not None else None,
            entries=entries,
        )
