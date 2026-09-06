from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sozograph import AccessPolicy, MemoryPassport, MemoryRecord, SemanticSidecar

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class MeaningEmbedder:
    model_id = "meaning-test-v1"

    @staticmethod
    def _vector(text: str) -> list[float]:
        value = text.casefold()
        return [
            float(any(word in value for word in ("doctor", "physician", "medical"))),
            float(any(word in value for word in ("harare", "city", "home"))),
            float(any(word in value for word in ("music", "guitar", "instrument"))),
            0.1,
        ]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]


def _passport() -> MemoryPassport:
    passport = MemoryPassport.new(subject_key="u1", replica_id="a", now=T0)
    passport.append(
        MemoryRecord(kind="fact", key="occupation", value="doctor", recorded_at=T0),
        occurred_at=T0,
    )
    passport.append(
        MemoryRecord(kind="fact", key="location", value="Harare", recorded_at=T0),
        occurred_at=T0,
    )
    passport.append(
        MemoryRecord(kind="observation", text="Mara practices guitar.", recorded_at=T0),
        occurred_at=T0,
    )
    return passport


def test_dense_sidecar_recovers_paraphrase_and_round_trips(tmp_path: Path):
    passport = _passport()
    embedder = MeaningEmbedder()
    sidecar = SemanticSidecar.build(passport, embedder)

    hit, score = sidecar.dense_search(passport, "Which physician?", embedder, limit=1)[0]
    assert hit.key == "occupation"
    assert score > 0.9

    path = sidecar.save(tmp_path / "memory.vectors.json")
    loaded = SemanticSidecar.load(path)
    assert loaded.to_dict() == sidecar.to_dict()


def test_sync_reembeds_changed_revisions_and_removes_old_vectors():
    passport = _passport()
    embedder = MeaningEmbedder()
    sidecar = SemanticSidecar.build(passport, embedder)
    occupation = next(record for record in passport.materialize() if record.key == "occupation")
    prior_hash = sidecar.entries[str(occupation.id)].revision_hash

    passport.upsert(
        MemoryRecord(kind="fact", key="occupation", value="musician", recorded_at=T0),
        occurred_at=T0,
    )
    stats = sidecar.sync(passport, embedder)
    current = next(record for record in passport.materialize() if record.key == "occupation")

    assert stats.removed_records == 1
    assert stats.embedded_records == 1
    assert sidecar.entries[str(current.id)].revision_hash != prior_hash
    assert sidecar.dense_search(passport, "instrument", embedder, limit=1)[0][0].key == "occupation"


def test_hybrid_search_applies_policy_before_ranking():
    passport = _passport()
    secret = MemoryRecord(
        kind="fact",
        key="medical_note",
        value="doctor",
        sensitivity="restricted",
        scopes=["private"],
        recorded_at=T0,
    )
    passport.append(secret, occurred_at=T0)
    embedder = MeaningEmbedder()
    sidecar = SemanticSidecar.build(passport, embedder)
    policy = AccessPolicy(max_sensitivity="internal", scopes=[])

    results = sidecar.search_hybrid(passport, "physician", embedder, policy=policy)
    assert any(record.key == "occupation" for record in results)
    assert all(record.key != "medical_note" for record in results)


def test_sidecar_rejects_a_different_embedding_model():
    passport = _passport()
    sidecar = SemanticSidecar.build(passport, MeaningEmbedder())

    class OtherEmbedder(MeaningEmbedder):
        model_id = "other"

    try:
        sidecar.dense_search(passport, "doctor", OtherEmbedder())
    except ValueError as exc:
        assert "sidecar model" in str(exc)
    else:
        raise AssertionError("model mismatch should fail")
