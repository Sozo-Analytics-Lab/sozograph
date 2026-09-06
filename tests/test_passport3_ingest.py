from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sozograph import MemoryPassport, SozoGraph
from sozograph.providers.base import LLMProvider

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 10, tzinfo=UTC)


def _empty_payload() -> dict:
    return {
        "facts": [],
        "prefs": [],
        "entities": [],
        "open_loops": [],
        "observations": [],
        "episode": {
            "summary": "",
            "participants": [],
            "keywords": [],
            "salience": 0.0,
        },
    }


class Passport3Provider(LLMProvider):
    name = "passport3-test"

    def __init__(self) -> None:
        super().__init__(model="passport3-test")
        self.prompts: list[str] = []

    def complete_json(self, *, system, user, schema, temperature=0.2):
        self.prompts.append(user)
        payload = _empty_payload()
        if "I live in Harare" in user:
            payload["facts"] = [{
                "key": "city",
                "value": "Harare",
                "confidence": 0.97,
                "evidence_quote": "live in Harare",
            }]
            payload["entities"] = [{
                "name": "Harare",
                "type": "place",
                "aliases": [],
                "evidence_quote": "Harare",
            }]
            payload["episode"] = {
                "summary": "The user said they live in Harare.",
                "participants": ["user"],
                "keywords": ["Harare", "city"],
                "salience": 0.7,
            }
        if "I moved to Kwekwe" in user:
            payload["facts"] = [{
                "key": "city",
                "value": "Kwekwe",
                "confidence": 0.99,
                "evidence_quote": "moved to Kwekwe",
            }]
            payload["episode"] = {
                "summary": "The user moved to Kwekwe.",
                "participants": ["user"],
                "keywords": ["Kwekwe", "move"],
                "salience": 0.8,
            }
        if "painting above the stove" in user:
            payload["observations"] = [{
                "text": "Melanie hung the painting above the stove.",
                "when": "",
                "evidence_quote": "painting above the stove",
            }]
            payload["open_loops"] = [{
                "item": "Ask who painted it.",
                "evidence_quote": "Ask who painted it",
            }]
        return payload

    def complete_text(self, *, system, user, temperature=0.2):
        return "summary"


def _fact(passport: MemoryPassport, *, valid_at: datetime | None = None):
    return next(
        record
        for record in passport.materialize(valid_at=valid_at)
        if record.kind == "fact" and record.key == "city"
    )


def test_direct_v3_ingest_emits_governed_events_with_exact_evidence():
    graph = SozoGraph(provider=Passport3Provider())
    source = "I live in Harare."
    passport = graph.ingest_v3(
        {"text": source, "ts": T0.isoformat()},
        meta={"user_key": "u1"},
        batch=False,
        transaction_time=T0,
        authority="user",
        sensitivity="confidential",
        scopes=["profile"],
    )

    fact = _fact(passport)
    assert fact.value == "Harare"
    assert fact.authority == "user"
    assert fact.sensitivity == "confidential"
    assert fact.scopes == ["profile"]
    assert fact.evidence[0].exact is True
    assert fact.evidence[0].verify(source)
    assert passport.extensions["sozograph:last_ingest"] == {
        "events_appended": 2,
        "revisions_skipped": 0,
        "exact_evidence": 2,
        "coarse_evidence": 0,
        "transaction_time": T0.isoformat(),
        "evidence_linking": "deterministic",
        "evidence_deterministic": 2,
        "evidence_model_linked": 0,
        "evidence_unresolved": 0,
    }


def test_model_evidence_linker_only_handles_unresolved_candidates():
    class LinkProvider(Passport3Provider):
        def complete_json(self, *, system, user, schema, temperature=0.2):
            if "links" in schema.get("properties", {}):
                return {
                    "links": [{
                        "candidate_id": "observations:0",
                        "quote": "Oliver hid his bone in Melanie's slipper",
                    }]
                }
            payload = _empty_payload()
            payload["facts"] = [{"key": "pet", "value": "Oliver", "confidence": 0.9}]
            payload["observations"] = [{
                "text": "The canine concealed its chew toy in the footwear.",
                "when": "",
            }]
            return payload

    source = "Oliver hid his bone in Melanie's slipper."
    passport = SozoGraph(provider=LinkProvider()).ingest_v3(
        {"text": source, "ts": T0.isoformat()},
        batch=False,
        transaction_time=T0,
        evidence_linking="model",
    )
    observation = next(
        record for record in passport.materialize() if record.kind == "observation"
    )
    assert observation.evidence[0].quote == "Oliver hid his bone in Melanie's slipper"
    assert observation.evidence[0].verify(source)
    stats = passport.extensions["sozograph:last_ingest"]
    assert stats["evidence_deterministic"] == 1
    assert stats["evidence_model_linked"] == 1
    assert stats["evidence_unresolved"] == 0


def test_direct_v3_replay_is_idempotent_even_with_a_new_transaction_time():
    graph = SozoGraph(provider=Passport3Provider())
    source = {"text": "I live in Harare.", "ts": T0.isoformat()}
    passport = graph.ingest_v3(
        source,
        meta={"user_key": "u1"},
        batch=False,
        transaction_time=T0,
    )
    event_ids = [event.event_id for event in passport.events]

    replayed = graph.ingest_v3(
        source,
        passport=passport,
        batch=False,
        transaction_time=T0 + timedelta(days=4),
    )

    assert [event.event_id for event in replayed.events] == event_ids
    assert replayed.extensions["sozograph:last_ingest"]["events_appended"] == 0
    assert replayed.extensions["sozograph:last_ingest"]["revisions_skipped"] == 2


def test_direct_v3_keeps_current_and_valid_time_views_of_a_revision():
    graph = SozoGraph(provider=Passport3Provider())
    passport = graph.ingest_v3(
        {"text": "I live in Harare.", "ts": T0.isoformat()},
        meta={"user_key": "u1"},
        batch=False,
        transaction_time=T0,
    )
    later = T0 + timedelta(days=10)
    graph.ingest_v3(
        {"text": "I moved to Kwekwe.", "ts": later.isoformat()},
        passport=passport,
        batch=False,
        transaction_time=later,
    )

    current = _fact(passport)
    historical = _fact(passport, valid_at=T0 + timedelta(hours=1))
    assert current.value == "Kwekwe"
    assert historical.value == "Harare"
    assert current.supersedes


def test_tombstone_prevents_reingestion_from_resurrecting_a_fact():
    graph = SozoGraph(provider=Passport3Provider())
    source = {"text": "I live in Harare.", "ts": T0.isoformat()}
    passport = graph.ingest_v3(
        source,
        meta={"user_key": "u1"},
        batch=False,
        transaction_time=T0,
    )
    city_id = _fact(passport).id
    passport.delete(city_id, occurred_at=T0 + timedelta(days=1))
    count_after_delete = len(passport.events)

    graph.ingest_v3(
        source,
        passport=passport,
        batch=False,
        transaction_time=T0 + timedelta(days=2),
    )

    assert len(passport.events) == count_after_delete
    assert not any(record.id == city_id for record in passport.materialize())


def test_exact_quote_fallback_and_coarse_evidence_are_auditable():
    class BadQuoteProvider(Passport3Provider):
        def complete_json(self, *, system, user, schema, temperature=0.2):
            payload = _empty_payload()
            payload["facts"] = [{
                "key": "city",
                "value": "Harare",
                "confidence": 0.9,
                "evidence_quote": "not in source",
            }]
            payload["observations"] = [{
                "text": "The user lives in Zimbabwe.",
                "when": "",
                "evidence_quote": "also absent",
            }]
            return payload

    passport = SozoGraph(provider=BadQuoteProvider()).ingest_v3(
        {"text": "Harare is home.", "ts": T0.isoformat()},
        batch=False,
        transaction_time=T0,
    )
    records = passport.materialize()
    fact = next(record for record in records if record.kind == "fact")
    observation = next(record for record in records if record.kind == "observation")

    assert fact.evidence[0].quote == "Harare"
    assert fact.evidence[0].exact is True
    assert observation.evidence[0].exact is False
    assert observation.evidence[0].source_sha256
    assert passport.extensions["sozograph:last_ingest"]["coarse_evidence"] == 1


def test_direct_ingest_replica_merge_converges_and_unions_sources():
    graph = SozoGraph(provider=Passport3Provider())
    left = graph.ingest_v3(
        {"text": "I live in Harare.", "ts": T0.isoformat()},
        meta={"user_key": "u1"},
        batch=False,
        replica_id="left",
        transaction_time=T0,
    )
    later = T0 + timedelta(days=10)
    right = graph.ingest_v3(
        {"text": "I moved to Kwekwe.", "ts": later.isoformat()},
        meta={"user_key": "u1"},
        batch=False,
        replica_id="right",
        transaction_time=later,
    )

    merged_lr = left.merge(right)
    merged_rl = right.merge(left)
    assert merged_lr.state_hash() == merged_rl.state_hash()
    assert _fact(merged_lr).value == "Kwekwe"
    assert len(merged_lr.extensions["sozograph:sources"]) == 2


def test_direct_ingest_records_one_failed_source_and_continues():
    class FailingProvider(Passport3Provider):
        def complete_json(self, *, system, user, schema, temperature=0.2):
            raise RuntimeError("simulated extraction failure")

    passport = SozoGraph(provider=FailingProvider()).ingest_v3(
        {"text": "I live in Harare.", "ts": T0.isoformat()},
        batch=False,
        transaction_time=T0,
    )

    assert passport.events == []
    failure = passport.extensions["sozograph:ingest_failures"][0]
    assert failure["source_time"] == T0.isoformat()
    assert "simulated extraction failure" in failure["error"]
