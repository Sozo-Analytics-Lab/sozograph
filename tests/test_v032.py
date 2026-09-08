"""Public-contract regressions for the portability-first release."""
from datetime import datetime, timezone

import pytest

from sozograph import Fact, Observation, Passport
from sozograph.batching import segment_interactions
from sozograph.extractor import Extractor
from sozograph.interaction import Interaction
from sozograph.resolver import merge_passport_update
from sozograph.retrieve import tokenize

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("texts,dates", [
    (["Alice visited the museum."] * 2, ["2026-01-01", "2026-02-01"]),
    (["Alice will attend the annual community planning meeting next Monday.",
      "Alice will not attend the annual community planning meeting next Monday."], ["2026-01-01"] * 2),
    (["Alice gave Bob the red book.", "Bob gave Alice the red book."], ["2026-01-01"] * 2),
    (["Alice bought 20 blue pens for the office meeting.",
      "Alice bought 21 blue pens for the office meeting."], ["2026-01-01"] * 2),
])
def test_distinct_observations_survive(texts, dates):
    p = Passport(updated_at=T)
    for text, when in zip(texts, dates, strict=True):
        merge_passport_update(p, observations=[Observation(text=text, when=when, ts=T,
                                                         participants=["Alice"], source=when)])
    assert len(p.observations) == 2


def test_subject_is_not_every_speaker():
    out = Extractor(None).validate({"observations": [{"text": "Caroline adopted a dog."}]},
                                  source_id="s", ts=T, participants=["Melanie", "Caroline"])
    assert out["observations"][0].participants == ["Caroline"]


def test_unknown_fields_preserve_positions():
    raw = {"version": "2.2", "updated_at": T.isoformat(), "future": {"x": 1},
           "facts": [{"key": "a", "value": 1, "source": "s", "ts": T.isoformat(),
                      "future": {"y": 2}}]}
    p = Passport.from_dict(raw)
    out = Passport.from_json(p.to_json()).to_compact_dict()
    assert out["future"] == raw["future"]
    assert out["facts"][0]["future"] == {"y": 2}


def test_future_version_roundtrip_and_write_gate():
    raw = {"version": "9.0", "facts": {"new_shape": True}, "unknown": [1, 2]}
    p = Passport.from_dict(raw)
    assert p.to_compact_dict() == raw
    with pytest.raises(ValueError, match="version"):
        merge_passport_update(p, facts=[Fact(key="x", value=1, source="s")])


def test_identity_distinguishes_speakers():
    def unit(speaker):
        return segment_interactions([Interaction(text="I live in Paris", ts=T,
                                                meta={"speaker": speaker})])[0]
    assert unit("Alice").id != unit("Bob").id


def test_oversized_turn_is_fully_covered():
    body = "First sentence. " * 1000 + "THE FINAL ANSWER"
    segments = segment_interactions([Interaction(text=body, ts=T)], max_tokens=200)
    assert len(segments) > 1
    assert "THE FINAL ANSWER" in segments[-1].text()
    assert all(len(s.text()) <= 720 for s in segments)


def test_unicode_retrieval():
    assert tokenize("北京 東京 مرحبا")
    assert tokenize("café") == tokenize("cafe\u0301")


def test_equal_time_conflict_is_visible():
    p = Passport(updated_at=T)
    for value in ["Paris", "Rome"]:
        merge_passport_update(p, facts=[Fact(key="location", value=value, source=value, ts=T)])
    assert {f.value for f in p.facts} == {"Paris", "Rome"}
    assert all(f.status == "disputed" for f in p.facts)


def test_subject_scopes_do_not_overwrite():
    p = Passport(updated_at=T)
    for subject, value in [("Alice", "Paris"), ("Bob", "Rome")]:
        merge_passport_update(p, facts=[Fact(key="location", value=value, subject=subject,
                                           source=subject, ts=T)])
    assert len(p.facts) == 2


def test_recall_packs_complete_records():
    p = Passport(updated_at=T, observations=[
        Observation(text="The answer is blue.", source="s", ts=T),
        Observation(text="An unrelated " + "huge " * 200, source="x", ts=T)])
    result = p.recall(query="answer blue", budget_chars=240)
    assert len(result.context) <= 240
    assert "The answer is blue." in result.context
    assert "…" not in result.context
    assert result.selected and result.dropped


def test_forget_removes_evidence_and_recall():
    from sozograph.schema import SourceRef
    p = Passport(updated_at=T, observations=[Observation(text="Secret kiwi", source="s", ts=T)],
                 sources=[SourceRef(id="s", ts=T, text="Secret kiwi")])
    p.forget(source_ids=["s"])
    assert "kiwi" not in p.to_json().lower()
    assert "kiwi" not in p.context().lower()
