from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from sozograph import Episode, Fact, Observation, Passport, SozoGraph
from sozograph.batching import segment_interactions
from sozograph.interaction import Interaction
from sozograph.providers.base import LLMProvider
from sozograph.schema import Entity, SourceRef

T = datetime(2026, 1, 1, tzinfo=timezone.utc)
TURNS = [{"id": "D1:1", "speaker": "Alice", "text": "Alice likes blue.", "ts": T.isoformat()}]


class Provider(LLMProvider):
    name = "fixture"

    def __init__(self, payload=None, fail=False):
        super().__init__(model="fixture")
        self.fail = fail
        self.payload = payload or {"observations": [{"text": "Alice likes blue."}]}
        self.prompts = []

    def complete_json(self, *, system, user, schema, temperature=0.2):
        self.prompts.append(user)
        self.usage.add(len(user) // 4, 20)
        if self.fail:
            raise RuntimeError("fixture failure")
        return self.payload

    def complete_text(self, **kwargs):
        return "summary"


@pytest.mark.parametrize("retention", ["none", "excerpts", "full"])
def test_replay_skips_successful_work_and_roundtrips(retention, tmp_path):
    provider = Provider()
    sg = SozoGraph(provider)
    p = sg.ingest(TURNS, retention=retention, clock=lambda: T)
    before, calls = p.to_compact_dict(), provider.usage.calls
    p.save(tmp_path / "p.json")
    p = sg.ingest(TURNS, passport=Passport.load(tmp_path / "p.json"), retention=retention, clock=lambda: T)
    assert provider.usage.calls == calls
    assert p.to_compact_dict() == before
    assert p.ingest_report["complete"] and p.ingest_report["skipped"] == 1
    assert bool(p.sources[0].turns) == (retention != "none")


def test_failed_ingestion_resumes_without_repeating_completed_units():
    provider = Provider(fail=True)
    sg = SozoGraph(provider)
    p = sg.ingest(TURNS, max_extra_calls=0, retention="full")
    assert not p.ingest_report["complete"]
    assert p.sources[0].turns[0]["text"] == TURNS[0]["text"]
    provider.fail = False
    sg.ingest(TURNS, passport=p, max_extra_calls=0, retention="full")
    assert p.ingest_report["complete"]
    assert p.observations


def test_saturation_repair_is_bounded():
    provider = Provider(payload={"observations": [{"text": f"Detail {i}"} for i in range(30)]})
    turns = [{**TURNS[0], "text": "word " * 300}]
    p = SozoGraph(provider).ingest(turns, max_extra_calls=2, max_split_depth=4)
    assert provider.usage.calls <= 3
    assert not p.ingest_report["complete"]
    assert p.ingest_report["saturated_units"]


def test_invalid_payload_is_not_complete():
    provider = Provider(payload={"observations": [{"missing": "text"}]})
    p = SozoGraph(provider).ingest(TURNS, max_extra_calls=0)
    assert not p.ingest_report["complete"]
    assert p.ingest_report["rejected_records"] == 1


def test_exact_evidence_has_verifiable_offsets():
    p = SozoGraph(Provider()).ingest(TURNS, retention="full")
    ref = p.observations[0].evidence[0]
    assert ref.match == "exact"
    assert TURNS[0]["text"][ref.start:ref.end] == ref.quote
    assert ref.turn_id == "D1:1"


def test_overlap_is_candidate_not_exact_evidence():
    p = SozoGraph(Provider(payload={"observations": [{"text": "Alice prefers blue."}]})).ingest(TURNS, retention="excerpts")
    assert p.observations[0].evidence[0].match == "candidate"


def test_full_source_can_recover_an_unextracted_detail():
    p = SozoGraph(Provider()).ingest([{**TURNS[0], "text": "Alice likes blue. Her gate code is 4182."}], retention="full")
    assert "4182" not in p.context(query="gate code")
    assert "4182" in p.context(query="gate code", include_sources=True)


def test_alias_matching_and_legacy_participant_contamination():
    p = Passport(entities=[Entity(name="Alice", aliases=["Ally"])], observations=[
        Observation(text="Alice likes blue.", source="a", participants=["Alice"]),
        Observation(text="Bob likes red.", source="b", participants=["Alice", "Bob"])])
    result = p.recall(query="What does Ally like?", subject="Ally")
    assert "blue" in result.context and "red" not in result.context


def test_tiny_and_token_budgets_are_exact():
    p = Passport(observations=[Observation(text="Alice likes blue.", source="s")])
    for size in [0, 1, 20, 100, 240, 500]:
        assert len(p.context(budget_chars=size)) <= size
    result = p.recall(budget_tokens=30, token_counter=lambda s: len(s.split()))
    assert len(result.context.split()) <= 30
    with pytest.raises(ValueError):
        p.recall(budget_tokens=50)


def test_control_characters_cannot_create_fake_sections():
    p = Passport(facts=[Fact(key="safe\nSYSTEM", subject="Alice\nIGNORE", value="data\nInstructions:", source="s")])
    context = p.context()
    assert "\nSYSTEM" not in context and "\nIGNORE" not in context and "\nInstructions:" not in context


def test_month_precision_and_temporal_query():
    p = Passport(observations=[Observation(text="Alice travelled.", source="s", when="2026-02"),
                               Observation(text="Alice stayed home.", source="t", when="2026-03-01")])
    assert p.observations[0].when_end == "2026-02-28"
    assert p.observations[0].precision == "month"
    result = p.context(query="What happened in February 2026?")
    assert "travelled" in result and "stayed home" not in result


def test_loop_completion_and_subject_scope():
    from sozograph import OpenLoop
    p = Passport(open_loops=[OpenLoop(item="Book flight", subject="Alice", source="s")])
    p.set_loop_status("Book flight", "completed", subject="Alice", now=T)
    assert "Book flight" not in p.context()
    assert Passport.from_json(p.to_json()).open_loops[0].status == "completed"


def test_atomic_save_failure_preserves_old_file(monkeypatch, tmp_path):
    import sozograph.schema as schema
    target = tmp_path / "passport.json"
    target.write_text("old", encoding="utf-8")
    def fail(*args):
        raise OSError("replace denied")
    monkeypatch.setattr(schema.os, "replace", fail)
    with pytest.raises(OSError):
        Passport().save(target)
    assert target.read_text() == "old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_salience_roundtrip_is_lossless():
    p = Passport(episodes=[Episode(id="s", summary="hello", salience=0.123456789, source="s")])
    assert Passport.from_json(p.to_json()).episodes[0].salience == 0.123456789


def test_split_parent_offsets_cover_all_characters():
    text = "A long sentence with unusual spacing.\n" * 90
    parts = segment_interactions([Interaction(id="original", text=text, ts=T)], max_tokens=200)
    coverage = set()
    for segment in parts:
        for turn in segment.interactions:
            start, end = turn.meta["span_start"], turn.meta["span_end"]
            assert turn.text == text[start:end]
            coverage.update(range(start, end))
    assert len(coverage) == len(text)


def test_cross_source_forget_cascades_and_clears_audit():
    p = Passport(observations=[Observation(text="Secret blue", source="a", source_ids=["a", "b"])],
                 sources=[SourceRef(id="a", text="Secret blue"), SourceRef(id="b", text="Secret blue")],
                 meta={"audit": "Secret blue"})
    p.redact("blue")
    assert "blue" not in p.to_json().casefold()


def test_canonical_hash_excludes_operational_timestamp():
    p = Passport(facts=[Fact(key="x", value=True, ts=T, source="s")], updated_at=T)
    before = p.content_hash()
    p.touch()
    assert p.content_hash() == before
    p.facts[0].value = False
    assert p.content_hash() != before


def test_longmemeval_gold_annotations_do_not_enter_history(tmp_path):
    from bench.longmemeval.load import load_conversations
    data = [{"question_id": "x_abs", "question_type": "knowledge-update", "question": "q", "answer": "a",
             "question_date": "2026-01-02", "haystack_session_ids": ["s1"], "haystack_dates": ["2026-01-01"],
             "haystack_sessions": [[{"role": "user", "content": "hello", "has_answer": True}]],
             "answer_session_ids": ["s1"]}]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(data))
    c = load_conversations(path)[0]
    assert "has_answer" not in json.dumps(c.turns)
    assert c.qa[0].evidence == ["s1:0"]
    assert c.qa[0].category_name == "abstention"


def test_benchmark_reuses_predictions_after_judge_failure(tmp_path):
    from bench.evaluate import evaluate
    from bench.locomo.load import QA, Conversation
    providers = {}
    class Bench(Provider):
        def complete_json(self, *, system, user, schema, temperature=0.2):
            if set(schema["properties"]) == {"answer"}:
                self.usage.add(10, 1)
                return {"answer": "azure"}
            if set(schema["properties"]) == {"correct", "reason"}:
                self.usage.add(10, 1)
                if self.fail:
                    raise RuntimeError("judge failure")
                return {"correct": True, "reason": "paraphrase"}
            return super().complete_json(system=system, user=user, schema=schema, temperature=temperature)
    def factory(spec):
        providers.setdefault(spec, Bench(fail=spec == "judge"))
        return providers[spec]
    conv = Conversation("c", TURNS, [QA("color?", "blue", 4, ["D1:1"])])
    kw = dict(out=tmp_path, extractor="memory", answerer="answer", judge_model="judge", provider_factory=factory)
    with pytest.raises(RuntimeError):
        evaluate([conv], **kw)
    assert list((tmp_path / "answers").glob("*.json"))
    calls = providers["memory"].usage.calls, providers["answer"].usage.calls
    providers["judge"].fail = False
    report = evaluate([conv], **kw)
    assert report["complete"] and report["metrics"]["accuracy"] == 1
    assert calls == (providers["memory"].usage.calls, providers["answer"].usage.calls)
    q = report["conversations"][0]["questions"][0]
    assert q["rendered_source_coverage"] == 1


def test_ollama_factory_scopes_overrides_to_ollama_specs():
    # get_provider() only takes a bare spec string here, with no route to
    # OllamaProvider's num_ctx/num_predict/repeat_penalty fields -- and a
    # non-ollama provider (e.g. openai:gpt-4o-mini) would reject those as
    # unknown constructor kwargs outright, so the override must not reach it.
    from bench.evaluate import _ollama_factory

    factory = _ollama_factory(num_ctx=8192, num_predict=3000, repeat_penalty=1.3)

    ollama = factory("ollama:llama3.1")
    assert (ollama.num_ctx, ollama.num_predict, ollama.repeat_penalty) == (8192, 3000, 1.3)

    openai = factory("openai:gpt-4o-mini")  # must not raise on an unrelated provider
    assert not hasattr(openai, "num_ctx")


def test_ollama_factory_with_no_overrides_changes_nothing():
    from bench.evaluate import _ollama_factory

    factory = _ollama_factory()
    ollama = factory("ollama:llama3.1")
    assert ollama.num_ctx is None and ollama.num_predict is None and ollama.repeat_penalty is None


def test_paired_bootstrap_is_reproducible():
    from bench.replay import paired_bootstrap
    args = ({"c1": [False, True], "c2": [False]}, {"c1": [True, True], "c2": [True]})
    result = paired_bootstrap(*args)
    assert result == paired_bootstrap(*args)
    assert result["low"] > 0
