"""
The benchmark harness, against a LoCoMo-shaped fixture and a fake provider.

The harness produces the number that decides whether the headline claim is
true, so it needs to be correct itself. These tests run the whole path with no
dataset, no API key, and no network.
"""
from __future__ import annotations

import json

import pytest

from bench.locomo import runners as runners_mod
from bench.locomo.judge import judge
from bench.locomo.load import CATEGORY_NAMES, describe, load_conversations
from bench.locomo.metrics import aggregate, render_table, save
from bench.locomo.runners import run_full_context, run_sozograph
from sozograph.providers.base import LLMProvider

SAMPLE = [{
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Melanie",
        "speaker_b": "Caroline",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Melanie", "dia_id": "D1:1",
             "text": "I finished renovating my kitchen in sage green."},
            {"speaker": "Caroline", "dia_id": "D1:2", "text": "That sounds lovely."},
            {"speaker": "Melanie", "dia_id": "D1:3",
             "text": "Here is the result.", "blip_caption": "a green kitchen with a painting"},
        ],
        "session_2_date_time": "9:00 am on 11 June, 2023",
        "session_2": [
            {"speaker": "Melanie", "dia_id": "D2:1", "text": "I moved to Kwekwe."},
        ],
    },
    "qa": [
        {"question": "What colour is the kitchen?", "answer": "sage green",
         "category": 4, "evidence": ["D1:1"]},
        {"question": "Where does Melanie live now?", "answer": "Kwekwe",
         "category": 2, "evidence": ["D2:1"]},
        {"question": "Did Melanie buy a boat?", "answer": "Not mentioned", "category": 5},
    ],
}]


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return path


class BenchProvider(LLMProvider):
    name = "bench-fake"

    def __init__(self, model="fake"):
        super().__init__(model=model)

    def complete_json(self, *, system, user, schema, temperature=0.2):
        self.usage.add(max(1, len(user) // 4), 40)
        props = set(schema.get("properties", {}))

        if props == {"correct", "reason"}:
            gold = user.split("CORRECT ANSWER:")[1].split("CANDIDATE ANSWER:")[0].strip()
            candidate = user.split("CANDIDATE ANSWER:")[1].split("Does the")[0].strip()
            return {"correct": gold.lower() in candidate.lower(), "reason": "fixture"}

        if props == {"answer"}:
            low = user.lower()
            if "colour is the kitchen" in low:
                return {"answer": "sage green" if "sage green" in low else "unknown"}
            if "live now" in low:
                return {"answer": "Kwekwe" if "kwekwe" in low else "unknown"}
            return {"answer": "Not mentioned"}

        low = user.lower()
        if "sage green" in low:
            facts = [{"key": "kitchen_colour", "value": "sage green", "confidence": 0.9}]
        elif "kwekwe" in low:
            facts = [{"key": "location", "value": "Kwekwe", "confidence": 0.9}]
        else:
            facts = []
        return {
            "facts": facts,
            "prefs": [],
            "entities": [],
            "open_loops": [],
            "episode": {"summary": user.strip()[:300] or "conversation",
                        "participants": [], "keywords": [], "salience": 0.5},
        }

    def complete_text(self, *, system, user, temperature=0.2):
        self.usage.add(10, 5)
        return "summary"


@pytest.fixture
def fake_providers(monkeypatch):
    made = []

    def factory(spec, **kwargs):
        provider = BenchProvider(model=str(spec))
        made.append(provider)
        return provider

    monkeypatch.setattr(runners_mod, "get_provider", factory)
    return made


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_loads_the_locomo_shape(data_file):
    convs = load_conversations(data_file)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.sample_id == "conv-1"
    assert len(conv.turns) == 4
    assert conv.sessions == 2
    assert conv.speakers == ["Melanie", "Caroline"]


def test_adversarial_category_is_excluded_by_default(data_file):
    conv = load_conversations(data_file)[0]
    assert {q.category for q in conv.qa} == {4, 2}


def test_adversarial_can_be_included(data_file):
    conv = load_conversations(data_file, categories=(1, 2, 3, 4, 5))[0]
    assert 5 in {q.category for q in conv.qa}


def test_image_captions_are_folded_into_the_turn(data_file):
    conv = load_conversations(data_file)[0]
    text = " ".join(t["text"] for t in conv.turns)
    # Images are not released, so the caption is the only signal; dropping it
    # loses every question that depends on what was shown.
    assert "a green kitchen with a painting" in text


def test_session_dates_are_parsed_and_ordered(data_file):
    conv = load_conversations(data_file)[0]
    stamps = [t["ts"] for t in conv.turns]
    assert stamps == sorted(stamps)
    assert stamps[0].startswith("2023-05-08")
    assert stamps[-1].startswith("2023-06-11")


def test_missing_dataset_explains_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_conversations(tmp_path / "nope.json")
    assert "snap-research/locomo" in str(exc.value)


def _multi_conversation_file(tmp_path):
    samples = []
    for i in range(3):
        sample = json.loads(json.dumps(SAMPLE[0]))  # deep copy
        sample["sample_id"] = f"conv-{i}"
        samples.append(sample)
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(samples), encoding="utf-8")
    return path


def test_offset_skips_leading_conversations_for_daily_quota_batches(tmp_path):
    path = _multi_conversation_file(tmp_path)
    all_convs = load_conversations(path)
    assert [c.sample_id for c in all_convs] == ["conv-0", "conv-1", "conv-2"]

    batch = load_conversations(path, offset=1, limit=1)
    assert [c.sample_id for c in batch] == ["conv-1"]

    tail = load_conversations(path, offset=2)
    assert [c.sample_id for c in tail] == ["conv-2"]


def test_partial_results_survive_a_mid_run_crash(tmp_path, monkeypatch, fake_providers):
    # A quota-capped provider can die after several real, already-billed calls
    # succeeded. Losing that work on every crash defeats the point of running
    # in small daily batches, so whatever finished before the crash must land
    # on disk, not just the exception.
    from bench.locomo import run as run_mod

    data_path = _multi_conversation_file(tmp_path)
    out_dir = tmp_path / "results"

    real_judge = run_mod.judge
    calls = {"n": 0}

    def flaky_judge(provider, **kwargs):
        calls["n"] += 1
        # conv-0 has 3 questions, all exact-match (no provider call needed);
        # fail on conv-1's first question so conv-0 is the only completed one.
        if calls["n"] > 3:
            raise RuntimeError("simulated quota exhaustion")
        return real_judge(provider, **kwargs)

    monkeypatch.setattr(run_mod, "judge", flaky_judge)
    monkeypatch.setattr(run_mod, "get_provider", lambda spec, **kw: BenchProvider(model=str(spec)))

    with pytest.raises(RuntimeError, match="simulated quota exhaustion"):
        run_mod.main([
            "--data", str(data_path),
            "--provider", "bench:fake",
            "--judge", "bench:fake",
            "--out", str(out_dir),
        ])

    files = list(out_dir.glob("locomo_*.json"))
    assert len(files) == 1, "the crash must still leave exactly one results file"
    payload = json.loads(files[0].read_text(encoding="utf-8"))

    assert payload["config"]["note"].startswith("partial")
    assert payload["metrics"][0]["conversations"] == 1
    saved_ids = {r["sample_id"] for r in payload["per_conversation"]["sozograph"]}
    assert saved_ids == {"conv-0"}


def test_base_url_reaches_the_provider_and_judge_separately(tmp_path, monkeypatch, data_file):
    # openai-compatible gateways (Groq, Together, vLLM...) are reached by
    # pointing the existing openai provider at a base_url; --provider and
    # --judge can be different gateways, so each needs its own override.
    from bench.locomo import run as run_mod

    seen: list[tuple[str, dict]] = []

    def factory(spec, **kwargs):
        seen.append((spec, kwargs))
        return BenchProvider(model=str(spec))

    monkeypatch.setattr(runners_mod, "get_provider", factory)
    monkeypatch.setattr(run_mod, "get_provider", factory)

    run_mod.main([
        "--data", str(data_file),
        "--systems", "sozograph",
        "--provider", "openai:llama-model",
        "--base-url", "https://api.groq.com/openai/v1",
        "--judge", "openai:judge-model",
        "--judge-base-url", "https://example-judge-gateway.test/v1",
        "--out", str(tmp_path / "results"),
    ])

    provider_calls = {spec: kwargs for spec, kwargs in seen if spec == "openai:llama-model"}
    judge_calls = {spec: kwargs for spec, kwargs in seen if spec == "openai:judge-model"}
    assert provider_calls["openai:llama-model"]["base_url"] == "https://api.groq.com/openai/v1"
    assert judge_calls["openai:judge-model"]["base_url"] == "https://example-judge-gateway.test/v1"


def test_describe_summarizes_the_dataset(data_file):
    summary = describe(load_conversations(data_file))
    assert summary["conversations"] == 1
    assert summary["questions"] == 2
    assert set(summary["by_category"]) <= set(CATEGORY_NAMES.values())


# --------------------------------------------------------------------------
# Running and scoring
# --------------------------------------------------------------------------

def test_sozograph_runner_separates_memory_and_qa_costs(data_file, fake_providers):
    conv = load_conversations(data_file)[0]
    result = run_sozograph(conv, model="fake")

    assert result.memory_usage.calls > 0
    assert result.qa_usage.calls == len(conv.qa)
    assert result.memory_usage.total_tokens > 0
    # Separate provider instances, so the two columns cannot contaminate.
    assert result.memory_usage.calls + result.qa_usage.calls == result.total_calls
    assert result.passport_tokens > 0
    assert result.notes["episodes"] > 0


def test_sozograph_answers_correctly_from_the_passport(data_file, fake_providers):
    conv = load_conversations(data_file)[0]
    result = run_sozograph(conv, model="fake")
    predictions = {a.question: a.prediction for a in result.answers}
    assert predictions["What colour is the kitchen?"] == "sage green"


def test_full_context_baseline_runs(data_file, fake_providers):
    conv = load_conversations(data_file)[0]
    result = run_full_context(conv, model="fake")
    assert result.qa_usage.calls == len(conv.qa)
    assert result.memory_usage.calls == 0, "the baseline builds no memory"


def _long_sample(sessions: int = 20, turns_per_session: int = 30) -> list[dict]:
    """A conversation at the scale the token claim actually applies to."""
    conversation: dict = {"speaker_a": "Melanie", "speaker_b": "Caroline"}
    for s in range(1, sessions + 1):
        conversation[f"session_{s}_date_time"] = f"9:00 am on {s} May, 2023"
        conversation[f"session_{s}"] = [
            {
                "speaker": "Melanie" if t % 2 == 0 else "Caroline",
                "dia_id": f"D{s}:{t}",
                "text": (
                    f"In session {s} we talked at some length about topic {t}, "
                    "covering the details and the reasoning behind them, the way "
                    "people actually talk when they are catching up properly."
                ),
            }
            for t in range(turns_per_session)
        ]
    return [{
        "sample_id": "long-1",
        "conversation": conversation,
        "qa": [{"question": "What did they discuss?", "answer": "topics",
                "category": 4, "evidence": []}],
    }]


def test_sozograph_uses_fewer_qa_tokens_than_full_context_on_a_long_conversation(
    tmp_path, fake_providers
):
    """
    The token win is a property of long histories, not a universal one.

    On a four-turn conversation the passport is larger than the transcript and
    stuffing the transcript is cheaper. The crossover is what matters, and it
    arrives well before LoCoMo's ~600-turn scale.
    """
    path = tmp_path / "long.json"
    path.write_text(json.dumps(_long_sample()), encoding="utf-8")
    conv = load_conversations(path)[0]
    assert conv.estimated_tokens() > 10_000

    sozo = run_sozograph(conv, model="fake", budget_chars=6000)
    full = run_full_context(conv, model="fake")

    assert sozo.qa_usage.prompt_tokens < full.qa_usage.prompt_tokens

    # And the passport stays a small, portable object. The fixture's episode
    # summaries are deliberately long (300 chars each, where a real extractor
    # writes one to three sentences), so this is a pessimistic floor.
    compression = conv.estimated_tokens() / sozo.passport_tokens
    assert compression > 3, f"only {compression:.1f}x compression"


def test_provenance_is_recorded_per_segment_not_per_turn(tmp_path, fake_providers):
    """
    One SourceRef per turn made the evidence log larger than the memory.

    On a 600-turn conversation that was ~22k tokens of provenance nothing
    referenced, since facts cite the segment that produced them.
    """
    path = tmp_path / "long.json"
    path.write_text(json.dumps(_long_sample()), encoding="utf-8")
    conv = load_conversations(path)[0]

    result = run_sozograph(conv, model="fake")
    assert result.notes["segments"] < len(conv.turns) / 5


def test_batching_keeps_memory_calls_far_below_turn_count(tmp_path, fake_providers):
    path = tmp_path / "long.json"
    path.write_text(json.dumps(_long_sample()), encoding="utf-8")
    conv = load_conversations(path)[0]

    result = run_sozograph(conv, model="fake")
    assert result.memory_usage.calls < len(conv.turns) / 5, (
        "batching must cut calls by an order of magnitude, not a little"
    )


def test_judge_short_circuits_on_an_exact_match():
    provider = BenchProvider()
    verdict = judge(provider, question="q", gold="Kwekwe", prediction="kwekwe")
    assert verdict.correct
    assert provider.usage.calls == 0, "an exact match needs no judge call"


def test_judge_rejects_an_empty_prediction():
    provider = BenchProvider()
    assert not judge(provider, question="q", gold="Kwekwe", prediction="  ").correct
    assert provider.usage.calls == 0


def test_judge_calls_the_model_for_a_paraphrase():
    provider = BenchProvider()
    verdict = judge(provider, question="q", gold="Kwekwe",
                    prediction="She lives in Kwekwe now")
    assert verdict.correct
    assert provider.usage.calls == 1


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_metrics_aggregate_and_render(data_file, fake_providers, tmp_path):
    conv = load_conversations(data_file)[0]
    result = run_sozograph(conv, model="fake")
    marks = [True, False]

    metrics = aggregate("sozograph", [result], [marks])
    assert metrics.questions == 2
    assert metrics.correct == 1
    assert metrics.accuracy == 50.0
    assert set(metrics.category_accuracy()) == {"single_hop", "temporal"}

    table = render_table([metrics])
    assert "sozograph" in table and "LightMem" in table
    assert "not re-run here" in table

    path = save([metrics], {"sozograph": [result]}, tmp_path, config={"provider": "fake"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"][0]["accuracy_pct"] == 50.0
    assert "published_reference" in payload


def test_metrics_report_per_conversation_figures():
    from bench.locomo.metrics import Metrics

    m = Metrics(system="x", conversations=10, questions=200, correct=150,
                memory_tokens=500_000, qa_tokens=100_000,
                memory_calls=200, qa_calls=200)
    d = m.to_dict()
    assert d["accuracy_pct"] == 75.0
    assert d["memory_tokens_per_conversation"] == 50_000
    assert d["total_calls_per_conversation"] == 40.0


def test_render_table_without_published_rows():
    from bench.locomo.metrics import Metrics

    table = render_table([Metrics(system="x", conversations=1, questions=1, correct=1)],
                         include_published=False)
    assert "LightMem" not in table
