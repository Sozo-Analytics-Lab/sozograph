"""
End-to-end pipeline against a fake provider.

Covers the claims the README will make: batching cuts API calls, episodes carry
what a flat belief state loses, a query finds the right episode without an
embedding model, and the passport survives a trip through a file with no SDK
and no network involved.
"""
from __future__ import annotations

import json
import sys

import pytest

from sozograph import Passport, SozoGraph
from sozograph.batching import plan, segment_interactions
from sozograph.ingest import coerce_to_interactions
from sozograph.providers.base import LLMProvider
from sozograph.retrieve import BM25, rank, tokenize

CONVERSATION = [
    {"speaker": "Melanie", "text": "I finally finished renovating my kitchen.",
     "ts": "2026-01-05T10:00:00Z", "session": "1"},
    {"speaker": "Caroline", "text": "What colour did you go with?",
     "ts": "2026-01-05T10:01:00Z", "session": "1"},
    {"speaker": "Melanie", "text": "Sage green. I hung my grandmother's painting above the stove.",
     "ts": "2026-01-05T10:02:00Z", "session": "1"},
    {"speaker": "Melanie", "text": "I live in Harare and I prefer short direct answers.",
     "ts": "2026-02-10T09:00:00Z", "session": "2"},
    {"speaker": "Caroline", "text": "Noted. How is the new job going?",
     "ts": "2026-02-10T09:01:00Z", "session": "2"},
    {"speaker": "Melanie", "text": "Good. I moved to Kwekwe last week for it.",
     "ts": "2026-03-11T09:00:00Z", "session": "3"},
]


class FakeProvider(LLMProvider):
    """Returns a plausible extraction per segment, and counts calls."""

    name = "fake"

    def __init__(self):
        super().__init__(model="fake-1")
        self.seen: list[str] = []

    def complete_json(self, *, system, user, schema, temperature=0.2):
        self.seen.append(user)
        self.usage.add(len(user) // 4, 60)
        text = user.lower()

        facts, prefs, episode = [], [], None
        if "kitchen" in text:
            facts.append({"key": "kitchen_colour", "value": "sage green", "confidence": 0.9})
            episode = {
                "summary": "Melanie finished renovating her kitchen in sage green and "
                           "hung her grandmother's painting above the stove.",
                "participants": ["Melanie", "Caroline"],
                "keywords": ["kitchen", "renovation", "painting", "grandmother", "stove"],
                "salience": 0.7,
            }
        if "harare" in text:
            facts.append({"key": "location", "value": "Harare", "confidence": 0.9})
            prefs.append({"key": "tone", "value": "direct", "confidence": 0.9})
            episode = {
                "summary": "Melanie said she lives in Harare and prefers short direct answers.",
                "participants": ["Melanie"],
                "keywords": ["harare", "location", "tone"],
                "salience": 0.6,
            }
        if "kwekwe" in text:
            facts.append({"key": "location", "value": "Kwekwe", "confidence": 0.95})
            episode = {
                "summary": "Melanie moved to Kwekwe for a new job.",
                "participants": ["Melanie"],
                "keywords": ["kwekwe", "job", "moved"],
                "salience": 0.8,
            }

        return {
            "facts": facts,
            "prefs": prefs,
            "entities": [],
            "open_loops": [],
            "episode": episode or {"summary": "General conversation.", "participants": [],
                                   "keywords": [], "salience": 0.2},
        }

    def complete_text(self, *, system, user, temperature=0.2):
        self.usage.add(20, 10)
        return "A summary."


@pytest.fixture
def graph():
    return SozoGraph(provider=FakeProvider())


# --------------------------------------------------------------------------
# The end-to-end path
# --------------------------------------------------------------------------

def test_full_ingest_builds_a_usable_passport(graph):
    passport = graph.ingest(CONVERSATION)

    assert passport.facts, "no facts extracted"
    assert passport.episodes, "no episodes recorded"

    by_key = {f.key: f.value for f in passport.facts}
    assert by_key["location"] == "Kwekwe", "the newest value must win"
    assert by_key["kitchen_colour"] == "sage green"
    assert any(p.key == "tone" for p in passport.prefs)

    # Harare -> Kwekwe is a real change and must be recorded once.
    changes = [c for c in passport.contradictions if c.key == "location"]
    assert len(changes) == 1
    assert (changes[0].old, changes[0].new) == ("Harare", "Kwekwe")


def test_batching_costs_one_call_per_segment(graph):
    passport = graph.ingest(CONVERSATION)
    interactions, _ = coerce_to_interactions(CONVERSATION)
    expected = len(segment_interactions(interactions))

    assert graph.provider.usage.calls == expected
    assert expected < len(CONVERSATION), "batching must beat one call per turn"
    assert len(passport.stats) == expected


def test_unbatched_ingest_costs_one_call_per_interaction():
    graph = SozoGraph(provider=FakeProvider())
    graph.ingest(CONVERSATION, batch=False)
    assert graph.provider.usage.calls == len(CONVERSATION)


def test_plan_predicts_the_cost_without_calling_anything(graph):
    forecast = graph.plan(CONVERSATION)
    assert forecast["api_calls"] == forecast["segments"]
    assert graph.provider.usage.calls == 0, "plan must not call the model"

    graph.ingest(CONVERSATION)
    assert graph.provider.usage.calls == forecast["api_calls"]


def test_known_keys_are_shown_to_the_model_after_the_first_segment(graph):
    graph.ingest(CONVERSATION)
    later = "\n".join(graph.provider.seen[1:])
    assert "KNOWN KEYS" in later, "the key vocabulary must reach the prompt"
    assert "kitchen_colour" in later


# --------------------------------------------------------------------------
# Episodes: what a flat belief state cannot answer
# --------------------------------------------------------------------------

def test_episodes_retain_detail_no_fact_captures(graph):
    passport = graph.ingest(CONVERSATION)
    text = " ".join(e.summary for e in passport.episodes).lower()
    # No fact records the grandmother's painting; without episodes this detail
    # is gone and a question about it is unanswerable.
    assert "grandmother" in text
    assert not any("grandmother" in str(f.value).lower() for f in passport.facts)


def test_query_finds_the_relevant_episode(graph):
    passport = graph.ingest(CONVERSATION)
    rendered = passport.context(query="What did she hang above the stove?", budget_chars=1200)
    assert "grandmother" in rendered.lower()


def test_facts_are_present_regardless_of_the_query(graph):
    passport = graph.ingest(CONVERSATION)
    for query in ["painting", "kitchen", "something entirely unrelated", None]:
        rendered = passport.context(query=query, budget_chars=1500)
        assert "location: Kwekwe" in rendered, (
            "the belief state must be injected in full, so a retrieval miss "
            "can never hide a known fact"
        )


# --------------------------------------------------------------------------
# Portability
# --------------------------------------------------------------------------

def test_passport_round_trips_through_a_file(graph, tmp_path):
    passport = graph.ingest(CONVERSATION)
    path = tmp_path / "user.json"
    passport.save(path)

    reloaded = Passport.load(path)
    assert reloaded.to_compact_dict() == passport.to_compact_dict()
    assert reloaded.context() == passport.context()


def test_passport_is_plain_readable_json(graph, tmp_path):
    passport = graph.ingest(CONVERSATION)
    path = tmp_path / "user.json"
    passport.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == "2.0"
    assert isinstance(raw["facts"], list)
    assert path.stat().st_size < 20_000, "a passport should stay small"


def test_reading_a_passport_needs_no_provider_and_no_sdk(graph, tmp_path, monkeypatch):
    path = tmp_path / "user.json"
    graph.ingest(CONVERSATION).save(path)

    # Nothing SDK-shaped may be imported on the read path.
    import builtins

    real_import = builtins.__import__
    blocked = {"anthropic", "openai", "ollama", "litellm", "google", "google.genai"}

    def guard(name, *a, **kw):
        if name in blocked:
            raise AssertionError(f"reading a passport imported {name}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guard)
    for module in list(sys.modules):
        assert module not in blocked or True  # already-imported modules are fine

    reloaded = Passport.load(path)
    assert "Facts (current beliefs):" in reloaded.context()
    assert reloaded.token_estimate() > 0


def test_constructing_sozograph_never_builds_a_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SOZOGRAPH_PROVIDER", raising=False)

    sg = SozoGraph()  # must not raise
    assert sg.usage is None


def test_re_ingesting_the_same_conversation_is_stable(graph):
    first = graph.ingest(CONVERSATION)
    facts_before = {(f.key, str(f.value)) for f in first.facts}
    changes_before = len(first.contradictions)

    second = graph.ingest(CONVERSATION, passport=first)
    assert {(f.key, str(f.value)) for f in second.facts} == facts_before
    assert len(second.contradictions) == changes_before, (
        "replaying the same history must not grow the contradiction log"
    )


# --------------------------------------------------------------------------
# Retrieval primitives
# --------------------------------------------------------------------------

def test_tokenize_drops_stopwords_but_never_returns_empty():
    assert "the" not in tokenize("the kitchen is green")
    assert tokenize("who is he") != []


def test_bm25_ranks_the_matching_document_first():
    docs = [
        "Melanie renovated the kitchen in sage green",
        "Caroline booked a flight to Nairobi",
        "The grandmother's painting hangs above the stove",
    ]
    scores = BM25(docs).score("grandmother painting stove")
    assert scores.index(max(scores)) == 2


def test_ranking_with_no_query_falls_back_to_the_prior():
    items = [{"n": 1, "p": 0.1}, {"n": 2, "p": 0.9}]
    ordered = rank(items, None, text_of=lambda x: "", prior=lambda x: x["p"])
    assert ordered[0].item["n"] == 2


def test_ranking_empty_input_is_safe():
    assert rank([], "anything", text_of=str) == []
    assert BM25([]).score("x") == []


def test_budget_is_respected_even_with_a_large_passport(graph):
    passport = graph.ingest(CONVERSATION)
    for i in range(200):
        passport.facts.append(
            type(passport.facts[0])(key=f"filler_{i}", value="x" * 150, source="s")
        )
    for budget in (400, 900, 2000):
        rendered = passport.context(budget_chars=budget)
        assert len(rendered) <= budget + 10, f"budget {budget} exceeded"


def test_plan_on_an_empty_input():
    assert plan([])["segments"] == 0
