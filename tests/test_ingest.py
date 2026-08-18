"""
Live smoke tests against a real provider.

Skipped unless a key is configured. Everything else in the suite runs against
fake transports; this is the one place that proves the wire format actually
works against a real engine.

    pytest -m live
    SOZOGRAPH_PROVIDER=anthropic pytest -m live
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sozograph import Passport, SozoGraph

FIXTURES = Path(__file__).parent / "fixtures"

_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "SOZOGRAPH_PROVIDER")
requires_provider = pytest.mark.skipif(
    not any(os.getenv(k) for k in _KEYS),
    reason=(
        "No provider configured. Set one of "
        f"{', '.join(_KEYS)} to run the live smoke tests."
    ),
)

pytestmark = [pytest.mark.live, requires_provider]


@pytest.fixture(scope="module")
def graph() -> SozoGraph:
    return SozoGraph()


def test_transcript_round_trips_through_a_real_provider(graph, tmp_path):
    transcript = (FIXTURES / "sample_transcripts.txt").read_text(encoding="utf-8")

    passport = graph.ingest(transcript, meta={"user_key": "u_fixture"})

    assert passport.user_key == "u_fixture"
    assert not passport.is_empty(), "a real provider returned nothing usable"
    assert graph.usage.calls >= 1
    assert graph.usage.total_tokens > 0

    compact = passport.to_compact_dict()
    assert compact["version"] == "2.0"
    assert compact["user_key"] == "u_fixture"

    # The wire format must survive a real round trip, not just a synthetic one.
    path = tmp_path / "live.json"
    passport.save(path)
    assert Passport.load(path).to_compact_dict() == compact

    context = passport.context(budget_chars=1500)
    assert len(context) <= 1510
    assert "SOZOGRAPH PASSPORT" in context


def test_schema_enforcement_holds_on_a_real_engine(graph):
    """
    Every provider claims native structured output. This checks the claim.
    """
    passport = graph.ingest(
        "My name is Rairo. I live in Kwekwe, I lead a team of seven, "
        "and I prefer short direct answers with no preamble."
    )

    for fact in passport.facts:
        assert fact.key == fact.key.strip().lower()
        assert 0.0 <= fact.confidence <= 1.0
        assert fact.source
    for episode in passport.episodes:
        assert episode.summary.strip()
        assert 0.0 <= episode.salience <= 1.0

    keys = {f.key for f in passport.facts} | {p.key for p in passport.prefs}
    assert keys, "nothing was extracted from an information-dense transcript"


def test_mixed_database_objects_ingest(graph):
    firestore = json.loads((FIXTURES / "sample_firestore_doc.json").read_text(encoding="utf-8"))
    rtdb = json.loads((FIXTURES / "sample_rtdb_node.json").read_text(encoding="utf-8"))
    supabase = json.loads((FIXTURES / "sample_supabase_row.json").read_text(encoding="utf-8"))
    no_ts = json.loads((FIXTURES / "sample_no_timestamp.json").read_text(encoding="utf-8"))

    passport = graph.ingest([firestore, rtdb, supabase, no_ts])

    context = passport.context(budget_chars=1800)
    assert "Facts (current beliefs):" in context or "Preferences:" in context


def test_chat_turns_produce_episodes(graph):
    passport = graph.ingest([
        {"speaker": "Melanie", "text": "I renovated my kitchen in sage green.",
         "ts": "2026-01-05T10:00:00Z", "session": "1"},
        {"speaker": "Caroline", "text": "What did you put on the wall?",
         "ts": "2026-01-05T10:01:00Z", "session": "1"},
        {"speaker": "Melanie", "text": "My grandmother's painting, above the stove.",
         "ts": "2026-01-05T10:02:00Z", "session": "1"},
    ])

    assert passport.episodes, "a conversation must produce at least one episode"
    rendered = passport.context(query="What is above the stove?", budget_chars=1200)
    assert rendered.strip()
