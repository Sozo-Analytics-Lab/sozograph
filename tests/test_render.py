from __future__ import annotations

from datetime import datetime, timezone

from sozograph.render import export_context
from sozograph.schema import (
    Contradiction,
    Entity,
    Fact,
    Observation,
    OpenLoop,
    Passport,
    Preference,
)


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_export_context_basic_sections():
    p = Passport(user_key="u1")
    p.facts.append(Fact(key="role", value="developer", ts=dt("2026-02-03T10:00:00Z"), confidence=0.9, source="t1"))
    p.prefs.append(Preference(key="tone", value="direct", ts=dt("2026-02-03T10:00:00Z"), confidence=0.9, source="t1"))
    p.entities.append(Entity(name="SozoGraph", type="project", aliases=["Sozo Graph"]))
    p.open_loops.append(OpenLoop(item="Finalize v1 repo", ts=dt("2026-02-03T10:00:00Z"), source="t1"))
    p.contradictions.append(
        Contradiction(
            key="location",
            old="Harare",
            new="Bulawayo",
            ts_old=dt("2026-02-01T10:00:00Z"),
            ts_new=dt("2026-02-03T10:00:00Z"),
            source_old="t0",
            source_new="t1",
        )
    )

    txt = export_context(p, budget_chars=3000)

    assert "SOZOGRAPH PASSPORT" in txt
    assert "Facts (current beliefs):" in txt
    assert "- role: developer" in txt
    assert "Preferences:" in txt
    assert "- tone: direct" in txt
    assert "Key entities:" in txt
    assert "- SozoGraph (project)" in txt
    assert "Open loops:" in txt
    assert "Finalize v1 repo" in txt
    assert "Recent updates (contradictions resolved by time):" in txt
    assert "location changed" in txt


def test_export_context_budget_trims():
    p = Passport(user_key="u1")
    # Add many facts to force trimming
    for i in range(60):
        p.facts.append(
            Fact(
                key=f"fact_{i}",
                value="x" * 200,
                ts=dt("2026-02-03T10:00:00Z"),
                confidence=0.5,
                source="t1",
            )
        )

    txt = export_context(p, budget_chars=900)

    # Must not exceed budget by much (allow tiny overhead due to truncation char)
    assert len(txt) <= 910
    assert "Facts (current beliefs):" in txt


def test_query_ranks_facts_past_the_cap():
    """Past caps.facts, a matching old low-confidence fact survives the cut."""
    p = Passport()
    p.facts.append(
        Fact(
            key="childhood_pet",
            value="golden retriever named Biscuit",
            ts=dt("2020-01-01T00:00:00Z"),
            confidence=0.3,
            source="t0",
        )
    )
    for i in range(69):
        p.facts.append(
            Fact(
                key=f"note_{i}",
                value=f"unrelated detail {i}",
                ts=dt("2026-02-03T10:00:00Z"),
                confidence=0.9,
                source="t1",
            )
        )

    txt = export_context(p, query="what pet did she have growing up", budget_chars=20_000)
    assert "childhood_pet" in txt
    assert len(p.facts) == 70


def test_no_query_drops_oldest_fact_past_the_cap():
    """Without a query, prior order alone decides, as before."""
    p = Passport()
    p.facts.append(
        Fact(
            key="childhood_pet",
            value="golden retriever named Biscuit",
            ts=dt("2020-01-01T00:00:00Z"),
            confidence=0.3,
            source="t0",
        )
    )
    for i in range(69):
        p.facts.append(
            Fact(
                key=f"note_{i}",
                value=f"unrelated detail {i}",
                ts=dt("2026-02-03T10:00:00Z"),
                confidence=0.9,
                source="t1",
            )
        )

    txt = export_context(p, budget_chars=20_000)
    assert "childhood_pet" not in txt


def test_query_ranks_prefs_and_loops():
    p = Passport()
    for i in range(35):
        p.prefs.append(
            Preference(key=f"style_{i}", value="generic", ts=dt("2026-02-03T10:00:00Z"), confidence=0.9, source="t1")
        )
    # Old, low-confidence pref that matches the query.
    p.prefs.append(
        Preference(key="email_style", value="wants bullet points", ts=dt("2021-06-01T00:00:00Z"), confidence=0.4, source="t0")
    )
    for i in range(15):
        p.open_loops.append(OpenLoop(item=f"misc task {i}", ts=dt("2026-02-03T10:00:00Z"), source="t1"))
    p.open_loops.append(OpenLoop(item="renew passport before Berlin trip", ts=dt("2020-06-01T00:00:00Z"), source="t0"))

    txt = export_context(p, query="bullet points email", budget_chars=30_000)
    assert "email_style" in txt

    txt2 = export_context(p, query="passport berlin", budget_chars=30_000)
    assert "renew passport" in txt2


def test_search_texts_cover_key_value_item():
    f = Fact(key="home_city", value="Kwekwe", source="s")
    pref = Preference(key="tone", value="terse", source="s")
    e = Entity(name="SozoGraph", type="project", aliases=["Sozo Graph"])
    loop = OpenLoop(item="book flight", source="s")
    obs = Observation(text="Oliver hid his bone in Melanie's slipper", source="s",
                      participants=["Melanie"])

    assert "kwekwe" in f.search_text().lower()
    assert "terse" in pref.search_text().lower()
    assert "sozo graph" in e.search_text().lower()
    assert "flight" in loop.search_text().lower()
    assert "slipper" in obs.search_text().lower()
    assert "melanie" in obs.search_text().lower()


def test_observations_render_and_rank_against_query():
    """A single-hop detail lives in an observation and surfaces on its query."""
    p = Passport()
    # The answer-bearing observation, old and buried among noise.
    p.observations.append(
        Observation(
            text="Oliver the dog hid his bone in Melanie's slipper",
            ts=dt("2021-01-01T00:00:00Z"), source="t0", participants=["Melanie"],
        )
    )
    for i in range(80):
        p.observations.append(
            Observation(text=f"unrelated remark number {i} about the weather",
                        ts=dt("2026-02-03T10:00:00Z"), source="t1")
        )

    txt = export_context(p, query="where did Oliver hide his bone", budget_chars=20_000)
    assert "Details recalled:" in txt
    assert "slipper" in txt


def test_observations_survive_a_tight_budget_when_relevant():
    """Past the cap and under a small budget, the matched observation wins."""
    p = Passport()
    p.observations.append(
        Observation(text="Tim is learning German for his semester in Galway",
                    ts=dt("2020-06-01T00:00:00Z"), source="t0")
    )
    for i in range(120):
        p.observations.append(
            Observation(text=f"generic filler observation {i}",
                        ts=dt("2026-02-03T10:00:00Z"), source="t1")
        )
    txt = export_context(p, query="which language is Tim learning", budget_chars=1500)
    assert "German" in txt
