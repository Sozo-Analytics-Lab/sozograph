from __future__ import annotations

from datetime import datetime, timezone

from sozograph.render import Caps, export_context
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


def test_entity_expansion_surfaces_the_whole_subject():
    """A multi-hop list question needs every observation about its subject,
    including ones sharing no distinctive word with the query."""
    p = Passport()
    p.entities.append(Entity(name="Tim", type="person"))
    # Ten Tim observations, most lexically unrelated to "summer activities".
    tim_lines = [
        "Tim kayaked the river gorge",
        "Tim is learning German for his semester in Galway",
        "Tim adopted a border collie named Oliver",
        "Tim rebuilt the deck railing himself",
        "Tim plays goalkeeper on the office five-a-side team",
        "Tim quit caffeine in March and regrets nothing",
        "Tim's sourdough starter survived three weeks away",
        "Tim fixed the office espresso machine twice",
        "Tim cycled two hundred kilometres last month",
        "Tim is reading a biography of Roald Amundsen",
    ]
    for line in tim_lines:
        p.observations.append(
            Observation(text=line, ts=dt("2026-01-10T00:00:00Z"), source="t0")
        )
    # Noise floor above the observation cap so a cut definitely happens.
    for i in range(60):
        p.observations.append(
            Observation(text=f"unrelated chatter item {i} about the neighbours",
                        ts=dt("2026-02-03T10:00:00Z"), source="t1")
        )

    txt = export_context(p, query="What did Tim do this summer?",
                         budget_chars=30_000, caps=Caps(observations=20))
    recalled = [line for line in txt.splitlines()
                if line.startswith("- ") and "Tim" in line]
    assert len(recalled) >= 8


def test_entity_expansion_beats_the_cap_against_higher_scoring_noise():
    """The subject's records must survive the cut even when lexically stronger
    off-subject records would otherwise fill it. This fails on plain BM25 top-k;
    only lifting the named subject above the cut passes it."""
    p = Passport()
    p.entities.append(Entity(name="Tim", type="person"))
    # Three on-subject records that never mention the query's strong terms.
    for line in ("Tim kayaked the river gorge",
                 "Tim rebuilt the deck railing",
                 "Tim adopted a border collie"):
        p.observations.append(
            Observation(text=line, ts=dt("2026-01-10T00:00:00Z"), source="t0",
                        participants=["Tim"])
        )
    # Ten off-subject records that match the query's strong terms strongly, so
    # a pure BM25 top-5 is entirely these and excludes every Tim record.
    for i in range(10):
        p.observations.append(
            Observation(text=f"the summer beach holiday itinerary draft {i}",
                        ts=dt("2026-02-03T10:00:00Z"), source="t1")
        )

    txt = export_context(p, query="What did Tim do on his summer beach holiday?",
                         budget_chars=30_000, caps=Caps(observations=5))
    for line in ("kayaked", "deck railing", "border collie"):
        assert line in txt, f"entity expansion did not surface: {line}"


def test_observation_event_date_is_annotated_and_sorted():
    """Temporal queries read a timeline: event dates shown, ordered."""
    p = Passport()
    p.observations.append(
        Observation(text="Melanie ran the charity race",
                    ts=dt("2026-02-01T00:00:00Z"), source="t0",
                    when="2026-05-21")
    )
    p.observations.append(
        Observation(text="Melanie adopted Oliver from the shelter",
                    ts=dt("2026-01-15T00:00:00Z"), source="t0",
                    when="2026-01-04")
    )
    p.observations.append(
        Observation(text="Melanie repainted the kitchen sage green",
                    ts=dt("2026-03-01T00:00:00Z"), source="t0",
                    when="2026-03-12")
    )

    txt = export_context(p, query="when did Melanie adopt Oliver?", budget_chars=6_000)
    lines = [line for line in txt.splitlines() if line.startswith("- [")]
    dates = [line.split("]")[0][2:] for line in lines]
    assert dates == sorted(dates)
    assert "[2026-01-04]" in txt

    # Without temporal intent, relevance order stays.
    txt2 = export_context(p, query="kitchen colour", budget_chars=6_000)
    assert "[2026-03-12]" in txt2
    first = next(line for line in txt2.splitlines() if line.startswith("- "))
    assert "sage green" in first


def test_co_occurrence_graph_surfaces_a_relative_never_named_in_the_query():
    """A multi-hop question naming only the subject must also surface a
    record about a family member it never names, because the two co-occur
    elsewhere in the passport. This is the graph, not entity expansion: no
    substring in the query matches "Caroline" at all."""
    p = Passport()
    # Establishes the co-occurrence edge: Melanie and Caroline appear together.
    p.observations.append(
        Observation(text="Melanie and Caroline went apple picking together",
                    ts=dt("2026-01-01T00:00:00Z"), source="t0",
                    participants=["Melanie", "Caroline"])
    )
    # Caroline-only record the graph should lift; nothing here names Melanie.
    p.observations.append(
        Observation(text="Caroline started volunteering at the animal shelter",
                    ts=dt("2026-01-10T00:00:00Z"), source="t0",
                    participants=["Caroline"])
    )
    # Noise: unrelated to either, must not be lifted.
    for i in range(20):
        p.observations.append(
            Observation(text=f"unrelated chatter item {i} about the commute",
                        ts=dt("2026-02-03T10:00:00Z"), source="t1")
        )

    txt = export_context(p, query="What does Melanie do with her family?",
                         budget_chars=30_000, caps=Caps(observations=5))
    assert "volunteering at the animal shelter" in txt


def test_co_occurrence_graph_never_fires_without_a_direct_name_match():
    """No name from the query appears anywhere, so rank_expanded returns
    before the graph is ever consulted. Everything ties at zero score, so a
    correct implementation keeps the cap's five lowest-index items; a false
    graph lift would instead jump the high-index Caroline record ahead of
    them despite ties, which is exactly what this catches. The noise text
    shares no word with the query, so nothing here wins on lexical score
    either -- the only way "volunteering" could appear is a stray bonus."""
    p = Passport()
    for i in range(20):
        p.observations.append(
            Observation(text=f"routine grocery note number {i} about milk and bread",
                        ts=dt("2026-02-03T10:00:00Z"), source="t1")
        )
    # Appended last, so these sit at the highest indices: only a wrongly
    # applied bonus could pull the Caroline record ahead of the tied noise.
    p.observations.append(
        Observation(text="Melanie and Caroline went apple picking together",
                    ts=dt("2026-01-01T00:00:00Z"), source="t0",
                    participants=["Melanie", "Caroline"])
    )
    p.observations.append(
        Observation(text="Caroline started volunteering at the animal shelter",
                    ts=dt("2026-01-10T00:00:00Z"), source="t0",
                    participants=["Caroline"])
    )

    txt = export_context(p, query="what is the forecast for this weekend",
                         budget_chars=30_000, caps=Caps(observations=5))
    assert "volunteering at the animal shelter" not in txt
