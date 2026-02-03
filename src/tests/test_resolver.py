from __future__ import annotations

from datetime import datetime, timezone

from sozograph.schema import Passport, Fact, Preference, Entity, OpenLoop
from sozograph.resolver import merge_passport_update


def dt(s: str) -> datetime:
    # ISO with Z support
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_fact_temporal_priority_and_contradiction():
    base = Passport(user_key="u1")

    # Existing fact
    base.facts.append(
        Fact(key="location", value="Harare", ts=dt("2026-02-01T10:00:00Z"), confidence=0.9, source="t1")
    )

    # Incoming newer update
    incoming = [
        Fact(key="location", value="Bulawayo", ts=dt("2026-02-03T10:00:00Z"), confidence=0.9, source="t2")
    ]

    out, stats = merge_passport_update(
        base,
        facts=incoming,
        prefs=[],
        entities=[],
        open_loops=[],
    )

    assert any(f.key == "location" and f.value == "Bulawayo" for f in out.facts)
    assert len(out.contradictions) == 1
    c = out.contradictions[0]
    assert c.key == "location"
    assert c.old == "Harare"
    assert c.new == "Bulawayo"
    assert stats.facts_upserted == 1
    assert stats.contradictions_added == 1


def test_fact_older_update_does_not_override_but_records_contradiction():
    base = Passport(user_key="u1")
    base.facts.append(
        Fact(key="role", value="developer", ts=dt("2026-02-03T10:00:00Z"), confidence=0.8, source="t2")
    )

    # Incoming older conflicting value
    incoming = [
        Fact(key="role", value="student", ts=dt("2026-02-01T10:00:00Z"), confidence=0.8, source="t1")
    ]

    out, stats = merge_passport_update(
        base,
        facts=incoming,
        prefs=[],
        entities=[],
        open_loops=[],
    )

    # Should keep newest
    assert any(f.key == "role" and f.value == "developer" for f in out.facts)
    # But contradiction should exist
    assert len(out.contradictions) == 1
    assert stats.contradictions_added == 1


def test_preference_merge_and_key_normalization():
    base = Passport(user_key="u1")
    base.prefs.append(
        Preference(key="Tone", value="direct", ts=dt("2026-02-02T10:00:00Z"), confidence=0.9, source="t1")
    )

    incoming = [
        Preference(key="tone", value="direct", ts=dt("2026-02-03T10:00:00Z"), confidence=0.7, source="t2")
    ]

    out, stats = merge_passport_update(
        base,
        facts=[],
        prefs=incoming,
        entities=[],
        open_loops=[],
    )

    # Same value; should not create contradiction
    assert len(out.contradictions) == 0
    assert any(p.key == "tone" and p.value == "direct" for p in out.prefs)
    # Should not count as upsert if value unchanged
    assert stats.prefs_upserted == 0


def test_entity_merge_by_alias():
    base = Passport()
    base.entities.append(Entity(name="SozoGraph", type="project", aliases=["Sozo Graph"]))

    incoming = [Entity(name="Sozo Graph", type="project", aliases=["SozoGraph v1"])]

    out, stats = merge_passport_update(
        base,
        facts=[],
        prefs=[],
        entities=incoming,
        open_loops=[],
    )

    assert len(out.entities) == 1
    e = out.entities[0]
    assert e.name == "SozoGraph"
    assert "Sozo Graph" in e.aliases
    assert "SozoGraph v1" in e.aliases
    assert stats.entities_merged == 1


def test_open_loop_dedupe():
    base = Passport()
    base.open_loops.append(OpenLoop(item="Finalize v1 repo", ts=dt("2026-02-02T10:00:00Z"), source="t1"))

    incoming = [OpenLoop(item="  finalize   v1  repo  ", ts=dt("2026-02-03T10:00:00Z"), source="t2")]

    out, stats = merge_passport_update(
        base,
        facts=[],
        prefs=[],
        entities=[],
        open_loops=incoming,
    )

    assert len(out.open_loops) == 1
    assert out.open_loops[0].source == "t2"
    assert stats.open_loops_added == 1
