from __future__ import annotations

from datetime import datetime, timezone

from sozograph.schema import Passport, Fact, Preference, Entity, OpenLoop, Contradiction
from sozograph.render import export_context


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

    assert "SOZOGRAPH PASSPORT v1" in txt
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
