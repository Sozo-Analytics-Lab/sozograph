from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sozograph import SozoGraph


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="Requires GEMINI_API_KEY (or GOOGLE_API_KEY) to run end-to-end extraction test.",
)
def test_end_to_end_ingest_transcript_and_export_context():
    sg = SozoGraph()

    transcript = (FIXTURES / "sample_transcript.txt").read_text(encoding="utf-8")

    passport, stats = sg.ingest(
        transcript,
        meta={"user_key": "u_fixture", "source": "transcript:fixture"}
    )

    # Passport must have valid compact dict
    compact = passport.to_compact_dict()
    assert compact["version"] == "1.0"
    assert "facts" in compact and "prefs" in compact

    # Export must be non-empty and within budget
    ctx = sg.export_context(passport, budget_chars=1500)
    assert isinstance(ctx, str)
    assert len(ctx) <= 1510
    assert "SOZOGRAPH PASSPORT v1" in ctx


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="Requires GEMINI_API_KEY (or GOOGLE_API_KEY) to run end-to-end extraction test.",
)
def test_end_to_end_ingest_objects_fire_rtdb_supabase():
    sg = SozoGraph()

    firestore_doc = json.loads((FIXTURES / "sample_firestore_doc.json").read_text(encoding="utf-8"))
    rtdb_node = json.loads((FIXTURES / "sample_rtdb_node.json").read_text(encoding="utf-8"))
    supa_row = json.loads((FIXTURES / "sample_supabase_row.json").read_text(encoding="utf-8"))

    passport, _ = sg.ingest(
        [firestore_doc, rtdb_node, supa_row],
        meta={"user_key": "u_fixture", "source": "mixed:fixtures"},
        # hint omitted on purpose to exercise auto-detection
    )

    ctx = sg.export_context(passport, budget_chars=1800)
    assert "Facts (current beliefs):" in ctx or "Preferences:" in ctx
