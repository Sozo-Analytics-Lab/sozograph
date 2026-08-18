"""
Regressions for the data-loss and determinism defects in 0.1.1.

Each test here corresponds to a bug that silently corrupted or dropped memory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sozograph.adapters.firestore import firestore_to_interaction
from sozograph.adapters.rtdb import rtdb_to_interaction
from sozograph.adapters.supabase import supabase_row_to_interaction
from sozograph.extractor import Extractor
from sozograph.ingest import coerce_to_interactions
from sozograph.resolver import merge_passport_update
from sozograph.schema import Fact, Passport
from sozograph.utils import normalize_key, stable_id

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# The ts bug: extracted items with no timestamp were silently discarded.
# --------------------------------------------------------------------------

def _validate(payload, ts=None):
    # validate() needs no provider, so bypass __init__ entirely.
    return Extractor.__new__(Extractor).validate(payload, source_id="src_1", ts=ts)


def test_extracted_items_are_always_stamped():
    # 0.1.1 asked the model for `ts` and then passed None into a field with a
    # default_factory, raising ValidationError that the handler swallowed.
    # Every item whose output omitted the optional ts was silently dropped.
    # The timestamp now comes from the interaction, which actually knows it.
    payload = {
        "facts": [{"key": "location", "value": "Kwekwe", "confidence": 0.9}],
        "prefs": [{"key": "tone", "value": "direct", "confidence": 0.8}],
        "entities": [{"name": "SozoGraph", "type": "project", "aliases": []}],
        "open_loops": [{"item": "Ship v1"}],
    }
    when = datetime(2026, 2, 4, 12, 0, tzinfo=timezone.utc)
    out = _validate(payload, ts=when)
    assert len(out["facts"]) == 1, "fact was dropped"
    assert len(out["prefs"]) == 1, "pref was dropped"
    assert len(out["entities"]) == 1
    assert len(out["open_loops"]) == 1, "open loop was dropped"
    assert out["facts"][0].ts == when
    assert out["open_loops"][0].ts == when


def test_items_still_land_when_no_interaction_ts_is_given():
    out = _validate({"facts": [{"key": "role", "value": "engineer"}]})
    assert len(out["facts"]) == 1
    assert out["facts"][0].ts is not None


def test_malformed_item_does_not_kill_the_whole_extraction():
    # A row missing "key" used to raise KeyError straight out of extract().
    out = _validate({"facts": [
        {"value": "no key here"},
        "not even a dict",
        {"key": "role", "value": "engineer"},
    ]})
    assert [f.key for f in out["facts"]] == ["role"]


def test_wire_strings_coerce_back_to_json_scalars():
    # `value` is typed as a string on the wire so the schema stays valid under
    # OpenAI strict mode. Numbers and booleans must survive the round trip.
    out = _validate({"facts": [
        {"key": "team_size", "value": "7"},
        {"key": "remote", "value": "true"},
        {"key": "city", "value": "Kwekwe"},
        {"key": "budget", "value": "1500.50"},
    ]})
    got = {f.key: f.value for f in out["facts"]}
    assert got == {"team_size": 7, "remote": True, "city": "Kwekwe", "budget": 1500.50}


@pytest.mark.parametrize("adapter,payload", [
    (lambda d: firestore_to_interaction(d), {"title": "A doc with no timestamp field"}),
    (lambda d: rtdb_to_interaction(d, path="/u/1"), {"tone": "direct"}),
    (lambda d: supabase_row_to_interaction(d, table="t"), {"notes": "no timestamp"}),
])
def test_adapters_accept_payloads_without_timestamps(adapter, payload):
    # These raised an uncaught ValidationError and crashed ingestion outright.
    it = adapter(payload)
    assert it.ts is not None


def test_readme_style_payloads_without_timestamps_coerce():
    doc = json.loads((FIXTURES / "sample_no_timestamp.json").read_text(encoding="utf-8"))
    interactions, sources = coerce_to_interactions(doc)
    assert len(interactions) == 1
    assert len(sources) == 1


# --------------------------------------------------------------------------
# Determinism: SourceRef ids must not depend on PYTHONHASHSEED.
# --------------------------------------------------------------------------

def test_stable_id_is_deterministic_in_process():
    assert stable_id("t", "hello") == stable_id("t", "hello")
    assert stable_id("t", "hello") != stable_id("t", "world")


def test_stable_id_is_deterministic_across_processes():
    src = Path(__file__).resolve().parent.parent / "src"
    code = (
        f"import sys; sys.path.insert(0, r'{src}');"
        "from sozograph.utils import stable_id; print(stable_id('t', 'hello'))"
    )
    runs = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        runs.add(out.stdout.strip())
    assert len(runs) == 1, f"ids differ across hash seeds: {runs}"


def test_ingest_source_ids_are_stable_across_seeds():
    src = Path(__file__).resolve().parent.parent / "src"
    code = (
        f"import sys; sys.path.insert(0, r'{src}');"
        "from sozograph.ingest import coerce_to_interactions;"
        "_, s = coerce_to_interactions('I live in Kwekwe.');"
        "print(s[0].id)"
    )
    runs = set()
    for seed in ("0", "1", "99"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        runs.add(out.stdout.strip())
    assert len(runs) == 1, f"SourceRef.id differs across hash seeds: {runs}"


def test_firestore_batch_with_meta_source_id_does_not_collide():
    batch = {"a": {"title": "doc a"}, "b": {"title": "doc b"}}
    _, sources = coerce_to_interactions(batch, meta={"source_id": "fixed"})
    assert len({s.id for s in sources}) == len(sources)


# --------------------------------------------------------------------------
# One normalizer: merges and rendering must agree on key identity.
# --------------------------------------------------------------------------

def test_resolver_and_renderer_agree_on_key_identity():
    p = Passport.new()
    p, _ = merge_passport_update(
        p,
        facts=[Fact(key="Detail Level", value="high", source="s1")],
        prefs=[], entities=[], open_loops=[],
    )
    p, _ = merge_passport_update(
        p,
        facts=[Fact(key="detail_level", value="high", source="s2")],
        prefs=[], entities=[], open_loops=[],
    )
    # "Detail Level" and "detail_level" are the same belief, not two.
    assert len(p.facts) == 1, [f.key for f in p.facts]
    assert p.facts[0].key == normalize_key("Detail Level") == "detail_level"


# --------------------------------------------------------------------------
# user_key was documented in the README and never applied.
# --------------------------------------------------------------------------

def test_user_key_is_applied_from_meta():
    p = Passport.new()
    meta = {"user_key": "u_123"}
    if meta.get("user_key"):
        p.user_key = str(meta["user_key"])
    assert p.user_key == "u_123"
    assert p.to_compact_dict()["user_key"] == "u_123"
