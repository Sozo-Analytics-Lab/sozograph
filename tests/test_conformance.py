"""Passport 3.1 conformance: every fixture's projection and head, from its stamp log alone."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sozograph import Passport
from sozograph.projection import project
from sozograph.stamps import merkle_root

FIXTURES = sorted((Path(__file__).resolve().parents[1] / "conformance" / "fixtures").glob("*.json"))
SECTIONS = ("facts", "prefs", "entities", "open_loops", "episodes", "observations", "contradictions")


def test_fixtures_exist():
    assert len(FIXTURES) >= 6


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_projection_matches_the_fixture(path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    p = Passport.from_dict({"version": "3.1", "holder": {"id": "holder"}, **fixture["input"]})
    project(p)
    data = json.loads(p.to_json())
    assert merkle_root([s["id"] for s in fixture["input"]["stamps"]]) == fixture["expected"]["head"]
    for section in SECTIONS:
        assert data.get(section, []) == fixture["expected"][section], section


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_reversed_delivery_gives_the_same_projection(path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    stamps = list(reversed(fixture["input"]["stamps"]))
    p = Passport.from_dict({"version": "3.1", "holder": {"id": "holder"}, "stamps": stamps,
                            "agents": fixture["input"]["agents"]})
    project(p)
    data = json.loads(p.to_json())
    for section in SECTIONS:
        assert data.get(section, []) == fixture["expected"][section], section
