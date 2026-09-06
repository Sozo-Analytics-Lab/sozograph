from __future__ import annotations

from sozograph.evidence import deterministic_quote
from sozograph.schema import Observation


def test_deterministic_quote_anchors_a_paraphrased_observation_to_source_clause():
    source = "Melanie said she put the painting above the stove. The room was quiet."
    item = Observation(
        text="Melanie hung the painting above the stove.",
        source="s1",
    )
    quote = deterministic_quote("observations", item, source)
    assert quote == "Melanie said she put the painting above the stove."


def test_deterministic_quote_rejects_weak_lexical_overlap():
    source = "Harare is home."
    item = Observation(text="The user lives in Zimbabwe.", source="s1")
    assert deterministic_quote("observations", item, source) is None
