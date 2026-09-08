from __future__ import annotations

from typing import Any

from sozograph.prompts import ARRAY_LIMITS, EXTRACTION_SCHEMA, EXTRACTOR_SYSTEM_PROMPT

#: Keywords OpenAI strict mode accepts. maxItems is in the supported subset,
#: so the array bounds stay portable across every provider dialect.
_OPENAI_STRICT_ALLOWED = {
    "type", "enum", "properties", "required", "additionalProperties",
    "items", "anyOf", "minItems", "maxItems", "minimum", "maximum",
    "description", "pattern", "format", "default",
}


def _walk(node: Any) -> list[dict]:
    """Every schema node in the tree. The `properties` value is a name-keyed
    map of schemas rather than a schema itself, so its children are walked."""
    out: list[dict] = []
    if isinstance(node, dict):
        out.append(node)
        for k, v in node.items():
            if k == "properties" and isinstance(v, dict):
                for sub in v.values():
                    out.extend(_walk(sub))
            else:
                out.extend(_walk(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_walk(v))
    return out


def test_every_extraction_array_is_bounded():
    arrays = [
        n for n, s in EXTRACTION_SCHEMA["properties"].items()
        if isinstance(s, dict) and s.get("type") == "array"
    ]
    assert set(arrays) == {"facts", "prefs", "entities", "open_loops", "observations"}
    for name in arrays:
        assert EXTRACTION_SCHEMA["properties"][name]["maxItems"] == ARRAY_LIMITS[name]

    # Observation statements carry a resolved event date alongside the text.
    obs_items = EXTRACTION_SCHEMA["properties"]["observations"]["items"]
    assert set(obs_items["required"]) == {"text", "when"}

    episode = EXTRACTION_SCHEMA["properties"]["episode"]
    for name in ("participants", "keywords"):
        assert episode["properties"][name]["maxItems"] == ARRAY_LIMITS[name]
    assert (
        EXTRACTION_SCHEMA["properties"]["entities"]["items"]["properties"]["aliases"]["maxItems"]
        == ARRAY_LIMITS["aliases"]
    )


def test_schema_stays_inside_openai_strict_subset():
    for node in _walk(EXTRACTION_SCHEMA):
        keys = set(node) - {"description"}
        assert keys <= _OPENAI_STRICT_ALLOWED, f"unsupported keywords: {keys - _OPENAI_STRICT_ALLOWED}"
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) >= set(node.get("properties", {}))


def test_extractor_prompt_resolves_relative_dates():
    assert "TIMESTAMP" in EXTRACTOR_SYSTEM_PROMPT
    assert "absolute" in EXTRACTOR_SYSTEM_PROMPT.lower()


def test_every_required_wire_string_rejects_empty_and_blank():
    # A real Kaggle run's rejected_reasons showed the model emitting an
    # empty string for a required field ("String should have at least 1
    # character") -- syntactically valid JSON, semantically rejected
    # downstream with nothing to stop it at generation time. `minLength`
    # would be the obvious fix but is outside OpenAI strict mode's
    # supported subset (see test above); `pattern` is in it.
    import re

    props = EXTRACTION_SCHEMA["properties"]
    checks = [
        props["facts"]["items"]["properties"]["key"],
        props["prefs"]["items"]["properties"]["key"],
        props["entities"]["items"]["properties"]["name"],
        props["open_loops"]["items"]["properties"]["item"],
        props["observations"]["items"]["properties"]["text"],
        props["episode"]["properties"]["summary"],
    ]
    for node in checks:
        pattern = node["pattern"]
        assert not re.search(pattern, ""), f"{pattern} must reject an empty string"
        assert not re.search(pattern, "   "), f"{pattern} must reject a whitespace-only string"
        assert re.search(pattern, "x"), f"{pattern} must accept real content"
