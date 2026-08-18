"""
Deduplication pipeline: Tier 0 through Tier 3.

The headline assertion here is negative. A naive "merge above 0.85 similarity"
rule scores `budget_min` against `budget_max` at 0.92 and would fold two
opposite beliefs into one, unrecoverably. Every pair in
`test_polarity_guard_refuses_dangerous_merges` is one such trap.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sozograph.compact import compact
from sozograph.dedupe import (
    Verdict,
    compare,
    find_match,
    jaro,
    jaro_winkler,
    polarity_conflict,
    token_similarity,
)
from sozograph.prompts import format_known_keys
from sozograph.providers.base import LLMProvider
from sozograph.resolver import merge_passport_update
from sozograph.schema import Fact, Passport, Preference


def dt(offset_days: int = 0) -> datetime:
    return datetime(2026, 2, 1, tzinfo=timezone.utc) + timedelta(days=offset_days)


class ScriptedProvider(LLMProvider):
    """Returns a queued payload. Records what it was asked."""

    name = "scripted"

    def __init__(self, *payloads):
        super().__init__(model="scripted")
        self.payloads = list(payloads)
        self.prompts = []

    def complete_json(self, *, system, user, schema, temperature=0.2):
        self.prompts.append(user)
        self.usage.add(10, 5)
        return self.payloads.pop(0) if self.payloads else {}

    def complete_text(self, *, system, user, temperature=0.2):
        self.usage.add(10, 5)
        return "text"


# --------------------------------------------------------------------------
# String metrics
# --------------------------------------------------------------------------

def test_jaro_endpoints():
    assert jaro("abc", "abc") == 1.0
    assert jaro("", "abc") == 0.0
    assert jaro("abc", "xyz") == 0.0
    assert 0.0 < jaro("martha", "marhta") < 1.0


def test_jaro_winkler_rewards_shared_prefix():
    assert jaro_winkler("code_style", "code_styling") > jaro("code_style", "code_styling")
    assert jaro_winkler("abc", "abc") == 1.0


def test_token_similarity_is_order_independent():
    assert token_similarity("user_location", "location_user") == 1.0
    assert token_similarity("a_b", "c_d") == 0.0


# --------------------------------------------------------------------------
# Tier 2: the safety property
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("budget_min", "budget_max"),
    ("minimum_price", "maximum_price"),
    ("is_enabled", "is_disabled"),
    ("start_date", "end_date"),
    ("first_name", "last_name"),
    ("allow_list", "deny_list"),
    ("include_tests", "exclude_tests"),
    ("has_access", "has_no_access"),
    ("available", "unavailable"),
    ("likes_jazz", "dislikes_jazz"),
    ("child_1", "child_2"),
    ("sender_email", "recipient_email"),
    ("old_password", "new_password"),
    ("primary_contact", "secondary_contact"),
])
def test_polarity_guard_refuses_dangerous_merges(a, b):
    match = compare(a, b)
    assert not match.is_merge, (
        f"{a} and {b} mean opposite things and must never merge "
        f"(similarity {jaro_winkler(a, b):.3f})"
    )
    assert polarity_conflict(a, b) is not None


def test_similar_words_are_not_treated_as_synonyms():
    # 0.967 under Jaro-Winkler, and completely unrelated preferences.
    match = compare("likes_coffee", "likes_toffee")
    assert not match.is_merge
    assert jaro_winkler("likes_coffee", "likes_toffee") > 0.95


@pytest.mark.parametrize("a,b", [
    ("Detail Level", "detail_level"),
    ("code style", "code_style"),
    ("Tone", "tone"),
    ("user_location", "location_user"),
])
def test_confident_pairs_merge(a, b):
    assert compare(a, b).is_merge


@pytest.mark.parametrize("a,b", [
    ("code_style", "code_styling"),
    ("work_location", "work_locale"),
])
def test_uncertain_band_defers_to_semantic_review(a, b):
    assert compare(a, b).verdict is Verdict.REVIEW


@pytest.mark.parametrize("a,b", [
    ("code_style", "boilerplate_preference"),
    ("location", "team_size"),
])
def test_unrelated_keys_are_distinct(a, b):
    assert compare(a, b).verdict is Verdict.DISTINCT


def test_find_match_prefers_a_block_over_a_weaker_merge():
    match = find_match("budget_min", ["budget_max", "unrelated_thing"])
    assert match.verdict is Verdict.BLOCKED


def test_find_match_on_empty_vocabulary():
    assert find_match("anything", []).verdict is Verdict.DISTINCT


# --------------------------------------------------------------------------
# Tier 0: controlled vocabulary
# --------------------------------------------------------------------------

def test_known_keys_feed_the_prompt():
    p = Passport.new()
    p.facts.append(Fact(key="location", value="Kwekwe", source="s", ts=dt(1)))
    p.prefs.append(Preference(key="code_style", value="minimal", source="s", ts=dt(2)))
    keys = p.known_keys()
    assert set(keys) == {"location", "code_style"}

    block = format_known_keys(keys)
    assert "KNOWN KEYS" in block
    assert "code_style" in block


def test_known_keys_block_is_empty_for_a_new_passport():
    assert format_known_keys([]) == ""


def test_known_keys_are_capped():
    block = format_known_keys([f"key_{i}" for i in range(300)], limit=10)
    assert "+290 more" in block


# --------------------------------------------------------------------------
# Tier 1/2 through the resolver
# --------------------------------------------------------------------------

def test_case_and_punctuation_variants_collapse():
    p = Passport.new()
    for key in ("Detail Level", "detail level", "detail_level", "DETAIL_LEVEL"):
        p, _ = merge_passport_update(
            p, facts=[Fact(key=key, value="high", source="s", ts=dt(1))]
        )
    assert len(p.facts) == 1
    assert p.facts[0].key == "detail_level"


def test_opposing_keys_stay_separate_through_the_resolver():
    p = Passport.new()
    p, _ = merge_passport_update(p, facts=[Fact(key="budget_min", value=100, source="s")])
    p, stats = merge_passport_update(p, facts=[Fact(key="budget_max", value=900, source="s")])
    assert {f.key for f in p.facts} == {"budget_min", "budget_max"}
    assert stats.dedupe.blocked, "the block should be recorded for audit"
    assert p.meta["dedupe"]["blocked"]


def test_case_differing_values_are_not_a_contradiction():
    p = Passport.new()
    p, _ = merge_passport_update(p, prefs=[Preference(key="tone", value="Direct",
                                                      source="s", ts=dt(1))])
    p, stats = merge_passport_update(p, prefs=[Preference(key="tone", value="direct",
                                                          source="s", ts=dt(2))])
    assert p.contradictions == []
    assert stats.contradictions_added == 0


def test_numeric_string_and_number_are_the_same_value():
    p = Passport.new()
    p, _ = merge_passport_update(p, facts=[Fact(key="team_size", value=7, source="s", ts=dt(1))])
    p, _ = merge_passport_update(p, facts=[Fact(key="team_size", value="7", source="s", ts=dt(2))])
    assert p.contradictions == []


def test_contradictions_are_not_re_appended_on_re_ingest():
    p = Passport.new()
    p, _ = merge_passport_update(p, facts=[Fact(key="location", value="Harare",
                                                 source="a", ts=dt(1))])
    p, _ = merge_passport_update(p, facts=[Fact(key="location", value="Kwekwe",
                                                 source="b", ts=dt(2))])
    assert len(p.contradictions) == 1

    # Replaying the same change must not grow the list.
    for _ in range(5):
        p, _ = merge_passport_update(p, facts=[Fact(key="location", value="Kwekwe",
                                                     source="b", ts=dt(2))])
    assert len(p.contradictions) == 1


def test_real_contradictions_still_accumulate():
    p = Passport.new()
    for day, city in enumerate(["Harare", "Kwekwe", "Bulawayo"], start=1):
        p, _ = merge_passport_update(
            p, facts=[Fact(key="location", value=city, source=f"s{day}", ts=dt(day))]
        )
    assert len(p.contradictions) == 2
    assert p.facts[0].value == "Bulawayo"


# --------------------------------------------------------------------------
# Tier 3: semantic reconciliation
# --------------------------------------------------------------------------

def test_tier3_catches_what_string_distance_cannot():
    # The whitepaper's own example. Similarity is ~0.51, so no threshold
    # reaches it; only semantic reconciliation can.
    p = Passport.new()
    p.prefs.append(Preference(key="code_style", value="minimal", source="a", ts=dt(1)))
    p.prefs.append(Preference(key="boilerplate_preference", value="low",
                              source="b", ts=dt(2)))
    assert compare("code_style", "boilerplate_preference").verdict is Verdict.DISTINCT

    provider = ScriptedProvider({
        "merges": [{
            "canonical": "code_style",
            "aliases": ["boilerplate_preference"],
            "reason": "both record how terse the person wants generated code",
        }]
    })
    result = compact(p, provider)

    assert result.keys_removed == 1
    assert {pref.key for pref in p.prefs} == {"code_style"}
    assert p.prefs[0].value == "low", "the newer value wins"
    assert p.meta["dedupe"]["reconciled"]


def test_compaction_is_one_call_not_one_per_key():
    p = Passport.new()
    for i in range(40):
        p.facts.append(Fact(key=f"key_{i}", value=str(i), source="s", ts=dt(i)))
    provider = ScriptedProvider({"merges": []})
    compact(p, provider)
    assert provider.usage.calls == 1


def test_compaction_rejects_a_hallucinated_key():
    p = Passport.new()
    p.facts.append(Fact(key="location", value="Kwekwe", source="s"))
    p.facts.append(Fact(key="role", value="engineer", source="s"))
    provider = ScriptedProvider({
        "merges": [{"canonical": "invented_key", "aliases": ["location"], "reason": "x"}]
    })
    result = compact(p, provider)
    assert result.merged == []
    assert result.rejected
    assert {f.key for f in p.facts} == {"location", "role"}


def test_compaction_dry_run_changes_nothing():
    p = Passport.new()
    p.prefs.append(Preference(key="code_style", value="minimal", source="a"))
    p.prefs.append(Preference(key="boilerplate_preference", value="low", source="b"))
    provider = ScriptedProvider({
        "merges": [{"canonical": "code_style",
                    "aliases": ["boilerplate_preference"], "reason": "same"}]
    })
    result = compact(p, provider, apply=False)
    assert result.merged
    assert len(p.prefs) == 2, "dry run must not mutate the passport"


def test_compaction_forwards_deferred_pairs_as_hints():
    p = Passport.new()
    p.facts.append(Fact(key="work_location", value="Harare", source="s"))
    p.facts.append(Fact(key="work_locale", value="Harare", source="s"))
    p.meta["dedupe"] = {"pending_review": [
        {"incoming": "work_locale", "existing": "work_location", "score": 0.9}
    ]}
    provider = ScriptedProvider({"merges": []})
    compact(p, provider)
    assert "work_locale / work_location" in provider.prompts[0]


def test_compaction_on_an_empty_passport_makes_no_call():
    provider = ScriptedProvider({"merges": []})
    result = compact(Passport.new(), provider)
    assert provider.usage.calls == 0
    assert result.merged == []
