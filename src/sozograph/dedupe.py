"""
Key deduplication: exact, then fuzzy, then deferred to semantic reconciliation.

The problem is real. A person who says "I prefer concise syntax" in one session
and "I dislike verbose boilerplate" fifty sessions later has stated one belief,
and an unconstrained extractor records it twice under two keys. Enough of that
and the passport reacquires exactly the entropy it exists to remove.

The obvious fix is dangerous. "Merge when string similarity exceeds 0.85" reads
sensibly and is wrong in a way that destroys data:

    budget_min   vs budget_max     -> 0.93
    likes_coffee vs likes_toffee   -> 0.94
    is_admin     vs is_admin_at    -> 0.96

Those are opposites, near-opposites, and different scopes. A false merge
silently overwrites a real belief and there is no undo, which is strictly worse
than carrying a duplicate key. So Tier 2 here is a *candidate generator*, not a
decision maker: it auto-merges only under a conjunction of evidence, refuses
outright on a polarity conflict, and hands the uncertain middle band to Tier 3.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .utils import normalize_key

# Score at or above which two keys may merge without asking a model.
AUTO_MERGE = 0.95
# Below this, two keys are simply different.
REVIEW_FLOOR = 0.82


class Verdict(str, Enum):
    """What Tier 2 concluded about a pair of keys."""

    EXACT = "exact"        # identical after normalization
    MERGE = "merge"        # confident enough to merge locally
    REVIEW = "review"      # plausible; defer to semantic reconciliation
    DISTINCT = "distinct"  # different keys
    BLOCKED = "blocked"    # similar strings, opposing meaning; never merge


@dataclass
class Match:
    incoming: str
    existing: str | None
    score: float
    verdict: Verdict
    reason: str = ""

    @property
    def is_merge(self) -> bool:
        return self.verdict in (Verdict.EXACT, Verdict.MERGE)


# --------------------------------------------------------------------------
# String similarity
# --------------------------------------------------------------------------

def jaro(a: str, b: str) -> float:
    """Jaro similarity. 1.0 identical, 0.0 nothing in common."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0

    reach = max(la, lb) // 2 - 1
    if reach < 0:
        reach = 0

    a_hit = [False] * la
    b_hit = [False] * lb
    matches = 0
    for i, ch in enumerate(a):
        for j in range(max(0, i - reach), min(lb, i + reach + 1)):
            if not b_hit[j] and b[j] == ch:
                a_hit[i] = b_hit[j] = True
                matches += 1
                break
    if not matches:
        return 0.0

    transpositions = 0
    j = 0
    for i in range(la):
        if not a_hit[i]:
            continue
        while not b_hit[j]:
            j += 1
        if a[i] != b[j]:
            transpositions += 1
        j += 1
    transpositions //= 2

    return (matches / la + matches / lb + (matches - transpositions) / matches) / 3.0


def jaro_winkler(a: str, b: str, *, prefix_weight: float = 0.1) -> float:
    """
    Jaro with a bonus for a shared prefix.

    Suited to key names, which are short and usually share a leading token
    ("code_style" / "code_styling"). The prefix bonus is also why the polarity
    guard below is not optional: it inflates exactly the pairs that differ only
    in their tail, which is where opposites live.
    """
    score = jaro(a, b)
    if score < 0.7:
        return score
    prefix = 0
    for ca, cb in zip(a[:4], b[:4], strict=False):
        if ca != cb:
            break
        prefix += 1
    return score + prefix * prefix_weight * (1.0 - score)


def token_set(key: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", normalize_key(key)) if t}


def token_similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two keys' token sets."""
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --------------------------------------------------------------------------
# Polarity guard
# --------------------------------------------------------------------------

#: Token pairs that make two otherwise-similar keys mean opposite things.
OPPOSING_TOKENS: tuple[tuple[str, str], ...] = (
    ("min", "max"), ("minimum", "maximum"),
    ("low", "high"), ("lower", "upper"), ("least", "most"),
    ("start", "end"), ("first", "last"), ("before", "after"),
    ("begin", "finish"), ("open", "closed"), ("on", "off"),
    ("enable", "disable"), ("enabled", "disabled"),
    ("allow", "deny"), ("allowed", "blocked"), ("include", "exclude"),
    ("likes", "dislikes"), ("like", "dislike"), ("love", "hate"),
    ("want", "avoid"), ("prefers", "avoids"),
    ("true", "false"), ("yes", "no"), ("more", "less"),
    ("increase", "decrease"), ("add", "remove"), ("in", "out"),
    ("buy", "sell"), ("input", "output"), ("source", "target"),
    ("from", "to"), ("sender", "recipient"), ("parent", "child"),
    ("primary", "secondary"), ("old", "new"), ("previous", "next"),
)

_NEGATORS = ("no", "not", "non", "un", "never", "without", "anti", "dis")

_OPPOSITES: dict[str, set[str]] = {}
for _x, _y in OPPOSING_TOKENS:
    _OPPOSITES.setdefault(_x, set()).add(_y)
    _OPPOSITES.setdefault(_y, set()).add(_x)


def polarity_conflict(a: str, b: str) -> str | None:
    """
    Return a reason string when two keys look similar but mean opposite things.

    Compares only the tokens that differ, so `budget_min` / `budget_max` is
    caught while `budget_min` / `min_budget` is not.
    """
    ta, tb = token_set(a), token_set(b)
    only_a, only_b = ta - tb, tb - ta

    for token in only_a:
        for other in _OPPOSITES.get(token, ()):
            if other in only_b:
                return f"opposing tokens {token!r} / {other!r}"

    # A negation flips meaning without changing much of the string. Two shapes:
    # a bare negator token on one side only ("has_access" / "has_no_access"),
    # and a negated variant of a token the other side has plain
    # ("available" / "unavailable").
    for left, right in ((only_a, only_b), (only_b, only_a)):
        for token in left:
            if token in _NEGATORS:
                return f"bare negation {token!r}"
            for neg in _NEGATORS:
                if token.startswith(neg) and len(token) > len(neg):
                    if token[len(neg):].lstrip("_") in right:
                        return f"negation {token!r}"

    # A single differing token that is purely numeric is an index, not a synonym.
    if len(only_a) == 1 and len(only_b) == 1:
        (x,), (y,) = tuple(only_a), tuple(only_b)
        if x.isdigit() and y.isdigit() and x != y:
            return f"differing indices {x!r} / {y!r}"

    return None


# --------------------------------------------------------------------------
# The tiered matcher
# --------------------------------------------------------------------------

def _variants_only(a: str, b: str, *, min_shared_prefix: int = 3) -> bool:
    """
    True when every token that differs is a morphological variant of one on the
    other side, rather than a different word that happens to look similar.

    "code_style" / "code_styling" differ by a suffix and are the same idea.
    "likes_coffee" / "likes_toffee" score 0.967 under Jaro-Winkler and are two
    unrelated preferences; the differing tokens share no prefix at all. A raw
    similarity threshold cannot tell those apart, so this does.
    """
    only_a = token_set(a) - token_set(b)
    only_b = token_set(b) - token_set(a)
    if not only_a and not only_b:
        return True
    if not only_a or not only_b:
        # One side simply carries an extra qualifier. That narrows scope
        # ("budget" vs "budget_ceiling"), so it is not a safe auto-merge.
        return False

    for token in only_a:
        if not any(_shared_prefix(token, other) >= min_shared_prefix for other in only_b):
            return False
    return True


def _shared_prefix(a: str, b: str) -> int:
    count = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        count += 1
    return count


def compare(incoming: str, existing: str) -> Match:
    """Classify one candidate pair."""
    a, b = normalize_key(incoming), normalize_key(existing)
    if a == b:
        return Match(incoming, existing, 1.0, Verdict.EXACT, "identical after normalization")

    conflict = polarity_conflict(a, b)
    score = jaro_winkler(a, b)
    tokens = token_similarity(a, b)

    if conflict:
        # Refuse regardless of score. A false merge here is unrecoverable.
        return Match(incoming, existing, score, Verdict.BLOCKED, conflict)

    if score < REVIEW_FLOOR and tokens < 1.0:
        return Match(incoming, existing, score, Verdict.DISTINCT, "below review floor")

    # Auto-merge needs a conjunction, not a single threshold: a very high string
    # score AND the same token multiset (a pure reordering or a punctuation
    # difference), or an exact token-set match.
    if tokens >= 1.0:
        return Match(incoming, existing, max(score, 0.99), Verdict.MERGE,
                     "identical token set")
    if score >= AUTO_MERGE and token_set(a) & token_set(b) and _variants_only(a, b):
        return Match(incoming, existing, score, Verdict.MERGE,
                     f"string similarity {score:.3f} with morphological variants")

    return Match(incoming, existing, score, Verdict.REVIEW,
                 f"string similarity {score:.3f} needs semantic review")


def find_match(incoming: str, existing_keys: Iterable[str]) -> Match:
    """
    Match one incoming key against the passport's current vocabulary.

    Tier 1 (exact) short-circuits. Otherwise the best-scoring candidate is
    classified, with any polarity-blocked pair remembered so a blocked
    near-neighbour never gets quietly merged via a lower-scoring alternative.
    """
    keys = list(existing_keys)
    normalized = normalize_key(incoming)

    for key in keys:
        if normalize_key(key) == normalized:
            return Match(incoming, key, 1.0, Verdict.EXACT, "exact key hit")

    best: Match | None = None
    blocked: Match | None = None
    for key in keys:
        match = compare(incoming, key)
        if match.verdict is Verdict.BLOCKED:
            if blocked is None or match.score > blocked.score:
                blocked = match
            continue
        if match.verdict is Verdict.DISTINCT:
            continue
        if best is None or match.score > best.score:
            best = match

    if best is not None:
        if blocked is not None and blocked.score > best.score:
            return blocked
        return best
    if blocked is not None:
        return blocked
    return Match(incoming, None, 0.0, Verdict.DISTINCT, "no candidate")


@dataclass
class DedupeReport:
    """What the local tiers did, so a merge is auditable after the fact."""

    merged: list[Match] = field(default_factory=list)
    review: list[Match] = field(default_factory=list)
    blocked: list[Match] = field(default_factory=list)

    def record(self, match: Match) -> None:
        if match.verdict is Verdict.MERGE:
            self.merged.append(match)
        elif match.verdict is Verdict.REVIEW:
            self.review.append(match)
        elif match.verdict is Verdict.BLOCKED:
            self.blocked.append(match)

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        def rows(items: Sequence[Match]):
            return [
                {"incoming": m.incoming, "existing": m.existing,
                 "score": round(m.score, 4), "reason": m.reason}
                for m in items
            ]

        out: dict[str, list[dict[str, object]]] = {}
        if self.merged:
            out["merged"] = rows(self.merged)
        if self.review:
            out["pending_review"] = rows(self.review)
        if self.blocked:
            out["blocked"] = rows(self.blocked)
        return out

    def __bool__(self) -> bool:
        return bool(self.merged or self.review or self.blocked)
