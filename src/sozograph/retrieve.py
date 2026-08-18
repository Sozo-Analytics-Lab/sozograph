"""
Zero-dependency lexical ranking over passport records.

A vector database and an embedding model are the two heaviest things a memory
library can ask you to install, and for a few hundred short records they buy
very little. This is Okapi BM25 in plain Python over the passport's episodes:
microseconds to run, nothing to download, nothing to migrate, and it works
identically on a laptop, in a Lambda, and in a browser via Pyodide.

It ranks episodes only. Facts and preferences are small enough to inject in
full, so a ranking miss costs episodic detail rather than a fact.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that match everything and therefore discriminate nothing.
_STOPWORDS = frozenset("""
a an and are as at be been being but by did do does for from had has have he her
hers him his how i if in into is it its me my of on or our ours she so that the
their theirs them then there these they this those to too was we were what when
where which who whom why will with you your yours
""".split())

K1 = 1.5   # term-frequency saturation
B = 0.75   # length normalization


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords dropped."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(str(text).lower())
    if keep_stopwords:
        return tokens
    kept = [t for t in tokens if t not in _STOPWORDS]
    # A query made entirely of stopwords ("who is he") would otherwise score
    # nothing at all; fall back to the raw tokens rather than returning empty.
    return kept or tokens


@dataclass
class Scored:
    index: int
    score: float
    item: Any


class BM25:
    """
    Okapi BM25 over a fixed set of short documents.

    Built per query in practice: a passport holds hundreds of records, not
    millions, so indexing cost is irrelevant and there is no stale index to
    invalidate when memory changes.
    """

    def __init__(self, documents: Sequence[str]):
        self.docs: list[list[str]] = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        # Add-one smoothed IDF, floored at zero so a term present in every
        # document contributes nothing rather than a negative score.
        self.idf: dict[str, float] = {
            term: max(0.0, math.log(1.0 + (self.n - count + 0.5) / (count + 0.5)))
            for term, count in df.items()
        }

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        if not terms or not self.n:
            return [0.0] * self.n

        scores = [0.0] * self.n
        for term in terms:
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, freq_map in enumerate(self.tf):
                freq = freq_map.get(term)
                if not freq:
                    continue
                norm = 1.0 - B + B * (self.lengths[i] / self.avg_len or 0.0)
                scores[i] += idf * (freq * (K1 + 1.0)) / (freq + K1 * norm)
        return scores


def rank(
    items: Sequence[Any],
    query: str | None,
    *,
    text_of: Callable[[Any], str],
    limit: int | None = None,
    prior: Callable[[Any], float] | None = None,
    prior_weight: float = 0.35,
) -> list[Scored]:
    """
    Rank items against a query, blending lexical relevance with a prior.

    With no query the prior alone decides the order, which is how a
    passport renders without one (recency and salience). With a query, the
    lexical score dominates and the prior breaks ties among equally relevant
    records so a recent match outranks a stale one.
    """
    if not items:
        return []

    priors = [float(prior(it)) if prior else 0.0 for it in items]
    lo, hi = (min(priors), max(priors)) if priors else (0.0, 0.0)
    span = hi - lo
    priors = [(p - lo) / span if span else 0.0 for p in priors]

    if query:
        raw = BM25([text_of(it) for it in items]).score(query)
        peak = max(raw) or 1.0
        combined = [r / peak + prior_weight * p for r, p in zip(raw, priors, strict=True)]
    else:
        combined = priors

    scored = [Scored(index=i, score=s, item=items[i]) for i, s in enumerate(combined)]
    scored.sort(key=lambda s: (-s.score, s.index))
    return scored[:limit] if limit is not None else scored


def top_items(
    items: Sequence[Any],
    query: str | None,
    *,
    text_of: Callable[[Any], str],
    limit: int,
    prior: Callable[[Any], float] | None = None,
) -> list[Any]:
    """Convenience wrapper returning the items themselves."""
    return [s.item for s in rank(items, query, text_of=text_of, limit=limit, prior=prior)]


def keywords_from(text: str, *, limit: int = 8) -> list[str]:
    """
    Pick the most distinctive tokens in a piece of text.

    Used to fill an episode's `keywords` when the extractor did not supply
    any, so the record is still findable.
    """
    counts = Counter(t for t in tokenize(text) if len(t) > 2)
    return [t for t, _ in counts.most_common(limit)]


def iter_texts(items: Iterable[Any], text_of: Callable[[Any], str]) -> list[tuple[int, str]]:
    return [(i, text_of(it)) for i, it in enumerate(items)]
