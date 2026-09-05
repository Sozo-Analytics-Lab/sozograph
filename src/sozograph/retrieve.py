"""
Zero-dependency lexical ranking over passport records.

A vector database and an embedding model are the two heaviest things a memory
library can ask you to install, and for a few hundred short records they buy
very little. This is Okapi BM25 in plain Python over the passport's episodes:
microseconds to run, nothing to download, nothing to migrate, and it works
identically on a laptop, in a Lambda, and in a browser via Pyodide.

It ranks whatever a caller hands it: episodes, and now facts, preferences,
entities, loops, and contradictions against the query. Facts and preferences
are small enough to inject in full below the caps, so ranking only decides
what survives a cut.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
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


class BM25F:
    """Field-aware BM25 for structured memory records.

    Each field has its own weight and length normalization. For term ``t`` in
    document ``d`` the combined field frequency is::

        w_tf(t, d) = sum_f w_f * tf(t, d_f) /
                     (1 - b_f + b_f * len(d_f) / avg_len_f)

    The final contribution is::

        idf(t) * w_tf(t, d) * (k1 + 1) / (k1 + w_tf(t, d))

    This lets a match in a fact key carry more evidence than the same token in
    free text. The implementation stays deterministic and dependency free.
    """

    def __init__(
        self,
        documents: Sequence[Mapping[str, str]],
        *,
        weights: Mapping[str, float] | None = None,
        field_b: Mapping[str, float] | None = None,
        k1: float = 1.2,
    ):
        self.documents = list(documents)
        self.n = len(self.documents)
        fields = sorted({field for doc in self.documents for field in doc})
        self.weights = {field: float((weights or {}).get(field, 1.0)) for field in fields}
        self.field_b = {field: float((field_b or {}).get(field, B)) for field in fields}
        self.k1 = float(k1)

        self.tokens: dict[str, list[list[str]]] = {
            field: [tokenize(doc.get(field, "")) for doc in self.documents]
            for field in fields
        }
        self.avg_len: dict[str, float] = {}
        for field, rows in self.tokens.items():
            self.avg_len[field] = (sum(len(row) for row in rows) / self.n) if self.n else 0.0

        document_frequency: Counter = Counter()
        for index in range(self.n):
            terms: set[str] = set()
            for field in fields:
                terms.update(self.tokens[field][index])
            document_frequency.update(terms)
        self.idf = {
            term: max(0.0, math.log(1.0 + (self.n - count + 0.5) / (count + 0.5)))
            for term, count in document_frequency.items()
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
            for index in range(self.n):
                weighted_tf = 0.0
                for field, rows in self.tokens.items():
                    row = rows[index]
                    frequency = row.count(term)
                    if not frequency:
                        continue
                    average = self.avg_len[field]
                    length_ratio = (len(row) / average) if average else 0.0
                    norm = 1.0 - self.field_b[field] + self.field_b[field] * length_ratio
                    weighted_tf += self.weights[field] * frequency / max(norm, 1e-12)
                if weighted_tf:
                    scores[index] += (
                        idf * weighted_tf * (self.k1 + 1.0) / (self.k1 + weighted_tf)
                    )
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


_NAME_BOUND_RE_TMPL = r"(?<![a-z0-9]){}(?![a-z0-9])"


def match_names(query: str, vocabulary: Iterable[str]) -> list[str]:
    """
    Which known names appear in the query, on word boundaries.

    A list question ("What did Tim do this summer?") names its subject. Pure
    string matching over the passport's entity names, aliases, and record
    participants finds that subject with no model call and no dependency.
    Names shorter than two characters are ignored so stray initials cannot
    hijack the expansion.
    """
    q = (query or "").lower()
    if not q:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for name in vocabulary:
        n = (name or "").strip().lower()
        if len(n) < 2 or n in seen:
            continue
        seen.add(n)
        if re.search(_NAME_BOUND_RE_TMPL.format(re.escape(n)), q):
            found.append(n)
    return found


#: How much naming the query's subject lifts a record. Larger than any possible
#: BM25-plus-prior score (BM25 is normalized to 1.0, the prior adds at most
#: `prior_weight`), so every record about the named subject sorts above every
#: record that is not, while BM25 still orders within each group.
_ENTITY_MATCH_BONUS = 10.0


def rank_expanded(
    items: Sequence[Any],
    query: str | None,
    *,
    text_of: Callable[[Any], str],
    limit: int | None = None,
    prior: Callable[[Any], float] | None = None,
    vocabulary: Iterable[str] | None = None,
) -> list[Scored]:
    """
    BM25 ranking widened by named-entity recall.

    Lexical top-k alone answers "find the observation matching these words".
    A multi-hop list question needs the union of everything about its subject,
    including records sharing no distinctive word with the query ("What did
    Tim do this summer?" must also surface "Tim kayaked the river gorge",
    which says nothing about summers). When the query names a known entity,
    every record mentioning it is lifted above the records that do not, so the
    whole subject survives the cap cut. BM25 still orders within each group, so
    the widening changes which records survive `limit` without ever growing the
    rendered section.

    Lifting, not merely adding to a candidate set: ranking the union by score
    and cutting to `limit` would return the same top-`limit` as plain `rank`,
    because a low-BM25 record about the subject loses its slot to a higher-BM25
    record right back. The bonus is what makes the subject's records win the cut.
    """
    scored = rank(items, query, text_of=text_of, limit=None, prior=prior)
    if not scored or not query or not vocabulary:
        return scored[:limit] if limit is not None else scored

    names = match_names(query, vocabulary)
    if not names:
        return scored[:limit] if limit is not None else scored

    lifted: list[Scored] = []
    for s in scored:
        hit = any(n in text_of(items[s.index]).lower() for n in names)
        bonus = _ENTITY_MATCH_BONUS if hit else 0.0
        lifted.append(Scored(index=s.index, score=s.score + bonus, item=s.item))
    lifted.sort(key=lambda s: (-s.score, s.index))
    return lifted[:limit] if limit is not None else lifted


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
