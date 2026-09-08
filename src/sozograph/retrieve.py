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
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")

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
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    tokens = []
    for token in _TOKEN_RE.findall(normalized):
        if _CJK_RE.search(token):
            tokens.extend(token)
            tokens.extend(token[i:i + 2] for i in range(len(token) - 1))
        else:
            tokens.append(token)
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


_NAME_BOUND_RE_TMPL = r"(?<![^\W_]){}(?![^\W_])"


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


def _clean_group(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for n in names or []:
        n2 = (n or "").strip().lower()
        if len(n2) < 2 or n2 in seen:
            continue
        seen.add(n2)
        out.append(n2)
    return out


@dataclass
class EntityGraph:
    """
    A co-occurrence graph over names mentioned together in the same record.

    Two names get an edge whenever one record names both of them (an
    observation's participants, an episode's participants); the edge weight is
    how often that happens. Built once from data the extractor already
    captures: no separate schema, no persisted index, nothing to keep in sync
    with the passport. This is the pure-Python form of the frontier
    literature's graph-propagation idea for multi-hop retrieval (HippoRAG's
    Personalized PageRank, Graphiti's typed-edge graph): who is connected to
    whom, with no graph database and no embedding model behind it.

    Stays schema-agnostic like the rest of this module: it takes groups of
    plain name strings, not passport records, so it needs no import from
    `schema` and is directly testable with lists of names.
    """

    adjacency: dict[str, Counter[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, groups: Iterable[Sequence[str]]) -> EntityGraph:
        adjacency: dict[str, Counter[str]] = {}
        for group in groups:
            names = _clean_group(group)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    adjacency.setdefault(a, Counter())[b] += 1
                    adjacency.setdefault(b, Counter())[a] += 1
        return cls(adjacency=adjacency)

    def neighbors(self, name: str, *, hops: int = 1, limit: int = 6) -> list[str]:
        """
        Names connected to `name` within `hops`, strongest edge first.

        Fan-out is capped so one heavily-discussed person cannot pull in an
        entire cast of characters; the strongest edges survive the cap.
        """
        start = (name or "").strip().lower()
        if hops < 1 or limit <= 0 or start not in self.adjacency:
            return []
        visited = {start}
        frontier = {start}
        collected: list[tuple[int, str]] = []
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in sorted(frontier):
                for neighbor, weight in self.adjacency.get(node, {}).items():
                    if neighbor in visited:
                        continue
                    collected.append((weight, neighbor))
                    next_frontier.add(neighbor)
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier
        collected.sort(key=lambda pair: (-pair[0], pair[1]))
        ordered: list[str] = []
        seen: set[str] = set()
        for _weight, n in collected:
            if n in seen:
                continue
            seen.add(n)
            ordered.append(n)
            if len(ordered) >= limit:
                break
        return ordered


#: How much naming the query's subject lifts a record. Larger than any possible
#: BM25-plus-prior score (BM25 is normalized to 1.0, the prior adds at most
#: `prior_weight`), so every record about the named subject sorts above every
#: record that is not, while BM25 still orders within each group.
_ENTITY_MATCH_BONUS = 10.0

#: Smaller than a direct match: a record about someone who co-occurs with the
#: query's subject is relevant but unconfirmed, so it can rank above ordinary
#: lexical matches without ever displacing an actual name hit.
_GRAPH_NEIGHBOR_BONUS = 4.0


def rank_expanded(
    items: Sequence[Any],
    query: str | None,
    *,
    text_of: Callable[[Any], str],
    limit: int | None = None,
    prior: Callable[[Any], float] | None = None,
    vocabulary: Iterable[str] | None = None,
    graph: EntityGraph | None = None,
    graph_hops: int = 1,
) -> list[Scored]:
    """
    BM25 ranking widened by named-entity recall, optionally by graph recall.

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

    When `graph` is given, a matched name's co-occurrence neighbors also get a
    smaller lift. "What does Melanie do with her family?" names Melanie
    directly; if Caroline co-occurs with Melanie across several observations,
    a record naming only Caroline surfaces too, even though "Caroline" never
    appears in the query. Direct name matches always outrank connected ones,
    so a spurious graph edge can compete with an ordinary lexical match but can
    never displace an actual name hit.
    """
    scored = rank(items, query, text_of=text_of, limit=None, prior=prior)
    if not scored or not query or not vocabulary:
        return scored[:limit] if limit is not None else scored

    names = match_names(query, vocabulary)
    if not names:
        return scored[:limit] if limit is not None else scored

    neighbor_names: set[str] = set()
    if graph is not None:
        for name in names:
            neighbor_names.update(graph.neighbors(name, hops=graph_hops))
        neighbor_names -= set(names)

    lifted: list[Scored] = []
    for s in scored:
        text = text_of(items[s.index]).lower()
        if match_names(text, names):
            bonus = _ENTITY_MATCH_BONUS
        elif neighbor_names and match_names(text, neighbor_names):
            bonus = _GRAPH_NEIGHBOR_BONUS
        else:
            bonus = 0.0
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
