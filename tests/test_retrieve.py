from __future__ import annotations

from sozograph.retrieve import EntityGraph, Scored, match_names, rank_expanded


def dummy(text: str, index: int) -> Scored:
    return Scored(index=index, score=0.0, item=text)


# --------------------------------------------------------------------------
# EntityGraph
# --------------------------------------------------------------------------

def test_graph_builds_edges_from_co_occurring_groups():
    graph = EntityGraph.build([["Melanie", "Caroline"], ["Melanie", "Caroline"], ["Tim"]])
    assert graph.neighbors("melanie") == ["caroline"]
    assert graph.neighbors("caroline") == ["melanie"]
    # A group of one names nobody: there is no pair to connect.
    assert graph.neighbors("tim") == []


def test_graph_neighbors_ranked_by_edge_weight():
    graph = EntityGraph.build([
        ["Melanie", "Caroline"],
        ["Melanie", "Caroline"],
        ["Melanie", "Caroline"],
        ["Melanie", "Bailey"],
    ])
    # Caroline co-occurs with Melanie three times, Bailey once: strongest first.
    assert graph.neighbors("melanie", limit=5) == ["caroline", "bailey"]
    assert graph.neighbors("melanie", limit=1) == ["caroline"]


def test_graph_two_hop_reaches_a_friend_of_a_friend():
    graph = EntityGraph.build([["Melanie", "Caroline"], ["Caroline", "Bailey"]])
    # Melanie and Bailey never co-occur directly.
    assert graph.neighbors("melanie", hops=1) == ["caroline"]
    assert set(graph.neighbors("melanie", hops=2)) == {"caroline", "bailey"}


def test_graph_unknown_name_returns_empty():
    graph = EntityGraph.build([["Melanie", "Caroline"]])
    assert graph.neighbors("stranger") == []
    assert graph.neighbors("") == []


def test_graph_ignores_short_and_blank_names():
    graph = EntityGraph.build([["Melanie", "A", "", "  ", "Caroline"]])
    assert graph.neighbors("melanie") == ["caroline"]
    assert graph.neighbors("a") == []


def test_graph_from_empty_groups_is_empty():
    graph = EntityGraph.build([])
    assert graph.adjacency == {}
    assert graph.neighbors("anyone") == []


# --------------------------------------------------------------------------
# rank_expanded with a graph
# --------------------------------------------------------------------------

def test_rank_expanded_lifts_a_graph_neighbor_above_plain_lexical_matches():
    """A record about Caroline alone must outrank unrelated noise once the
    query names Melanie and the graph knows Caroline co-occurs with her,
    even though "Caroline" never appears in the query text."""
    items = [
        "Caroline planted a rainbow of tulips in the garden",  # graph neighbor of Melanie
        "the weather was mild and pleasant that afternoon",     # unrelated, no overlap at all
        "someone mentioned tulips in passing once",             # off-topic noise sharing a word
    ]
    graph = EntityGraph.build([["Melanie", "Caroline"], ["Melanie", "Caroline"]])
    ranked = rank_expanded(
        items, "What did Melanie do with her family?",
        text_of=lambda t: t, vocabulary=["Melanie", "Caroline"], graph=graph,
    )
    assert ranked[0].item == items[0]


def test_rank_expanded_direct_match_still_beats_graph_neighbor():
    items = [
        "Caroline planted tulips",       # graph neighbor only
        "Melanie planted tulips too",    # direct match
    ]
    graph = EntityGraph.build([["Melanie", "Caroline"]])
    ranked = rank_expanded(
        items, "What did Melanie plant?",
        text_of=lambda t: t, vocabulary=["Melanie", "Caroline"], graph=graph,
    )
    assert ranked[0].item == items[1]


def test_rank_expanded_without_graph_matches_pre_graph_behaviour():
    """graph=None (the default) must be byte-for-byte the old rank_expanded."""
    items = ["Tim kayaked the river gorge", "unrelated chatter about the weather"]
    vocab = ["Tim"]
    with_none = rank_expanded(items, "What did Tim do?", text_of=lambda t: t, vocabulary=vocab)
    without_arg = rank_expanded(
        items, "What did Tim do?", text_of=lambda t: t, vocabulary=vocab, graph=None
    )
    assert [s.item for s in with_none] == [s.item for s in without_arg]
    assert with_none[0].item == items[0]


def test_rank_expanded_graph_neighbor_never_added_without_a_direct_match():
    """No name from the query appears at all: the graph must contribute nothing,
    matching plain rank() exactly (no query-less entity to expand from)."""
    items = ["Caroline planted tulips", "generic unrelated text"]
    graph = EntityGraph.build([["Melanie", "Caroline"]])
    ranked = rank_expanded(
        items, "How is the weather today?",
        text_of=lambda t: t, vocabulary=["Melanie", "Caroline"], graph=graph,
    )
    # Neither item names Melanie/Caroline in the query itself, so no bonus applies
    # and pure BM25+prior (all zero here) decides; order is stable by index.
    assert [s.index for s in ranked] == [0, 1]


def test_match_names_still_used_directly_for_direct_hits():
    assert match_names("What did Tim do?", ["Tim", "Melanie"]) == ["tim"]
