from unittest.mock import MagicMock

import pytest

from CodeEntropy.entropy.graph import EntropyGraph, NodeSpec


def test_build_creates_expected_nodes_and_edges():
    g = EntropyGraph().build()

    assert set(g._nodes.keys()) == {
        "vibrational_entropy",
        "configurational_entropy",
        "aggregate_entropy",
    }

    assert g._graph.has_edge("vibrational_entropy", "aggregate_entropy")
    assert g._graph.has_edge("configurational_entropy", "aggregate_entropy")


def test_execute_runs_nodes_in_topological_order_and_merges_dict_outputs(shared_data):
    g = EntropyGraph()

    node_a = MagicMock()
    node_b = MagicMock()
    node_c = MagicMock()

    node_a.run.return_value = {"a": 1}
    node_b.run.return_value = {"b": 2}
    node_c.run.return_value = "not-a-dict"

    g._add_node(NodeSpec("a", node_a))
    g._add_node(NodeSpec("b", node_b))
    g._add_node(NodeSpec("c", node_c, deps=("a", "b")))

    out = g.execute(shared_data)

    assert node_a.run.called
    assert node_b.run.called
    assert node_c.run.called
    assert out == {"a": 1, "b": 2}


def test_add_node_duplicate_name_raises():
    g = EntropyGraph()
    g._add_node(NodeSpec("x", object()))
    with pytest.raises(ValueError):
        g._add_node(NodeSpec("x", object()))
