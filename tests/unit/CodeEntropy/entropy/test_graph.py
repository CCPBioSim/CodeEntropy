from unittest.mock import MagicMock

import pytest

from CodeEntropy.entropy.graph import EntropyGraph, NodeSpec


def test_build_creates_expected_nodes_and_edges():
    g = EntropyGraph().build()

    assert set(g._nodes.keys()) == {
        "vibrational_entropy",
        "configurational_entropy",
        "orientational_entropy",
        "aggregate_entropy",
    }

    assert g._graph.has_edge("vibrational_entropy", "aggregate_entropy")
    assert g._graph.has_edge("configurational_entropy", "aggregate_entropy")
    assert g._graph.has_edge("orientational_entropy", "aggregate_entropy")


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


def test_execute_forwards_progress_to_nodes_that_accept_it(shared_data):
    g = EntropyGraph()

    node_a = MagicMock()
    node_a.run.return_value = {"a": 1}

    g._add_node(NodeSpec("a", node_a))

    progress = MagicMock()
    out = g.execute(shared_data, progress=progress)

    node_a.run.assert_called_once_with(shared_data, progress=progress)
    assert out == {"a": 1}


def test_execute_falls_back_when_node_run_does_not_accept_progress(shared_data):
    g = EntropyGraph()

    class NoProgressNode:
        def run(self, shared_data):
            return {"x": 1}

    node = NoProgressNode()
    g._add_node(NodeSpec("x", node))

    progress = MagicMock()
    out = g.execute(shared_data, progress=progress)

    assert out == {"x": 1}
