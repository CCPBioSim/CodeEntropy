from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.levels.nodes.find_neighbors import ComputeNeighborsNode


def test_compute_find_neighbors_node_runs_and_writes_shared_data():
    node = ComputeNeighborsNode()

    node._neighbor_analysis.get_neighbors = MagicMock(return_value=({0: 7.8}))

    node._neighbor_analysis.get_symmetry = MagicMock(return_value=({0: 2}, {0: False}))

    node._neighbor_analysis.get_bias = MagicMock(return_value=({0: 1}, {0: 1}))

    shared = {
        "reduced_universe": MagicMock(),
        "levels": {0: ["united_atom"]},
        "groups": {0: [0]},
        "start": 0,
        "end": 10,
        "step": 1,
        "args": SimpleNamespace(search_type="RAD"),
    }

    out = node.run(shared)

    assert "neighbors" in out
    assert "symmetry_number" in out
    assert "linear" in out
    assert shared["neighbors"] == {0: 7.8}
    assert shared["symmetry_number"] == {0: 2}
    assert shared["linear"] == {0: False}
    assert shared["hbond_bias"] == {0: 1}
    assert shared["n_factors"] == {0: 1}
    node._neighbor_analysis.get_neighbors.assert_called_once()
    node._neighbor_analysis.get_symmetry.assert_called_once()
    node._neighbor_analysis.get_bias.assert_called_once()
