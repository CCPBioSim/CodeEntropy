from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.levels.nodes.find_neighbors import ComputeNeighborsNode


def test_compute_find_neighbors_node_runs_and_writes_shared_data():
    node = ComputeNeighborsNode()

    frame_source = MagicMock()

    node._neighbor_analysis.get_neighbors = MagicMock(return_value={0: 7.8})
    node._neighbor_analysis.get_symmetry = MagicMock(return_value=({0: 2}, {0: False}))

    shared = {
        "reduced_universe": MagicMock(),
        "levels": {0: ["united_atom"]},
        "groups": {0: [0]},
        "frame_source": frame_source,
        "args": SimpleNamespace(search_type="RAD"),
    }

    out = node.run(shared)

    assert out is shared
    assert shared["neighbors"] == {0: 7.8}
    assert shared["symmetry_number"] == {0: 2}
    assert shared["linear"] == {0: False}

    node._neighbor_analysis.get_neighbors.assert_called_once_with(
        universe=shared["reduced_universe"],
        levels=shared["levels"],
        groups=shared["groups"],
        frame_source=frame_source,
        search_type="RAD",
    )
    node._neighbor_analysis.get_symmetry.assert_called_once_with(
        universe=shared["reduced_universe"],
        groups=shared["groups"],
    )
