from unittest.mock import patch

from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode


def test_detect_levels_node_stores_results(reduced_universe):
    node = DetectLevelsNode()
    shared = {"reduced_universe": reduced_universe}

    with patch.object(
        node._hierarchy,
        "select_levels",
        return_value=(2, [["united_atom"], ["united_atom", "residue"]]),
    ):
        out = node.run(shared)

    assert shared["number_molecules"] == 2
    assert shared["levels"] == [["united_atom"], ["united_atom", "residue"]]
    assert out["levels"] == shared["levels"]
