from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.levels.nodes.conformations import ComputeConformationalStatesNode


def test_compute_conformational_states_node_runs_and_writes_shared_data():
    uops = MagicMock()
    node = ComputeConformationalStatesNode(universe_operations=uops)

    node._dihedral_analysis.build_conformational_states = MagicMock(
        return_value=({"ua_key": ["0", "1"]}, [["00", "01"]])
    )

    shared = {
        "reduced_universe": MagicMock(),
        "levels": {0: ["united_atom"]},
        "groups": {0: [0]},
        "start": 0,
        "end": 10,
        "step": 1,
        "args": SimpleNamespace(bin_width=10),
    }

    out = node.run(shared)

    assert "conformational_states" in out
    assert shared["conformational_states"]["ua"] == {"ua_key": ["0", "1"]}
    assert shared["conformational_states"]["res"] == [["00", "01"]]
    node._dihedral_analysis.build_conformational_states.assert_called_once()
