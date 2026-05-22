from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.levels.nodes.conformations import ComputeConformationalStatesNode
from CodeEntropy.trajectory.frames import FrameSelection


def test_compute_conformational_states_node_runs_and_writes_shared_data():
    uops = MagicMock()
    node = ComputeConformationalStatesNode(universe_operations=uops)

    frame_selection = FrameSelection.from_bounds(start=0, stop=10, step=1)

    node._dihedral_analysis.build_conformational_states = MagicMock(
        return_value=(
            {"ua_key": ["0", "1"]},
            [["00", "01"]],
            {"ua_key": [0]},
            [0],
        )
    )

    shared = {
        "reduced_universe": MagicMock(),
        "levels": {0: ["united_atom"]},
        "groups": {0: [0]},
        "frame_selection": frame_selection,
        "args": SimpleNamespace(bin_width=10),
    }

    out = node.run(shared)

    assert out == {
        "conformational_states": {
            "ua": {"ua_key": ["0", "1"]},
            "res": [["00", "01"]],
        }
    }

    assert shared["conformational_states"] == {
        "ua": {"ua_key": ["0", "1"]},
        "res": [["00", "01"]],
    }
    assert shared["flexible_dihedrals"] == {
        "ua": {"ua_key": [0]},
        "res": [0],
    }

    node._dihedral_analysis.build_conformational_states.assert_called_once_with(
        data_container=shared["reduced_universe"],
        levels=shared["levels"],
        groups=shared["groups"],
        bin_width=10,
        frame_selection=frame_selection,
        progress=None,
    )
