from unittest.mock import MagicMock

from CodeEntropy.entropy.nodes.orientational import OrientationalEntropyNode


def test_config_node_run_writes_results(shared_data):
    node = OrientationalEntropyNode()

    shared_data["levels"] = {0: ["united_atom", "residue"]}
    shared_data["groups"] = {0: [0]}
    shared_data["neighbors"] = {0: 0}
    shared_data["symmetry_number"] = {0: 0}
    shared_data["linear"] = {0: False}
    shared_data["hbond_factor"] = {0: 1}

    out = node.run(shared_data)

    assert "orientational_entropy" in out
    assert "orientational_entropy" in shared_data
    assert 0 in shared_data["orientational_entropy"]


def test_run_skips_empty_mol_ids_group():
    node = OrientationalEntropyNode()

    shared_data = {
        "n_frames": 5,
        "groups": {0: []},
        "levels": {0: ["united_atom"]},
        "reduced_universe": MagicMock(),
        "neighbors": {0: 0},
        "symmetry_number": {0: 0},
        "linear": {0: False},
        "reporter": None,
        "hbond_factor": {0: 0},
    }

    out = node.run(shared_data)

    assert "orientational_entropy" in out
    assert out["orientational_entropy"][0] == 0.0
