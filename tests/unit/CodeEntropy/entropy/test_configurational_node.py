import pytest

from CodeEntropy.entropy.nodes.configurational import ConfigurationalEntropyNode


def test_config_node_raises_if_frame_count_missing():
    node = ConfigurationalEntropyNode()
    with pytest.raises(KeyError):
        node._get_n_frames({})


def test_config_node_run_writes_results(shared_data):
    node = ConfigurationalEntropyNode()

    shared_data["conformational_states"] = {
        "ua": {(0, 0): [0, 0, 1, 1]},
        "res": {0: [0, 1, 1, 1]},
    }

    shared_data["levels"] = {0: ["united_atom", "residue"]}
    shared_data["groups"] = {0: [0]}

    out = node.run(shared_data)

    assert "configurational_entropy" in out
    assert "configurational_entropy" in shared_data
    assert 0 in shared_data["configurational_entropy"]
