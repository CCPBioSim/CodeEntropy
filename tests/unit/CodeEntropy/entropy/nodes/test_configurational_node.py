from unittest.mock import MagicMock

import numpy as np
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


def test_run_skips_empty_mol_ids_group():
    node = ConfigurationalEntropyNode()

    shared_data = {
        "n_frames": 5,
        "groups": {0: []},
        "levels": {0: ["united_atom"]},
        "reduced_universe": MagicMock(),
        "conformational_states": {"ua": {}, "res": {}},
        "reporter": None,
    }

    out = node.run(shared_data)

    assert "configurational_entropy" in out
    assert out["configurational_entropy"][0]["ua"] == 0.0


def test_get_group_states_sequence_in_range_returns_value():
    states_res = [None, [1, 2, 3]]
    out = ConfigurationalEntropyNode._get_group_states(states_res, group_id=1)
    assert out == [1, 2, 3]


def test_get_group_states_sequence_out_of_range_returns_none():
    states_res = [None]
    out = ConfigurationalEntropyNode._get_group_states(states_res, group_id=2)
    assert out is None


def test_has_state_data_numpy_array_uses_np_any_branch():
    assert ConfigurationalEntropyNode._has_state_data(np.array([0, 0, 1])) is True
    assert ConfigurationalEntropyNode._has_state_data(np.array([0, 0, 0])) is False


def test_has_state_data_noniterable_hits_typeerror_fallback():
    # any(0) raises TypeError -> returns bool(0) == False
    assert ConfigurationalEntropyNode._has_state_data(0) is False
    # any(1) raises TypeError -> returns bool(1) == True
    assert ConfigurationalEntropyNode._has_state_data(1) is True
