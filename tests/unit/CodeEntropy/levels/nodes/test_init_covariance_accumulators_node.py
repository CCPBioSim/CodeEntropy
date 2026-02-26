import numpy as np

from CodeEntropy.levels.nodes.accumulators import InitCovarianceAccumulatorsNode


def test_init_covariance_accumulators_allocates_and_sets_aliases():
    node = InitCovarianceAccumulatorsNode()

    shared = {"groups": {9: [1, 2], 2: [3]}}

    out = node.run(shared)

    assert out["group_id_to_index"] == {9: 0, 2: 1}
    assert out["index_to_group_id"] == [9, 2]

    assert shared["force_covariances"]["res"] == [None, None]
    assert shared["torque_covariances"]["poly"] == [None, None]

    assert np.all(shared["frame_counts"]["res"] == np.array([0, 0]))
    assert np.all(shared["forcetorque_counts"]["poly"] == np.array([0, 0]))

    assert shared["force_torque_stats"] is shared["forcetorque_covariances"]
    assert shared["force_torque_counts"] is shared["forcetorque_counts"]
