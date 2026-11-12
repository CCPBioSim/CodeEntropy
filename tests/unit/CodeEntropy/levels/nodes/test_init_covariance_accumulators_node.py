"""Unit tests for covariance accumulator initialisation."""

from __future__ import annotations

import numpy as np

from CodeEntropy.levels.nodes.accumulators import InitCovarianceAccumulatorsNode


def test_init_covariance_accumulators_allocates_canonical_accumulators():
    shared = {
        "groups": {10: [0], 20: [1]},
    }

    result = InitCovarianceAccumulatorsNode().run(shared)

    assert shared["group_id_to_index"] == {10: 0, 20: 1}
    assert shared["index_to_group_id"] == [10, 20]

    assert shared["force_covariances"]["ua"] == {}
    assert shared["torque_covariances"]["ua"] == {}

    assert len(shared["force_covariances"]["res"]) == 2
    assert len(shared["torque_covariances"]["res"]) == 2
    assert len(shared["force_covariances"]["poly"]) == 2
    assert len(shared["torque_covariances"]["poly"]) == 2

    np.testing.assert_array_equal(shared["frame_counts"]["res"], np.zeros(2, dtype=int))
    np.testing.assert_array_equal(
        shared["frame_counts"]["poly"], np.zeros(2, dtype=int)
    )

    assert len(shared["forcetorque_covariances"]["res"]) == 2
    assert len(shared["forcetorque_covariances"]["poly"]) == 2
    np.testing.assert_array_equal(
        shared["forcetorque_counts"]["res"],
        np.zeros(2, dtype=int),
    )
    np.testing.assert_array_equal(
        shared["forcetorque_counts"]["poly"],
        np.zeros(2, dtype=int),
    )

    assert "force_torque_stats" not in shared
    assert "force_torque_counts" not in shared

    assert result["force_covariances"] is shared["force_covariances"]
    assert result["torque_covariances"] is shared["torque_covariances"]
    assert result["frame_counts"] is shared["frame_counts"]
    assert result["forcetorque_covariances"] is shared["forcetorque_covariances"]
    assert result["forcetorque_counts"] is shared["forcetorque_counts"]


def test_init_covariance_accumulators_requires_groups():
    import pytest

    with pytest.raises(KeyError):
        InitCovarianceAccumulatorsNode().run({})
