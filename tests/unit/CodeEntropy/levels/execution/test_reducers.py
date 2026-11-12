"""Unit tests for frame map-reduce reducers."""

from __future__ import annotations

import numpy as np

from CodeEntropy.levels.execution.reducers import (
    CovarianceReducer,
    NeighborReducer,
    incremental_mean,
    merge_means,
    stable_keys,
)
from CodeEntropy.levels.execution.tasks import CovarianceChunkPartial


def _shared_covariance_state() -> dict:
    return {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {
            "ua": {},
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "forcetorque_covariances": {"res": [None], "poly": [None]},
        "forcetorque_counts": {
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "group_id_to_index": {7: 0},
        "groups": {7: [0]},
    }


def test_stable_keys_orders_mixed_key_types_deterministically():
    mapping = {(2, 0): "tuple", 1: "int", "a": "str"}

    assert stable_keys(mapping) == [1, "a", (2, 0)]


def test_merge_means_returns_old_mean_when_new_count_is_zero():
    old = np.array([1.0, 2.0])

    assert merge_means(old, old_n=2, new_mean=np.array([9.0, 9.0]), new_n=0) is old


def test_merge_means_returns_copy_for_first_numpy_value():
    new = np.array([1.0, 2.0])

    merged = merge_means(None, old_n=0, new_mean=new, new_n=3)

    np.testing.assert_allclose(merged, new)
    new[0] = 99.0
    assert merged[0] != 99.0


def test_merge_means_combines_weighted_means():
    old = np.array([2.0, 4.0])
    new = np.array([8.0, 10.0])

    merged = merge_means(old, old_n=2, new_mean=new, new_n=1)

    np.testing.assert_allclose(merged, np.array([4.0, 6.0]))


def test_incremental_mean_returns_copy_for_first_numpy_value():
    new = np.array([1.0, 2.0])

    out = incremental_mean(None, new, n=1)

    np.testing.assert_allclose(out, new)
    new[0] = 99.0
    assert out[0] != 99.0


def test_incremental_mean_updates_mean():
    old = np.array([2.0, 2.0])
    new = np.array([4.0, 0.0])

    out = incremental_mean(old, new, n=2)

    np.testing.assert_allclose(out, np.array([3.0, 1.0]))


def test_neighbor_reducer_initialise_merge_and_finalise():
    shared_data = {"groups": {7: [0], 9: [1]}}

    NeighborReducer.initialise(shared_data)
    NeighborReducer.merge_chunk_partial(
        shared_data,
        neighbor_totals={7: 6, 9: 0},
        neighbor_samples={7: 3, 9: 0},
    )
    NeighborReducer.finalise(shared_data)

    assert shared_data["neighbor_totals"] == {7: 6, 9: 0}
    assert shared_data["neighbor_samples"] == {7: 3, 9: 0}
    assert shared_data["neighbors"] == {7: 2.0, 9: 0.0}


def test_neighbor_reducer_reduce_frame_output_none_is_noop():
    shared_data = {"groups": {7: [0]}}
    NeighborReducer.initialise(shared_data)

    NeighborReducer.reduce_frame_output(shared_data, None)

    assert shared_data["neighbor_totals"] == {7: 0}
    assert shared_data["neighbor_samples"] == {7: 0}


def test_neighbor_reducer_reduce_frame_output_merges_counts():
    shared_data = {"groups": {7: [0]}}
    NeighborReducer.initialise(shared_data)

    NeighborReducer.reduce_frame_output(shared_data, {7: (4, 2)})

    assert shared_data["neighbor_totals"] == {7: 4}
    assert shared_data["neighbor_samples"] == {7: 2}


def test_neighbor_reducer_merge_chunk_partial_noops_if_not_initialised():
    shared_data = {}

    NeighborReducer.merge_chunk_partial(
        shared_data,
        neighbor_totals={7: 1},
        neighbor_samples={7: 1},
    )

    assert shared_data == {}


def test_covariance_reducer_reduce_frame_map_output_merges_covariance_and_neighbors():
    shared_data = _shared_covariance_state()
    NeighborReducer.initialise(shared_data)

    force = np.eye(3)
    torque = 2.0 * np.eye(3)
    ft = np.ones((6, 6))

    frame_out = {
        "covariance": {
            "force": {"ua": {(7, 0): force}, "res": {7: force}, "poly": {}},
            "torque": {"ua": {(7, 0): torque}, "res": {7: torque}, "poly": {}},
            "forcetorque": {"res": {7: ft}, "poly": {}},
        },
        "neighbors": {7: (5, 1)},
    }

    CovarianceReducer().reduce_frame_map_output(shared_data, frame_out)

    np.testing.assert_allclose(shared_data["force_covariances"]["ua"][(7, 0)], force)
    np.testing.assert_allclose(shared_data["torque_covariances"]["ua"][(7, 0)], torque)
    np.testing.assert_allclose(shared_data["force_covariances"]["res"][0], force)
    np.testing.assert_allclose(shared_data["torque_covariances"]["res"][0], torque)
    np.testing.assert_allclose(shared_data["forcetorque_covariances"]["res"][0], ft)

    assert shared_data["frame_counts"]["ua"][(7, 0)] == 1
    assert shared_data["frame_counts"]["res"][0] == 1
    assert shared_data["forcetorque_counts"]["res"][0] == 1
    assert shared_data["neighbor_totals"] == {7: 5}
    assert shared_data["neighbor_samples"] == {7: 1}


def test_covariance_reducer_reduce_frame_map_output_accepts_missing_sections():
    shared_data = _shared_covariance_state()
    NeighborReducer.initialise(shared_data)

    CovarianceReducer().reduce_frame_map_output(shared_data, {})

    assert shared_data["frame_counts"]["res"][0] == 0
    assert shared_data["neighbor_totals"] == {7: 0}


def test_covariance_reducer_merge_chunk_partial():
    shared_data = _shared_covariance_state()

    force = np.eye(3)
    torque = 2.0 * np.eye(3)
    ft = np.ones((6, 6))

    partial = CovarianceChunkPartial()
    partial.force["ua"][(7, 0)] = force
    partial.torque["ua"][(7, 0)] = torque
    partial.frame_counts["ua"][(7, 0)] = 2
    partial.force["res"][7] = force
    partial.torque["res"][7] = torque
    partial.frame_counts["res"][7] = 2
    partial.force["poly"][7] = force
    partial.torque["poly"][7] = torque
    partial.frame_counts["poly"][7] = 2
    partial.forcetorque["res"][7] = ft
    partial.forcetorque_counts["res"][7] = 2
    partial.forcetorque["poly"][7] = ft
    partial.forcetorque_counts["poly"][7] = 2

    CovarianceReducer().merge_chunk_partial(shared_data, partial)

    np.testing.assert_allclose(shared_data["force_covariances"]["ua"][(7, 0)], force)
    np.testing.assert_allclose(shared_data["torque_covariances"]["ua"][(7, 0)], torque)
    np.testing.assert_allclose(shared_data["force_covariances"]["res"][0], force)
    np.testing.assert_allclose(shared_data["torque_covariances"]["res"][0], torque)
    np.testing.assert_allclose(shared_data["force_covariances"]["poly"][0], force)
    np.testing.assert_allclose(shared_data["torque_covariances"]["poly"][0], torque)
    np.testing.assert_allclose(shared_data["forcetorque_covariances"]["res"][0], ft)
    np.testing.assert_allclose(shared_data["forcetorque_covariances"]["poly"][0], ft)

    assert shared_data["frame_counts"]["ua"][(7, 0)] == 2
    assert shared_data["frame_counts"]["res"][0] == 2
    assert shared_data["frame_counts"]["poly"][0] == 2
    assert shared_data["forcetorque_counts"]["res"][0] == 2
    assert shared_data["forcetorque_counts"]["poly"][0] == 2


def test_covariance_reducer_reduce_frame_output_handles_torque_only_branches():
    shared_data = _shared_covariance_state()

    ua_torque = np.eye(3)
    res_torque = 2.0 * np.eye(3)
    poly_torque = 3.0 * np.eye(3)

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {
            "ua": {(7, 0): ua_torque},
            "res": {7: res_torque},
            "poly": {7: poly_torque},
        },
    }

    CovarianceReducer().reduce_frame_output(shared_data, frame_out)

    assert shared_data["frame_counts"]["ua"][(7, 0)] == 1
    assert shared_data["frame_counts"]["res"][0] == 1
    assert shared_data["frame_counts"]["poly"][0] == 1

    np.testing.assert_allclose(
        shared_data["torque_covariances"]["ua"][(7, 0)],
        ua_torque,
    )
    np.testing.assert_allclose(
        shared_data["torque_covariances"]["res"][0],
        res_torque,
    )
    np.testing.assert_allclose(
        shared_data["torque_covariances"]["poly"][0],
        poly_torque,
    )

    assert shared_data["force_covariances"]["ua"] == {}
    assert shared_data["force_covariances"]["res"][0] is None
    assert shared_data["force_covariances"]["poly"][0] is None


def test_covariance_reducer_reduce_frame_output_updates_poly_force_and_torque():
    shared_data = _shared_covariance_state()

    poly_force = np.eye(3)
    poly_torque = 2.0 * np.eye(3)

    frame_out = {
        "force": {
            "ua": {},
            "res": {},
            "poly": {7: poly_force},
        },
        "torque": {
            "ua": {},
            "res": {},
            "poly": {7: poly_torque},
        },
    }

    CovarianceReducer().reduce_frame_output(shared_data, frame_out)

    assert shared_data["frame_counts"]["poly"][0] == 1
    np.testing.assert_allclose(
        shared_data["force_covariances"]["poly"][0],
        poly_force,
    )
    np.testing.assert_allclose(
        shared_data["torque_covariances"]["poly"][0],
        poly_torque,
    )


def test_covariance_reducer_reduce_frame_output_without_forcetorque_is_noop_for_ft():
    shared_data = _shared_covariance_state()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
    }

    CovarianceReducer().reduce_frame_output(shared_data, frame_out)

    assert shared_data["forcetorque_counts"]["res"][0] == 0
    assert shared_data["forcetorque_counts"]["poly"][0] == 0
    assert shared_data["forcetorque_covariances"]["res"][0] is None
    assert shared_data["forcetorque_covariances"]["poly"][0] is None


def test_covariance_reducer_reduce_frame_output_updates_poly_forcetorque():
    shared_data = _shared_covariance_state()

    poly_ft = np.ones((6, 6))

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "forcetorque": {
            "res": {},
            "poly": {7: poly_ft},
        },
    }

    CovarianceReducer().reduce_frame_output(shared_data, frame_out)

    assert shared_data["forcetorque_counts"]["poly"][0] == 1
    np.testing.assert_allclose(
        shared_data["forcetorque_covariances"]["poly"][0],
        poly_ft,
    )

    assert shared_data["forcetorque_counts"]["res"][0] == 0
    assert shared_data["forcetorque_covariances"]["res"][0] is None
