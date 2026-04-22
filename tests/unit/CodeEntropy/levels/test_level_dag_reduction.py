from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.levels.level_dag import LevelDAG


def test_incremental_mean_none_returns_copy_for_numpy():
    arr = np.array([1.0, 2.0])
    out = LevelDAG._incremental_mean(None, arr, n=1)
    assert np.all(out == arr)
    arr[0] = 999.0
    assert out[0] != 999.0


def test_incremental_mean_updates_mean_correctly():
    old = np.array([2.0, 2.0])
    new = np.array([4.0, 0.0])
    out = LevelDAG._incremental_mean(old, new, n=2)
    np.testing.assert_allclose(out, np.array([3.0, 1.0]))


def test_reduce_force_and_torque_updates_counts_and_means():
    dag = LevelDAG()

    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {
            "ua": {},
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "group_id_to_index": {9: 0},
    }

    F1 = np.eye(3)
    T1 = 2.0 * np.eye(3)

    frame_out = {
        "force": {"ua": {(0, 0): F1}, "res": {9: F1}, "poly": {}},
        "torque": {"ua": {(0, 0): T1}, "res": {9: T1}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["ua"][(0, 0)] == 1
    np.testing.assert_allclose(shared["force_covariances"]["ua"][(0, 0)], F1)
    np.testing.assert_allclose(shared["torque_covariances"]["ua"][(0, 0)], T1)

    assert shared["frame_counts"]["res"][0] == 1
    np.testing.assert_allclose(shared["force_covariances"]["res"][0], F1)
    np.testing.assert_allclose(shared["torque_covariances"]["res"][0], T1)


def test_reduce_forcetorque_no_key_is_noop():
    dag = LevelDAG()
    shared = {
        "forcetorque_covariances": {"res": [None], "poly": [None]},
        "forcetorque_counts": {
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "group_id_to_index": {9: 0},
    }

    dag._reduce_forcetorque(shared, frame_out={})
    assert shared["forcetorque_counts"]["res"][0] == 0
    assert shared["forcetorque_covariances"]["res"][0] is None


def test_run_frame_stage_calls_execute_frame_for_each_ts(simple_ts_list):
    dag = LevelDAG()

    u = MagicMock()
    u.trajectory = simple_ts_list

    shared = {"reduced_universe": u, "start": 0, "end": 10, "step": 1, "n_frames": 10}

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.side_effect = lambda shared_data, frame_index: {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_one_frame = MagicMock()

    dag._run_frame_stage(shared)

    assert dag._frame_dag.execute_frame.call_count == 10
    assert dag._reduce_one_frame.call_count == 10
