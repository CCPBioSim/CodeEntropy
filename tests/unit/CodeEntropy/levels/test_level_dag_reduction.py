from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.levels.level_dag import LevelDAG


def test_reduce_force_and_torque_updates_counts_and_means():
    dag = LevelDAG()

    shared = {
        "force_sums": {"ua": {}, "res": [None], "poly": [None]},
        "torque_sums": {"ua": {}, "res": [None], "poly": [None]},
        "force_counts": {
            "ua": {},
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "torque_counts": {
            "ua": {},
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "forcetorque_sums": {"res": [None], "poly": [None]},
        "forcetorque_counts": {
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "force_covariances": {},
        "torque_covariances": {},
        "forcetorque_covariances": {},
        "group_id_to_index": {9: 0},
    }

    F1 = np.eye(3)
    T1 = 2.0 * np.eye(3)

    frame_out = {
        "force": {"ua": {(0, 0): F1}, "res": {9: F1}, "poly": {}},
        "torque": {"ua": {(0, 0): T1}, "res": {9: T1}, "poly": {}},
        "force_counts": {"ua": {(0, 0): 1}, "res": {9: 1}, "poly": {}},
        "torque_counts": {"ua": {(0, 0): 1}, "res": {9: 1}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_counts"]["ua"][(0, 0)] == 1
    np.testing.assert_allclose(shared["force_sums"]["ua"][(0, 0)], F1)
    np.testing.assert_allclose(shared["torque_sums"]["ua"][(0, 0)], T1)

    assert shared["force_counts"]["res"][0] == 1
    np.testing.assert_allclose(shared["force_sums"]["res"][0], F1)
    np.testing.assert_allclose(shared["torque_sums"]["res"][0], T1)

    dag._finalize_means(shared)

    np.testing.assert_allclose(shared["force_covariances"]["ua"][(0, 0)], F1)
    np.testing.assert_allclose(shared["torque_covariances"]["ua"][(0, 0)], T1)
    np.testing.assert_allclose(shared["force_covariances"]["res"][0], F1)
    np.testing.assert_allclose(shared["torque_covariances"]["res"][0], T1)


def test_reduce_forcetorque_no_key_is_noop():
    dag = LevelDAG()
    shared = {
        "forcetorque_sums": {"res": [None], "poly": [None]},
        "forcetorque_counts": {
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "group_id_to_index": {9: 0},
    }

    dag._reduce_forcetorque(shared, frame_out={})
    assert shared["forcetorque_counts"]["res"][0] == 0
    assert shared["forcetorque_sums"]["res"][0] is None


def test_run_frame_stage_calls_execute_frame_for_each_ts(simple_ts_list):
    dag = LevelDAG()

    u = MagicMock()
    u.trajectory = simple_ts_list

    shared = {"reduced_universe": u, "start": 0, "end": 3, "step": 1}

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.side_effect = lambda shared_data, frame_index: {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_one_frame = MagicMock()

    dag._run_frame_stage(shared)

    assert dag._frame_dag.execute_frame.call_count == 3
    assert dag._reduce_one_frame.call_count == 3
