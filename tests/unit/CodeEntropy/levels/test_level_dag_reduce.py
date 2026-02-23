import numpy as np

from CodeEntropy.levels.level_dag import LevelDAG


def test_incremental_mean_first_sample_copies():
    x = np.array([1.0, 2.0])
    out = LevelDAG._incremental_mean(None, x, n=1)
    assert np.allclose(out, x)
    x[0] = 999.0
    assert out[0] != 999.0


def test_reduce_force_and_torque_exercises_count_branches():
    dag = LevelDAG()

    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {"ua": {}, "res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "force": {
            "ua": {(9, 0): np.array([1.0])},
            "res": {7: np.array([2.0])},
            "poly": {7: np.array([3.0])},
        },
        "torque": {
            "ua": {(9, 0): np.array([4.0])},
            "res": {7: np.array([5.0])},
            "poly": {7: np.array([6.0])},
        },
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert (9, 0) in shared["torque_covariances"]["ua"]
    assert shared["frame_counts"]["res"][0] == 1
    assert shared["frame_counts"]["poly"][0] == 1


def test_reduce_forcetorque_returns_when_missing_key():
    dag = LevelDAG()
    shared = {
        "forcetorque_covariances": {"res": [None], "poly": [None]},
        "forcetorque_counts": {"res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }
    dag._reduce_forcetorque(shared, frame_out={})
    assert shared["forcetorque_counts"]["res"][0] == 0


def test_reduce_forcetorque_updates_res_and_poly():
    dag = LevelDAG()

    shared = {
        "forcetorque_covariances": {"res": [None], "poly": [None]},
        "forcetorque_counts": {"res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "forcetorque": {
            "res": {7: np.array([1.0, 1.0])},
            "poly": {7: np.array([2.0, 2.0])},
        }
    }

    dag._reduce_forcetorque(shared, frame_out)

    assert shared["forcetorque_counts"]["res"][0] == 1
    assert shared["forcetorque_counts"]["poly"][0] == 1
    assert shared["forcetorque_covariances"]["res"][0] is not None
    assert shared["forcetorque_covariances"]["poly"][0] is not None


def test_reduce_force_and_torque_res_torque_increments_when_res_count_is_zero():
    dag = LevelDAG()
    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {"ua": {}, "res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {7: np.eye(3)}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["res"][0] == 1
    assert shared["torque_covariances"]["res"][0] is not None


def test_reduce_force_and_torque_poly_torque_increments_when_poly_count_is_zero():
    dag = LevelDAG()
    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {"ua": {}, "res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {7: np.eye(3)}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["poly"][0] == 1
    assert shared["torque_covariances"]["poly"][0] is not None


def test_reduce_force_and_torque_increments_ua_frame_counts_for_force():
    dag = LevelDAG()

    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {"ua": {}, "res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    k = (9, 0)
    frame_out = {
        "force": {"ua": {k: np.eye(3)}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["ua"][k] == 1
    assert k in shared["force_covariances"]["ua"]


def test_reduce_force_and_torque_increments_ua_counts_from_zero():
    dag = LevelDAG()

    key = (9, 0)
    F = np.eye(3)

    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {"ua": {}, "res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "force": {"ua": {key: F}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["ua"][key] == 1

    np.testing.assert_array_equal(shared["force_covariances"]["ua"][key], F)


def test_reduce_force_and_torque_hits_ua_force_count_increment_line():
    dag = LevelDAG()
    key = (9, 0)

    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {"ua": {}, "res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "force": {"ua": {key: np.eye(3)}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["ua"][key] == 1


def test_reduce_force_and_torque_ua_torque_increments_count_when_force_missing_key():
    dag = LevelDAG()

    key = (9, 0)
    T = np.eye(3)

    shared = {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {"ua": {}, "res": [0], "poly": [0]},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {key: T}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["ua"][key] == 1
    np.testing.assert_array_equal(shared["torque_covariances"]["ua"][key], T)
