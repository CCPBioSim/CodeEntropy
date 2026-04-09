import numpy as np

from CodeEntropy.levels.level_dag import LevelDAG


def _shared():
    return {
        "force_sums": {"ua": {}, "res": [None], "poly": [None]},
        "torque_sums": {"ua": {}, "res": [None], "poly": [None]},
        "force_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
        "torque_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
        "forcetorque_sums": {"res": [None], "poly": [None]},
        "forcetorque_counts": {"res": np.array([0]), "poly": np.array([0])},
        "force_covariances": {},
        "torque_covariances": {},
        "forcetorque_covariances": {},
        "group_id_to_index": {7: 0},
    }


def test_reduce_force_and_torque_exercises_count_branches():
    dag = LevelDAG()

    shared = _shared()

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
        "force_counts": {"ua": {(9, 0): 1}, "res": {7: 1}, "poly": {7: 1}},
        "torque_counts": {"ua": {(9, 0): 1}, "res": {7: 1}, "poly": {7: 1}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert (9, 0) in shared["torque_sums"]["ua"]
    assert shared["force_counts"]["res"][0] == 1
    assert shared["force_counts"]["poly"][0] == 1
    assert shared["torque_counts"]["res"][0] == 1
    assert shared["torque_counts"]["poly"][0] == 1


def test_reduce_forcetorque_returns_when_missing_key():
    dag = LevelDAG()
    shared = {
        "forcetorque_sums": {"res": [None], "poly": [None]},
        "forcetorque_counts": {"res": np.array([0]), "poly": np.array([0])},
        "group_id_to_index": {7: 0},
    }
    dag._reduce_forcetorque(shared, frame_out={})
    assert shared["forcetorque_counts"]["res"][0] == 0


def test_reduce_forcetorque_updates_res_and_poly():
    dag = LevelDAG()

    shared = {
        "forcetorque_sums": {"res": [None], "poly": [None]},
        "forcetorque_counts": {"res": np.array([0]), "poly": np.array([0])},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "forcetorque": {
            "res": {7: np.array([1.0, 1.0])},
            "poly": {7: np.array([2.0, 2.0])},
        },
        "forcetorque_counts": {"res": {7: 1}, "poly": {7: 1}},
    }

    dag._reduce_forcetorque(shared, frame_out)

    assert shared["forcetorque_counts"]["res"][0] == 1
    assert shared["forcetorque_counts"]["poly"][0] == 1
    assert shared["forcetorque_sums"]["res"][0] is not None
    assert shared["forcetorque_sums"]["poly"][0] is not None


def test_reduce_force_and_torque_res_torque_increments_when_res_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {7: np.eye(3)}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {7: 1}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["torque_counts"]["res"][0] == 1
    assert shared["torque_sums"]["res"][0] is not None


def test_reduce_force_and_torque_poly_torque_increments_when_poly_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {7: np.eye(3)}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {7: 1}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["torque_counts"]["poly"][0] == 1
    assert shared["torque_sums"]["poly"][0] is not None


def test_reduce_force_and_torque_increments_ua_frame_counts_for_force():
    dag = LevelDAG()

    shared = _shared()

    k = (9, 0)
    frame_out = {
        "force": {"ua": {k: np.eye(3)}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {k: 1}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_counts"]["ua"][k] == 1
    assert k in shared["force_sums"]["ua"]


def test_reduce_force_and_torque_increments_ua_counts_from_zero():
    dag = LevelDAG()

    key = (9, 0)
    F = np.eye(3)

    shared = _shared()

    frame_out = {
        "force": {"ua": {key: F}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {key: 1}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_counts"]["ua"][key] == 1
    np.testing.assert_array_equal(shared["force_sums"]["ua"][key], F)


def test_reduce_force_and_torque_hits_ua_force_count_increment_line():
    dag = LevelDAG()
    key = (9, 0)

    shared = _shared()

    frame_out = {
        "force": {"ua": {key: np.eye(3)}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {key: 1}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_counts"]["ua"][key] == 1


def test_reduce_force_and_torque_ua_torque_increments_count_when_force_missing_key():
    dag = LevelDAG()

    key = (9, 0)
    T = np.eye(3)

    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {key: T}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {key: 1}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["torque_counts"]["ua"][key] == 1
    np.testing.assert_array_equal(shared["torque_sums"]["ua"][key], T)


def test_reduce_one_frame_calls_both_reducers():
    dag = LevelDAG()
    shared = _shared()
    frame_out = {"force": {}, "torque": {}}

    called = {"force": False, "ft": False}

    def fake_reduce_force(shared_data, frame_out_arg):
        called["force"] = True

    def fake_reduce_ft(shared_data, frame_out_arg):
        called["ft"] = True

    dag._reduce_force_and_torque = fake_reduce_force
    dag._reduce_forcetorque = fake_reduce_ft

    dag._reduce_one_frame(shared, frame_out)

    assert called["force"] is True
    assert called["ft"] is True


def test_reduce_force_and_torque_ua_force_continue_when_count_is_zero():
    dag = LevelDAG()
    shared = _shared()
    key = (7, 0)

    frame_out = {
        "force": {"ua": {key: np.eye(2)}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {key: 0}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert key not in shared["force_sums"]["ua"]
    assert key not in shared["force_counts"]["ua"]


def test_reduce_force_and_torque_ua_torque_continue_when_count_is_negative():
    dag = LevelDAG()
    shared = _shared()
    key = (7, 0)

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {key: np.eye(2)}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {key: -3}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert key not in shared["torque_sums"]["ua"]
    assert key not in shared["torque_counts"]["ua"]


def test_reduce_force_and_torque_res_force_continue_when_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {7: np.eye(3)}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {7: 0}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_sums"]["res"][0] is None
    assert shared["force_counts"]["res"][0] == 0


def test_reduce_force_and_torque_res_torque_continue_when_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {7: np.eye(3)}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {7: 0}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["torque_sums"]["res"][0] is None
    assert shared["torque_counts"]["res"][0] == 0


def test_reduce_force_and_torque_poly_force_continue_when_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {7: np.eye(3)}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {7: 0}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_sums"]["poly"][0] is None
    assert shared["force_counts"]["poly"][0] == 0


def test_reduce_force_and_torque_poly_torque_continue_when_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {7: np.eye(3)}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {7: 0}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["torque_sums"]["poly"][0] is None
    assert shared["torque_counts"]["poly"][0] == 0


def test_reduce_forcetorque_res_continue_when_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "forcetorque": {"res": {7: np.eye(4)}, "poly": {}},
        "forcetorque_counts": {"res": {7: 0}, "poly": {}},
    }

    dag._reduce_forcetorque(shared, frame_out)

    assert shared["forcetorque_sums"]["res"][0] is None
    assert shared["forcetorque_counts"]["res"][0] == 0


def test_reduce_forcetorque_poly_continue_when_count_is_zero():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "forcetorque": {"res": {}, "poly": {7: np.eye(4)}},
        "forcetorque_counts": {"res": {}, "poly": {7: 0}},
    }

    dag._reduce_forcetorque(shared, frame_out)

    assert shared["forcetorque_sums"]["poly"][0] is None
    assert shared["forcetorque_counts"]["poly"][0] == 0


def test_reduce_force_and_torque_updates_when_count_is_positive():
    dag = LevelDAG()
    shared = _shared()

    F = np.eye(3)
    T = np.eye(3) * 2

    frame_out = {
        "force": {"ua": {}, "res": {7: F}, "poly": {}},
        "torque": {"ua": {}, "res": {7: T}, "poly": {}},
        "force_counts": {"ua": {}, "res": {7: 1}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {7: 1}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_counts"]["res"][0] == 1
    assert shared["torque_counts"]["res"][0] == 1
    np.testing.assert_allclose(shared["force_sums"]["res"][0], F)
    np.testing.assert_allclose(shared["torque_sums"]["res"][0], T)
