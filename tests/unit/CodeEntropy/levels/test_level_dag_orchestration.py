from unittest.mock import MagicMock, patch

import numpy as np

from CodeEntropy.levels.level_dag import LevelDAG


def _shared():
    return {
        "levels": [["united_atom"]],
        "group_id_to_index": {0: 0},
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
    }


def test_execute_sets_default_axes_manager_once():
    dag = LevelDAG()

    shared = {
        "reduced_universe": MagicMock(),
        "start": 0,
        "end": 0,
        "step": 1,
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
    }

    dag._run_static_stage = MagicMock()
    dag._run_frame_stage = MagicMock()
    dag._finalize_means = MagicMock()

    dag.execute(shared)

    assert "axes_manager" in shared
    dag._run_static_stage.assert_called_once()
    dag._run_frame_stage.assert_called_once()
    dag._finalize_means.assert_called_once_with(shared)


def test_run_static_stage_calls_nodes_in_topological_sort_order():
    dag = LevelDAG()
    dag._static_graph.add_node("a")
    dag._static_graph.add_node("b")

    dag._static_nodes["a"] = MagicMock()
    dag._static_nodes["b"] = MagicMock()

    with patch("networkx.topological_sort", return_value=["a", "b"]):
        dag._run_static_stage({"X": 1})

    dag._static_nodes["a"].run.assert_called_once()
    dag._static_nodes["b"].run.assert_called_once()


def test_run_frame_stage_iterates_selected_frames_and_reduces_each():
    dag = LevelDAG()

    ts0 = MagicMock(frame=10)
    ts1 = MagicMock(frame=11)
    u = MagicMock()
    u.trajectory = [ts0, ts1]

    shared = {"reduced_universe": u, "start": 0, "end": 2, "step": 1}

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.side_effect = [
        {
            "force": {"ua": {}, "res": {}, "poly": {}},
            "torque": {"ua": {}, "res": {}, "poly": {}},
            "force_counts": {"ua": {}, "res": {}, "poly": {}},
            "torque_counts": {"ua": {}, "res": {}, "poly": {}},
        }
    ] * 2
    dag._reduce_one_frame = MagicMock()

    dag._run_frame_stage(shared)

    assert dag._frame_dag.execute_frame.call_count == 2
    assert dag._reduce_one_frame.call_count == 2
    dag._frame_dag.execute_frame.assert_any_call(shared, 10)
    dag._frame_dag.execute_frame.assert_any_call(shared, 11)


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


def test_build_registers_static_nodes_and_builds_frame_dag():
    with (
        patch("CodeEntropy.levels.level_dag.DetectMoleculesNode") as _,
        patch("CodeEntropy.levels.level_dag.DetectLevelsNode") as _,
        patch("CodeEntropy.levels.level_dag.BuildBeadsNode") as _,
        patch("CodeEntropy.levels.level_dag.InitCovarianceAccumulatorsNode") as _,
        patch("CodeEntropy.levels.level_dag.ComputeConformationalStatesNode") as _,
    ):
        dag = LevelDAG(universe_operations=MagicMock())
        dag._frame_dag.build = MagicMock()

        dag.build()

        assert "detect_molecules" in dag._static_nodes
        assert "detect_levels" in dag._static_nodes
        assert "build_beads" in dag._static_nodes
        assert "init_covariance_accumulators" in dag._static_nodes
        assert "compute_conformational_states" in dag._static_nodes
        dag._frame_dag.build.assert_called_once()


def test_add_static_adds_dependency_edges():
    dag = LevelDAG()
    dag._add_static("A", MagicMock())
    dag._add_static("B", MagicMock(), deps=["A"])

    assert ("A", "B") in dag._static_graph.edges


def test_reduce_force_and_torque_hits_zero_count_branches():
    dag = LevelDAG()

    shared = {
        "force_sums": {"ua": {}, "res": [None], "poly": [None]},
        "torque_sums": {"ua": {}, "res": [None], "poly": [None]},
        "force_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
        "torque_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
        "group_id_to_index": {7: 0},
    }

    frame_out = {
        "force": {
            "ua": {(7, 0): np.eye(1)},
            "res": {7: np.eye(2)},
            "poly": {7: np.eye(3)},
        },
        "torque": {
            "ua": {(7, 0): np.eye(1)},
            "res": {7: np.eye(2)},
            "poly": {7: np.eye(3)},
        },
        "force_counts": {"ua": {(7, 0): 1}, "res": {7: 1}, "poly": {7: 1}},
        "torque_counts": {"ua": {(7, 0): 1}, "res": {7: 1}, "poly": {7: 1}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["force_counts"]["ua"][(7, 0)] == 1
    assert (7, 0) in shared["force_sums"]["ua"]
    assert (7, 0) in shared["torque_sums"]["ua"]

    assert shared["force_counts"]["res"][0] == 1
    assert shared["force_counts"]["poly"][0] == 1
    assert shared["torque_counts"]["res"][0] == 1
    assert shared["torque_counts"]["poly"][0] == 1


def test_reduce_force_and_torque_handles_empty_frame_gracefully():
    dag = LevelDAG()

    shared = {
        "group_id_to_index": {0: 0},
        "force_sums": {"ua": {}, "res": [None], "poly": [None]},
        "torque_sums": {"ua": {}, "res": [None], "poly": [None]},
        "force_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
        "torque_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
    }

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared_data=shared, frame_out=frame_out)

    assert shared["force_sums"]["ua"] == {}
    assert shared["torque_sums"]["ua"] == {}
    assert shared["force_counts"]["res"][0] == 0
    assert shared["force_counts"]["poly"][0] == 0
    assert shared["torque_counts"]["res"][0] == 0
    assert shared["torque_counts"]["poly"][0] == 0


def test_reduce_force_and_torque_increments_res_and_poly_counts_from_zero():
    dag = LevelDAG()

    shared = {
        "group_id_to_index": {7: 0},
        "force_sums": {"ua": {}, "res": [None], "poly": [None]},
        "torque_sums": {"ua": {}, "res": [None], "poly": [None]},
        "force_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
        "torque_counts": {"ua": {}, "res": np.array([0]), "poly": np.array([0])},
        "forcetorque_sums": {"res": [None], "poly": [None]},
        "forcetorque_counts": {"res": np.array([0]), "poly": np.array([0])},
    }

    F = np.eye(3)
    T = np.eye(3) * 2

    frame_out = {
        "force": {"ua": {}, "res": {7: F}, "poly": {7: F}},
        "torque": {"ua": {}, "res": {7: T}, "poly": {7: T}},
        "force_counts": {"ua": {}, "res": {7: 1}, "poly": {7: 1}},
        "torque_counts": {"ua": {}, "res": {7: 1}, "poly": {7: 1}},
    }

    dag._reduce_force_and_torque(shared_data=shared, frame_out=frame_out)

    assert shared["force_counts"]["res"][0] == 1
    assert shared["force_counts"]["poly"][0] == 1
    assert shared["torque_counts"]["res"][0] == 1
    assert shared["torque_counts"]["poly"][0] == 1
    assert np.allclose(shared["torque_sums"]["res"][0], T)
    assert np.allclose(shared["torque_sums"]["poly"][0], T)

    shared["force_covariances"] = {}
    shared["torque_covariances"] = {}
    shared["forcetorque_covariances"] = {}
    dag._finalize_means(shared)

    assert np.allclose(shared["torque_covariances"]["res"][0], T)
    assert np.allclose(shared["torque_covariances"]["poly"][0], T)


def test_reduce_one_frame_skips_missing_force_and_torque_keys():
    dag = LevelDAG()
    shared = _shared()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_one_frame(shared_data=shared, frame_out=frame_out)

    assert shared["force_sums"]["ua"] == {}
    assert shared["torque_sums"]["ua"] == {}


def test_reduce_force_and_torque_skips_when_counts_are_zero():
    dag = LevelDAG()
    shared = _shared()

    k = (0, 0)
    shared["force_sums"]["ua"][k] = np.eye(3)
    shared["torque_sums"]["ua"][k] = np.eye(3)
    shared["force_counts"]["ua"][k] = 0
    shared["torque_counts"]["ua"][k] = 0

    frame_out = {
        "force": {"ua": {k: np.eye(3)}, "res": {}, "poly": {}},
        "torque": {"ua": {k: np.eye(3)}, "res": {}, "poly": {}},
        "force_counts": {"ua": {k: 0}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {k: 0}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared_data=shared, frame_out=frame_out)

    np.testing.assert_array_equal(shared["force_sums"]["ua"][k], np.eye(3))
    np.testing.assert_array_equal(shared["torque_sums"]["ua"][k], np.eye(3))
    assert shared["force_counts"]["ua"][k] == 0
    assert shared["torque_counts"]["ua"][k] == 0


def test_run_static_stage_forwards_progress_when_node_accepts_it():
    dag = LevelDAG()
    dag._static_graph.add_node("a")

    node = MagicMock()
    dag._static_nodes["a"] = node

    progress = MagicMock()

    with patch("networkx.topological_sort", return_value=["a"]):
        dag._run_static_stage({"X": 1}, progress=progress)

    node.run.assert_called_once_with({"X": 1}, progress=progress)


def test_run_static_stage_falls_back_when_node_does_not_accept_progress():
    dag = LevelDAG()
    dag._static_graph.add_node("a")

    class NoProgressNode:
        def run(self, shared_data):
            return None

    dag._static_nodes["a"] = NoProgressNode()
    progress = MagicMock()

    with patch("networkx.topological_sort", return_value=["a"]):
        dag._run_static_stage({"X": 1}, progress=progress)


def test_run_frame_stage_with_progress_creates_task_and_updates_titles():
    dag = LevelDAG()

    ts0 = MagicMock(frame=10)
    ts1 = MagicMock(frame=11)
    u = MagicMock()
    u.trajectory = [ts0, ts1]

    shared = {"reduced_universe": u, "start": 0, "end": 2, "step": 1}

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.return_value = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }
    dag._reduce_one_frame = MagicMock()

    progress = MagicMock()
    progress.add_task.return_value = 77

    dag._run_frame_stage(shared, progress=progress)

    progress.add_task.assert_called_once()
    progress.update.assert_any_call(77, title="Frame 10")
    progress.update.assert_any_call(77, title="Frame 11")
    assert progress.advance.call_count == 2


def test_run_frame_stage_with_negative_end_computes_total_frames():
    dag = LevelDAG()

    ts_list = [MagicMock(frame=i) for i in range(10)]
    u = MagicMock()
    u.trajectory = ts_list

    shared = {
        "reduced_universe": u,
        "start": 0,
        "end": -1,
        "step": 1,
    }

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.return_value = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "force_counts": {"ua": {}, "res": {}, "poly": {}},
        "torque_counts": {"ua": {}, "res": {}, "poly": {}},
    }
    dag._reduce_one_frame = MagicMock()

    progress = MagicMock()
    progress.add_task.return_value = 123

    dag._run_frame_stage(shared, progress=progress)

    progress.add_task.assert_called_once()
    _, kwargs = progress.add_task.call_args
    assert kwargs["total"] == 9

    assert progress.advance.call_count == 9


def test_run_frame_stage_progress_total_frames_falls_back_to_none_on_error():
    dag = LevelDAG()

    class BadTrajectory:
        def __len__(self):
            raise RuntimeError("boom")

        def __getitem__(self, item):
            return []

    u = type("U", (), {})()
    u.trajectory = BadTrajectory()

    shared = {
        "reduced_universe": u,
        "start": 0,
        "end": 10,
        "step": 1,
    }

    dag._frame_dag = MagicMock()
    dag._reduce_one_frame = MagicMock()

    progress = MagicMock()
    progress.add_task.return_value = 99

    dag._run_frame_stage(shared, progress=progress)

    _, kwargs = progress.add_task.call_args
    assert kwargs["total"] is None
