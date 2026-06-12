"""Unit tests for LevelDAG orchestration, reduction, and parallel frame execution."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from CodeEntropy.levels import level_dag as level_dag_module
from CodeEntropy.levels.level_dag import LevelDAG


def _empty_frame_out() -> dict:
    """Return an empty frame-local covariance payload."""
    return {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
    }


def _shared_force_torque() -> dict:
    """Return minimal shared data for force/torque reduction tests."""
    return {
        "force_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "torque_covariances": {"ua": {}, "res": [None], "poly": [None]},
        "frame_counts": {
            "ua": {},
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "group_id_to_index": {7: 0, 9: 0},
    }


def _shared_forcetorque() -> dict:
    """Return minimal shared data for combined force-torque reduction tests."""
    return {
        "forcetorque_covariances": {"res": [None], "poly": [None]},
        "forcetorque_counts": {
            "res": np.zeros(1, dtype=int),
            "poly": np.zeros(1, dtype=int),
        },
        "group_id_to_index": {7: 0, 9: 0},
    }


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


def test_incremental_mean_handles_non_copyable_values():
    out = LevelDAG._incremental_mean(old=None, new=3.0, n=1)

    assert out == 3.0


def test_execute_sets_default_axes_manager_and_runs_stages():
    dag = LevelDAG()

    shared = {
        "reduced_universe": MagicMock(),
        "start": 0,
        "end": 0,
        "step": 1,
        "n_frames": 1,
    }

    dag._run_static_stage = MagicMock()
    dag._run_frame_stage = MagicMock()

    out = dag.execute(shared)

    assert out is shared
    assert "axes_manager" in shared
    dag._run_static_stage.assert_called_once_with(shared, progress=None)
    dag._run_frame_stage.assert_called_once_with(shared, progress=None)


def test_build_registers_static_nodes_and_builds_frame_dag():
    with (
        patch("CodeEntropy.levels.level_dag.DetectMoleculesNode"),
        patch("CodeEntropy.levels.level_dag.DetectLevelsNode"),
        patch("CodeEntropy.levels.level_dag.BuildBeadsNode"),
        patch("CodeEntropy.levels.level_dag.InitCovarianceAccumulatorsNode"),
        patch("CodeEntropy.levels.level_dag.ComputeConformationalStatesNode"),
        patch("CodeEntropy.levels.level_dag.ComputeNeighborsNode"),
    ):
        dag = LevelDAG(universe_operations=MagicMock())
        dag._frame_dag.build = MagicMock()

        out = dag.build()

    assert out is dag
    assert "detect_molecules" in dag._static_nodes
    assert "detect_levels" in dag._static_nodes
    assert "build_beads" in dag._static_nodes
    assert "init_covariance_accumulators" in dag._static_nodes
    assert "compute_conformational_states" in dag._static_nodes
    assert "find_neighbors" in dag._static_nodes
    dag._frame_dag.build.assert_called_once()


def test_add_static_adds_dependency_edges():
    dag = LevelDAG()

    dag._add_static("A", MagicMock())
    dag._add_static("B", MagicMock(), deps=["A"])

    assert dag._static_nodes["A"] is not None
    assert dag._static_nodes["B"] is not None
    assert ("A", "B") in dag._static_graph.edges


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

    node = MagicMock()
    node.run.side_effect = [TypeError("no progress"), None]
    dag._static_nodes["a"] = node

    progress = MagicMock()

    with patch("networkx.topological_sort", return_value=["a"]):
        dag._run_static_stage({"X": 1}, progress=progress)

    assert node.run.call_count == 2
    node.run.assert_any_call({"X": 1}, progress=progress)
    node.run.assert_any_call({"X": 1})


def test_run_frame_stage_iterates_selected_frames_and_reduces_each():
    dag = LevelDAG()

    frame_source = MagicMock()
    frame_source.iter_indices.return_value = [10, 11]

    shared = {
        "frame_source": frame_source,
        "n_frames": 2,
    }

    frame_outputs = [_empty_frame_out(), _empty_frame_out()]

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.side_effect = frame_outputs
    dag._reduce_one_frame = MagicMock()

    dag._run_frame_stage(shared)

    assert shared["n_frames"] == 2
    frame_source.iter_indices.assert_called_once()

    assert dag._frame_dag.execute_frame.call_count == 2
    dag._frame_dag.execute_frame.assert_any_call(shared, 10)
    dag._frame_dag.execute_frame.assert_any_call(shared, 11)

    assert dag._reduce_one_frame.call_count == 2
    dag._reduce_one_frame.assert_any_call(shared, frame_outputs[0])
    dag._reduce_one_frame.assert_any_call(shared, frame_outputs[1])


def test_run_frame_stage_progress_total_comes_from_frame_source_indices():
    dag = LevelDAG()

    frame_source = MagicMock()
    frame_source.iter_indices.return_value = list(range(10))

    shared = {
        "frame_source": frame_source,
        "n_frames": 0,
    }

    frame_out = _empty_frame_out()

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.return_value = frame_out
    dag._reduce_one_frame = MagicMock()

    progress = MagicMock()
    progress.add_task.return_value = 123

    dag._run_frame_stage(shared, progress=progress)

    progress.add_task.assert_called_once_with(
        "[green]Frame processing",
        total=10,
        title="Initializing",
    )

    assert shared["n_frames"] == 10
    frame_source.iter_indices.assert_called_once()
    assert dag._frame_dag.execute_frame.call_count == 10
    assert dag._reduce_one_frame.call_count == 10
    assert progress.advance.call_count == 10


def test_run_frame_stage_with_progress_creates_task_and_updates_titles():
    dag = LevelDAG()

    frame_source = MagicMock()
    frame_source.iter_indices.return_value = [10, 11]

    shared = {
        "frame_source": frame_source,
        "n_frames": 2,
    }

    frame_out = _empty_frame_out()

    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.return_value = frame_out
    dag._reduce_one_frame = MagicMock()

    progress = MagicMock()
    progress.add_task.return_value = 77

    dag._run_frame_stage(shared, progress=progress)

    progress.add_task.assert_called_once_with(
        "[green]Frame processing",
        total=2,
        title="Initializing",
    )

    assert progress.update.call_count == 2
    progress.update.assert_any_call(77, title="Frame 10")
    progress.update.assert_any_call(77, title="Frame 11")

    assert progress.advance.call_count == 2
    progress.advance.assert_any_call(77)

    assert dag._frame_dag.execute_frame.call_count == 2
    dag._frame_dag.execute_frame.assert_any_call(shared, 10)
    dag._frame_dag.execute_frame.assert_any_call(shared, 11)

    assert dag._reduce_one_frame.call_count == 2
    dag._reduce_one_frame.assert_any_call(shared, frame_out)


def test_run_frame_stage_falls_back_to_sequential_when_only_one_frame():
    dag = LevelDAG()

    frame_source = MagicMock()
    frame_source.iter_indices.return_value = [0]

    client = MagicMock()

    shared_data = {
        "frame_source": frame_source,
        "dask_client": client,
        "parallel_frames": True,
    }

    frame_out = _empty_frame_out()

    dag._run_frame_stage_dask = MagicMock()
    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.return_value = frame_out
    dag._reduce_one_frame = MagicMock()

    dag._run_frame_stage(shared_data)

    dag._run_frame_stage_dask.assert_not_called()
    dag._frame_dag.execute_frame.assert_called_once_with(shared_data, 0)
    dag._reduce_one_frame.assert_called_once_with(shared_data, frame_out)
    assert shared_data["n_frames"] == 1


def test_run_frame_stage_falls_back_to_sequential_without_client():
    dag = LevelDAG()

    frame_source = MagicMock()
    frame_source.iter_indices.return_value = [0, 1]

    shared_data = {
        "frame_source": frame_source,
        "parallel_frames": True,
    }

    frame_out = _empty_frame_out()

    dag._run_frame_stage_dask = MagicMock()
    dag._frame_dag = MagicMock()
    dag._frame_dag.execute_frame.return_value = frame_out
    dag._reduce_one_frame = MagicMock()

    dag._run_frame_stage(shared_data)

    dag._run_frame_stage_dask.assert_not_called()
    assert dag._frame_dag.execute_frame.call_count == 2
    assert dag._reduce_one_frame.call_count == 2
    assert shared_data["n_frames"] == 2


def test_reduce_force_and_torque_handles_empty_frame_gracefully():
    dag = LevelDAG()
    shared = _shared_force_torque()

    dag._reduce_force_and_torque(shared_data=shared, frame_out=_empty_frame_out())

    assert shared["force_covariances"]["ua"] == {}
    assert shared["torque_covariances"]["ua"] == {}
    assert shared["frame_counts"]["res"][0] == 0
    assert shared["frame_counts"]["poly"][0] == 0


def test_reduce_force_and_torque_updates_counts_and_means():
    dag = LevelDAG()
    shared = _shared_force_torque()

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


def test_reduce_force_and_torque_exercises_count_branches():
    dag = LevelDAG()
    shared = _shared_force_torque()

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
    np.testing.assert_allclose(shared["force_covariances"]["res"][0], np.array([2.0]))
    np.testing.assert_allclose(shared["torque_covariances"]["res"][0], np.array([5.0]))
    np.testing.assert_allclose(shared["force_covariances"]["poly"][0], np.array([3.0]))
    np.testing.assert_allclose(shared["torque_covariances"]["poly"][0], np.array([6.0]))


def test_reduce_force_and_torque_res_torque_increments_when_res_count_is_zero():
    dag = LevelDAG()
    shared = _shared_force_torque()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {7: np.eye(3)}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["res"][0] == 1
    assert shared["torque_covariances"]["res"][0] is not None


def test_reduce_force_and_torque_poly_torque_increments_when_poly_count_is_zero():
    dag = LevelDAG()
    shared = _shared_force_torque()

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {7: np.eye(3)}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["poly"][0] == 1
    assert shared["torque_covariances"]["poly"][0] is not None


def test_reduce_force_and_torque_increments_ua_frame_counts_for_force():
    dag = LevelDAG()
    shared = _shared_force_torque()

    key = (9, 0)
    F = np.eye(3)

    frame_out = {
        "force": {"ua": {key: F}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["ua"][key] == 1
    assert key in shared["force_covariances"]["ua"]
    np.testing.assert_array_equal(shared["force_covariances"]["ua"][key], F)


def test_reduce_force_and_torque_ua_torque_increments_count_when_force_missing_key():
    dag = LevelDAG()
    shared = _shared_force_torque()

    key = (9, 0)
    T = np.eye(3)

    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {key: T}, "res": {}, "poly": {}},
    }

    dag._reduce_force_and_torque(shared, frame_out)

    assert shared["frame_counts"]["ua"][key] == 1
    np.testing.assert_array_equal(shared["torque_covariances"]["ua"][key], T)


def test_reduce_one_frame_calls_force_torque_and_forcetorque_reducers():
    dag = LevelDAG()
    shared = {}
    frame_out = {}

    dag._reduce_force_and_torque = MagicMock()
    dag._reduce_forcetorque = MagicMock()

    dag._reduce_one_frame(shared, frame_out)

    dag._reduce_force_and_torque.assert_called_once_with(shared, frame_out)
    dag._reduce_forcetorque.assert_called_once_with(shared, frame_out)


def test_reduce_forcetorque_no_key_is_noop():
    dag = LevelDAG()
    shared = _shared_forcetorque()

    dag._reduce_forcetorque(shared, frame_out={})

    assert shared["forcetorque_counts"]["res"][0] == 0
    assert shared["forcetorque_covariances"]["res"][0] is None


def test_reduce_forcetorque_updates_res_and_poly():
    dag = LevelDAG()
    shared = _shared_forcetorque()

    frame_out = {
        "forcetorque": {
            "res": {7: np.array([1.0, 1.0])},
            "poly": {7: np.array([2.0, 2.0])},
        }
    }

    dag._reduce_forcetorque(shared, frame_out)

    assert shared["forcetorque_counts"]["res"][0] == 1
    assert shared["forcetorque_counts"]["poly"][0] == 1
    np.testing.assert_allclose(
        shared["forcetorque_covariances"]["res"][0],
        np.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(
        shared["forcetorque_covariances"]["poly"][0],
        np.array([2.0, 2.0]),
    )


def test_make_frame_worker_shared_data_excludes_parent_only_keys():
    shared_data = {
        "force_covariances": "force accumulator",
        "torque_covariances": "torque accumulator",
        "forcetorque_covariances": "ft accumulator",
        "frame_counts": "frame counts",
        "forcetorque_counts": "ft counts",
        "force_torque_stats": "legacy ft accumulator alias",
        "force_torque_counts": "legacy ft counts alias",
        "n_frames": 10,
        "entropy_manager": "manager",
        "run_manager": "run manager",
        "reporter": "reporter",
        "dask_client": "client",
        "frame_source": "frame source",
        "levels": "levels",
        "groups": "groups",
        "args": "args",
    }

    worker_shared = LevelDAG._make_frame_worker_shared_data(shared_data)

    assert worker_shared == {
        "frame_source": "frame source",
        "levels": "levels",
        "groups": "groups",
        "args": "args",
    }


def test_execute_frame_worker_builds_frame_graph_and_returns_frame_output():
    shared_data = {"x": 1}
    universe_operations = MagicMock()

    with patch("CodeEntropy.levels.level_dag.FrameGraph") as FrameGraphCls:
        graph = MagicMock()
        graph.execute_frame.return_value = {"force": {}, "torque": {}}
        FrameGraphCls.return_value.build.return_value = graph

        frame_index, frame_out = level_dag_module._execute_frame_worker(
            shared_data,
            frame_index="5",
            universe_operations=universe_operations,
        )

    FrameGraphCls.assert_called_once_with(universe_operations=universe_operations)
    FrameGraphCls.return_value.build.assert_called_once()
    graph.execute_frame.assert_called_once_with(shared_data, 5)

    assert frame_index == 5
    assert frame_out == {"force": {}, "torque": {}}


def test_run_frame_stage_uses_dask_when_client_present():
    dag = LevelDAG()

    frame_source = MagicMock()
    frame_source.iter_indices.return_value = [0, 1, 2]

    client = MagicMock()

    shared_data = {
        "frame_source": frame_source,
        "dask_client": client,
        "parallel_frames": True,
    }

    dag._run_frame_stage_dask = MagicMock()
    dag._frame_dag = MagicMock()
    dag._reduce_one_frame = MagicMock()

    dag._run_frame_stage(shared_data)

    dag._run_frame_stage_dask.assert_called_once_with(
        shared_data,
        frame_indices=[0, 1, 2],
        client=client,
        progress=None,
        task=None,
    )
    dag._frame_dag.execute_frame.assert_not_called()
    dag._reduce_one_frame.assert_not_called()
    assert shared_data["n_frames"] == 3


def test_run_frame_stage_dask_submits_each_frame_and_reduces_completed_results():
    dag = LevelDAG()

    shared_data = {
        "keep": "value",
        "force_covariances": "exclude me",
        "reporter": "exclude me too",
    }

    client = MagicMock()

    frame_out0 = _empty_frame_out()
    frame_out1 = _empty_frame_out()

    future0 = MagicMock()
    future0.result.return_value = (0, frame_out0)

    future1 = MagicMock()
    future1.result.return_value = (1, frame_out1)

    client.submit.side_effect = [future0, future1]

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.as_completed = MagicMock(return_value=[future0, future1])

    dag._reduce_one_frame = MagicMock()

    with patch.dict(sys.modules, {"distributed": fake_distributed}):
        dag._run_frame_stage_dask(
            shared_data,
            frame_indices=[0, 1],
            client=client,
            progress=None,
            task=None,
        )

    assert client.submit.call_count == 2

    for call in client.submit.call_args_list:
        args, kwargs = call
        assert args[0] is level_dag_module._execute_frame_worker
        assert args[1] == {"keep": "value"}
        assert kwargs == {"pure": False}

    assert dag._reduce_one_frame.call_count == 2
    dag._reduce_one_frame.assert_any_call(shared_data, frame_out0)
    dag._reduce_one_frame.assert_any_call(shared_data, frame_out1)
    client.cancel.assert_not_called()


def test_run_frame_stage_dask_updates_progress():
    dag = LevelDAG()

    shared_data = {"keep": "value"}
    client = MagicMock()

    frame_out = _empty_frame_out()
    future = MagicMock()
    future.result.return_value = (7, frame_out)
    client.submit.return_value = future

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.as_completed = MagicMock(return_value=[future])

    progress = MagicMock()
    dag._reduce_one_frame = MagicMock()

    with patch.dict(sys.modules, {"distributed": fake_distributed}):
        dag._run_frame_stage_dask(
            shared_data,
            frame_indices=[7],
            client=client,
            progress=progress,
            task="task-id",
        )

    progress.update.assert_called_once_with("task-id", title="Frame 7")
    progress.advance.assert_called_once_with("task-id")
    dag._reduce_one_frame.assert_called_once_with(shared_data, frame_out)


def test_run_frame_stage_dask_cancels_futures_and_reraises_on_result_error():
    dag = LevelDAG()

    shared_data = {"keep": "value"}
    client = MagicMock()

    future = MagicMock()
    future.result.side_effect = RuntimeError("worker failed")
    client.submit.return_value = future

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.as_completed = MagicMock(return_value=[future])

    with patch.dict(sys.modules, {"distributed": fake_distributed}):
        with pytest.raises(RuntimeError, match="worker failed"):
            dag._run_frame_stage_dask(
                shared_data,
                frame_indices=[0],
                client=client,
                progress=None,
                task=None,
            )

    client.cancel.assert_called_once_with([future])


def test_run_frame_stage_dask_raises_when_distributed_missing():
    dag = LevelDAG()
    client = MagicMock()

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "distributed":
            raise ImportError("No module named distributed")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="requires dask.distributed"):
            dag._run_frame_stage_dask(
                {"keep": "value"},
                frame_indices=[0],
                client=client,
                progress=None,
                task=None,
            )


def test_run_frame_stage_dask_raises_if_completed_count_mismatch():
    dag = LevelDAG()

    shared_data = {"keep": "value"}
    client = MagicMock()

    future0 = MagicMock()
    future0.result.return_value = (0, _empty_frame_out())

    future1 = MagicMock()

    client.submit.side_effect = [future0, future1]

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.as_completed = MagicMock(return_value=[future0])

    dag._reduce_one_frame = MagicMock()

    with patch.dict(sys.modules, {"distributed": fake_distributed}):
        with pytest.raises(
            RuntimeError,
            match="Parallel frame execution completed 1 frames, but expected 2",
        ):
            dag._run_frame_stage_dask(
                shared_data,
                frame_indices=[0, 1],
                client=client,
                progress=None,
                task=None,
            )

    client.cancel.assert_called_once()
