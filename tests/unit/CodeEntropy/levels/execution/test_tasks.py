"""Unit tests for frame-chunk task helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from CodeEntropy.levels.execution.tasks import (
    CovarianceChunkPartial,
    FrameChunkResult,
    FrameChunkTask,
    execute_frame_chunk_worker,
    execute_frame_map_output,
    get_worker_frame_universe,
    incremental_mean_value,
    make_frame_worker_shared_data,
    reduce_frame_covariance_into_partial,
)


def _frame_covariance(force_value: float) -> dict:
    force = force_value * np.eye(3)
    torque = (force_value + 1.0) * np.eye(3)

    return {
        "force": {"ua": {(0, 0): force}, "res": {0: force}, "poly": {}},
        "torque": {"ua": {(0, 0): torque}, "res": {0: torque}, "poly": {}},
    }


def test_make_frame_worker_shared_data_excludes_parent_owned_state():
    shared_data = {
        "force_covariances": "exclude",
        "torque_covariances": "exclude",
        "forcetorque_covariances": "exclude",
        "frame_counts": "exclude",
        "forcetorque_counts": "exclude",
        "neighbor_totals": "exclude",
        "neighbor_samples": "exclude",
        "n_frames": 10,
        "entropy_manager": "exclude",
        "run_manager": "exclude",
        "reporter": "exclude",
        "dask_client": "exclude",
        "frame_source": "keep",
        "levels": "keep",
        "groups": "keep",
        "args": "keep",
    }

    assert make_frame_worker_shared_data(shared_data) == {
        "frame_source": "keep",
        "levels": "keep",
        "groups": "keep",
        "args": "keep",
    }


def test_frame_chunk_task_contains_only_lightweight_task_descriptor():
    task = FrameChunkTask(chunk_index=3, frame_indices=(10, 11))

    assert task.chunk_index == 3
    assert task.frame_indices == (10, 11)
    assert not hasattr(task, "worker_shared_data")
    assert not hasattr(task, "include_neighbors")


def test_incremental_mean_value_returns_copy_for_first_numpy_value():
    new = np.array([1.0, 2.0])

    out = incremental_mean_value(None, new, n=1)

    np.testing.assert_allclose(out, new)
    new[0] = 99.0
    assert out[0] != 99.0


def test_incremental_mean_value_handles_non_copyable_first_value():
    assert incremental_mean_value(None, 3.0, n=1) == 3.0


def test_incremental_mean_value_updates_mean():
    old = np.array([2.0, 2.0])
    new = np.array([4.0, 0.0])

    np.testing.assert_allclose(
        incremental_mean_value(old, new, n=2),
        np.array([3.0, 1.0]),
    )


def test_reduce_frame_covariance_into_partial_accumulates_running_means():
    partial = CovarianceChunkPartial()

    reduce_frame_covariance_into_partial(partial, _frame_covariance(1.0))
    reduce_frame_covariance_into_partial(partial, _frame_covariance(3.0))

    np.testing.assert_allclose(partial.force["ua"][(0, 0)], 2.0 * np.eye(3))
    np.testing.assert_allclose(partial.torque["ua"][(0, 0)], 3.0 * np.eye(3))
    np.testing.assert_allclose(partial.force["res"][0], 2.0 * np.eye(3))
    np.testing.assert_allclose(partial.torque["res"][0], 3.0 * np.eye(3))

    assert partial.frame_counts["ua"][(0, 0)] == 2
    assert partial.frame_counts["res"][0] == 2


def test_reduce_frame_covariance_into_partial_handles_missing_force_keys_for_torque():
    partial = CovarianceChunkPartial()
    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {
            "ua": {(0, 0): np.eye(3)},
            "res": {0: np.eye(3)},
            "poly": {0: np.eye(3)},
        },
    }

    reduce_frame_covariance_into_partial(partial, frame_out)

    assert partial.frame_counts["ua"][(0, 0)] == 1
    assert partial.frame_counts["res"][0] == 1
    assert partial.frame_counts["poly"][0] == 1


def test_reduce_frame_covariance_into_partial_handles_forcetorque_blocks():
    partial = CovarianceChunkPartial()
    ft = np.ones((6, 6))
    frame_out = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "forcetorque": {"res": {0: ft}, "poly": {0: 2.0 * ft}},
    }

    reduce_frame_covariance_into_partial(partial, frame_out)

    np.testing.assert_allclose(partial.forcetorque["res"][0], ft)
    np.testing.assert_allclose(partial.forcetorque["poly"][0], 2.0 * ft)
    assert partial.forcetorque_counts["res"][0] == 1
    assert partial.forcetorque_counts["poly"][0] == 1


def test_execute_frame_map_output_runs_covariance_and_neighbor_count():
    frame_dag = MagicMock()
    frame_dag.execute_frame.return_value = _frame_covariance(1.0)

    neighbor_helper = MagicMock()
    neighbor_helper.get_frame_neighbor_counts.return_value = {0: (4, 2)}

    shared_data = {
        "reduced_universe": "universe",
        "levels": [["united_atom"]],
        "groups": {0: [0]},
        "frame_source": "frame-source",
        "args": SimpleNamespace(search_type="RAD"),
    }

    out = execute_frame_map_output(
        shared_data=shared_data,
        frame_index="5",
        frame_dag=frame_dag,
        neighbor_helper=neighbor_helper,
    )

    frame_dag.execute_frame.assert_called_once_with(shared_data, 5)
    neighbor_helper.get_frame_neighbor_counts.assert_called_once_with(
        universe="universe",
        levels=[["united_atom"]],
        groups={0: [0]},
        frame_source="frame-source",
        frame_index=5,
        search_type="RAD",
    )

    assert out["covariance"] == frame_dag.execute_frame.return_value
    assert out["neighbors"] == {0: (4, 2)}


def test_execute_frame_map_output_constructs_neighbor_helper_when_not_provided():
    frame_dag = MagicMock()
    frame_dag.execute_frame.return_value = _frame_covariance(1.0)

    shared_data = {
        "universe": "fallback-universe",
        "levels": [["united_atom"]],
        "groups": {0: [0]},
        "frame_source": "frame-source",
        "args": SimpleNamespace(search_type="RAD"),
    }

    with patch("CodeEntropy.levels.execution.tasks.Neighbors") as Neighbors:
        helper = Neighbors.return_value
        helper.get_frame_neighbor_counts.return_value = {0: (1, 1)}

        out = execute_frame_map_output(
            shared_data=shared_data,
            frame_index=0,
            frame_dag=frame_dag,
        )

    helper.get_frame_neighbor_counts.assert_called_once_with(
        universe="fallback-universe",
        levels=[["united_atom"]],
        groups={0: [0]},
        frame_source="frame-source",
        frame_index=0,
        search_type="RAD",
    )
    assert out["neighbors"] == {0: (1, 1)}


def test_execute_frame_chunk_worker_returns_compact_partials():
    worker_shared_data = {
        "reduced_universe": "universe",
        "levels": [["united_atom"]],
        "groups": {0: [0]},
        "frame_source": "frame-source",
        "args": SimpleNamespace(search_type="RAD"),
    }
    task = FrameChunkTask(chunk_index=1, frame_indices=(0, 1))
    universe_operations = object()

    graph = MagicMock()
    graph.execute_frame.side_effect = [
        _frame_covariance(1.0),
        _frame_covariance(3.0),
    ]

    neighbor_helper = MagicMock()
    neighbor_helper.get_frame_neighbor_counts.side_effect = [
        {0: (2, 1)},
        {0: (4, 1)},
    ]

    with (
        patch("CodeEntropy.levels.execution.tasks.FrameGraph") as FrameGraph,
        patch("CodeEntropy.levels.execution.tasks.Neighbors") as Neighbors,
    ):
        FrameGraph.return_value.build.return_value = graph
        Neighbors.return_value = neighbor_helper

        result = execute_frame_chunk_worker(
            task,
            worker_shared_data,
            universe_operations=universe_operations,
        )

    FrameGraph.assert_called_once_with(universe_operations=universe_operations)
    graph.execute_frame.assert_any_call(worker_shared_data, 0)
    graph.execute_frame.assert_any_call(worker_shared_data, 1)

    assert isinstance(result, FrameChunkResult)
    assert result.chunk_index == 1
    assert result.frame_indices == (0, 1)
    assert result.neighbor_totals == {0: 6}
    assert result.neighbor_samples == {0: 2}
    np.testing.assert_allclose(
        result.covariance_partial.force["ua"][(0, 0)],
        2.0 * np.eye(3),
    )


def test_covariance_chunk_partial_default_factories_are_independent():
    partial_a = CovarianceChunkPartial()
    partial_b = CovarianceChunkPartial()

    partial_a.force["ua"][(0, 0)] = "value"
    partial_a.frame_counts["res"][0] = 1
    partial_a.forcetorque["poly"][0] = "ft"

    assert partial_b.force["ua"] == {}
    assert partial_b.frame_counts["res"] == {}
    assert partial_b.forcetorque["poly"] == {}


def test_reduce_frame_covariance_into_partial_accumulates_poly_force_and_torque():
    partial = CovarianceChunkPartial()

    poly_force_1 = np.eye(3)
    poly_torque_1 = 2.0 * np.eye(3)

    poly_force_2 = 3.0 * np.eye(3)
    poly_torque_2 = 4.0 * np.eye(3)

    frame_out_1 = {
        "force": {"ua": {}, "res": {}, "poly": {0: poly_force_1}},
        "torque": {"ua": {}, "res": {}, "poly": {0: poly_torque_1}},
    }
    frame_out_2 = {
        "force": {"ua": {}, "res": {}, "poly": {0: poly_force_2}},
        "torque": {"ua": {}, "res": {}, "poly": {0: poly_torque_2}},
    }

    reduce_frame_covariance_into_partial(partial, frame_out_1)
    reduce_frame_covariance_into_partial(partial, frame_out_2)

    assert partial.frame_counts["poly"][0] == 2
    np.testing.assert_allclose(
        partial.force["poly"][0],
        2.0 * np.eye(3),
    )
    np.testing.assert_allclose(
        partial.torque["poly"][0],
        3.0 * np.eye(3),
    )


def test_reduce_frame_covariance_into_partial_accumulates_forcetorque_running_means():
    partial = CovarianceChunkPartial()

    res_ft_1 = np.ones((6, 6))
    res_ft_2 = 3.0 * np.ones((6, 6))

    poly_ft_1 = 2.0 * np.ones((6, 6))
    poly_ft_2 = 4.0 * np.ones((6, 6))

    frame_out_1 = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "forcetorque": {
            "res": {0: res_ft_1},
            "poly": {0: poly_ft_1},
        },
    }
    frame_out_2 = {
        "force": {"ua": {}, "res": {}, "poly": {}},
        "torque": {"ua": {}, "res": {}, "poly": {}},
        "forcetorque": {
            "res": {0: res_ft_2},
            "poly": {0: poly_ft_2},
        },
    }

    reduce_frame_covariance_into_partial(partial, frame_out_1)
    reduce_frame_covariance_into_partial(partial, frame_out_2)

    assert partial.forcetorque_counts["res"][0] == 2
    assert partial.forcetorque_counts["poly"][0] == 2

    np.testing.assert_allclose(
        partial.forcetorque["res"][0],
        2.0 * np.ones((6, 6)),
    )
    np.testing.assert_allclose(
        partial.forcetorque["poly"][0],
        3.0 * np.ones((6, 6)),
    )


def test_execute_frame_chunk_worker_handles_empty_chunk():
    worker_shared_data = {
        "reduced_universe": "universe",
        "levels": [["united_atom"]],
        "groups": {0: [0], 1: [1]},
        "frame_source": "frame-source",
        "args": SimpleNamespace(search_type="RAD"),
    }
    task = FrameChunkTask(chunk_index=4, frame_indices=())

    graph = MagicMock()
    neighbor_helper = MagicMock()

    with (
        patch("CodeEntropy.levels.execution.tasks.FrameGraph") as FrameGraph,
        patch("CodeEntropy.levels.execution.tasks.Neighbors") as Neighbors,
    ):
        FrameGraph.return_value.build.return_value = graph
        Neighbors.return_value = neighbor_helper

        result = execute_frame_chunk_worker(task, worker_shared_data)

    FrameGraph.assert_called_once_with(universe_operations=None)
    graph.execute_frame.assert_not_called()
    neighbor_helper.get_frame_neighbor_counts.assert_not_called()

    assert isinstance(result, FrameChunkResult)
    assert result.chunk_index == 4
    assert result.frame_indices == ()
    assert result.neighbor_totals == {0: 0, 1: 0}
    assert result.neighbor_samples == {0: 0, 1: 0}
    assert result.covariance_partial.force == {"ua": {}, "res": {}, "poly": {}}
    assert result.covariance_partial.torque == {"ua": {}, "res": {}, "poly": {}}


def test_execute_frame_chunk_worker_falls_back_to_universe_and_adds_new_neighbor():
    worker_shared_data = {
        "universe": "fallback-universe",
        "levels": [["united_atom"]],
        "groups": {0: [0]},
        "frame_source": "frame-source",
        "args": SimpleNamespace(search_type="grid"),
    }
    task = FrameChunkTask(chunk_index=2, frame_indices=("5",))

    graph = MagicMock()
    graph.execute_frame.return_value = _frame_covariance(1.0)

    neighbor_helper = MagicMock()
    neighbor_helper.get_frame_neighbor_counts.return_value = {
        99: (3, 2),
    }

    universe_operations = object()

    with (
        patch("CodeEntropy.levels.execution.tasks.FrameGraph") as FrameGraph,
        patch("CodeEntropy.levels.execution.tasks.Neighbors") as Neighbors,
    ):
        FrameGraph.return_value.build.return_value = graph
        Neighbors.return_value = neighbor_helper

        result = execute_frame_chunk_worker(
            task,
            worker_shared_data,
            universe_operations=universe_operations,
        )

    FrameGraph.assert_called_once_with(universe_operations=universe_operations)
    graph.execute_frame.assert_called_once_with(worker_shared_data, 5)

    neighbor_helper.get_frame_neighbor_counts.assert_called_once_with(
        universe="fallback-universe",
        levels=[["united_atom"]],
        groups={0: [0]},
        frame_source="frame-source",
        frame_index=5,
        search_type="grid",
    )

    assert result.chunk_index == 2
    assert result.frame_indices == ("5",)
    assert result.neighbor_totals == {0: 0, 99: 3}
    assert result.neighbor_samples == {0: 0, 99: 2}


def test_get_worker_frame_universe_prefers_frame_source_universe():
    frame_source = SimpleNamespace(universe="positioned-universe")
    worker_shared_data = {
        "frame_source": frame_source,
        "reduced_universe": "stale-reduced-universe",
        "universe": "fallback-universe",
    }

    result = get_worker_frame_universe(worker_shared_data)

    assert result == "positioned-universe"


def test_get_worker_frame_universe_falls_back_to_reduced_universe():
    worker_shared_data = {
        "frame_source": "mock-frame-source",
        "reduced_universe": "reduced-universe",
        "universe": "fallback-universe",
    }

    result = get_worker_frame_universe(worker_shared_data)

    assert result == "reduced-universe"


def test_get_worker_frame_universe_falls_back_to_universe():
    worker_shared_data = {
        "frame_source": "mock-frame-source",
        "universe": "fallback-universe",
    }

    result = get_worker_frame_universe(worker_shared_data)

    assert result == "fallback-universe"
