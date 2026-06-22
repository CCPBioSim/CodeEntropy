"""Unit tests for serial and Dask frame schedulers."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from CodeEntropy.levels.execution.policy import ExecutionPolicy
from CodeEntropy.levels.execution.scheduler import FrameScheduler
from CodeEntropy.levels.execution.tasks import (
    CovarianceChunkPartial,
    FrameChunkResult,
    FrameChunkTask,
)


def _scheduler(policy: ExecutionPolicy | MagicMock | None = None) -> FrameScheduler:
    return FrameScheduler(
        frame_dag=MagicMock(),
        policy=policy or ExecutionPolicy(),
        universe_operations=MagicMock(),
    )


def _chunk_result(chunk_index: int, frame_indices: tuple[int, ...]) -> FrameChunkResult:
    return FrameChunkResult(
        chunk_index=chunk_index,
        covariance_partial=CovarianceChunkPartial(),
        neighbor_totals={0: len(frame_indices)},
        neighbor_samples={0: len(frame_indices)},
        frame_indices=frame_indices,
    )


def test_execute_creates_progress_task_when_progress_is_supplied():
    scheduler = _scheduler()
    scheduler._run_serial = MagicMock()

    progress = MagicMock()
    progress.add_task.return_value = "task-id"

    scheduler.execute({}, frame_indices=[0], progress=progress)

    progress.add_task.assert_called_once_with(
        "[green]Frame processing",
        total=1,
        title="Initializing frame stage",
    )
    scheduler._run_serial.assert_called_once_with(
        {},
        frame_indices=[0],
        progress=progress,
        task="task-id",
    )


def test_execute_uses_dask_when_client_is_available_and_multiple_frames():
    scheduler = _scheduler()
    scheduler._run_dask = MagicMock()
    scheduler._run_serial = MagicMock()

    client = MagicMock()
    shared_data = {"dask_client": client, "parallel_frames": True}

    scheduler.execute(shared_data, frame_indices=[0, 1], progress=None)

    scheduler._run_dask.assert_called_once_with(
        shared_data,
        frame_indices=[0, 1],
        client=client,
        progress=None,
        task=None,
    )
    scheduler._run_serial.assert_not_called()


def test_execute_uses_serial_when_only_one_frame_even_with_client():
    scheduler = _scheduler()
    scheduler._run_dask = MagicMock()
    scheduler._run_serial = MagicMock()

    shared_data = {"dask_client": MagicMock(), "parallel_frames": True}

    scheduler.execute(shared_data, frame_indices=[0])

    scheduler._run_dask.assert_not_called()
    scheduler._run_serial.assert_called_once()


def test_execute_uses_serial_when_no_client():
    scheduler = _scheduler()
    scheduler._run_dask = MagicMock()
    scheduler._run_serial = MagicMock()

    shared_data = {"parallel_frames": True}

    scheduler.execute(shared_data, frame_indices=[0, 1])

    scheduler._run_dask.assert_not_called()
    scheduler._run_serial.assert_called_once()


def test_run_serial_executes_and_reduces_each_frame():
    scheduler = _scheduler()
    shared_data = {"groups": {0: [0]}}
    progress = MagicMock()
    task_id = "task-id"

    frame_out0 = {"covariance": "cov0", "neighbors": {0: (1, 1)}}
    frame_out1 = {"covariance": "cov1", "neighbors": {0: (2, 1)}}

    with patch(
        "CodeEntropy.levels.execution.scheduler.execute_frame_map_output",
        side_effect=[frame_out0, frame_out1],
    ) as execute_frame:
        scheduler._covariance_reducer.reduce_frame_map_output = MagicMock()

        scheduler._run_serial(
            shared_data,
            frame_indices=[0, 1],
            progress=progress,
            task=task_id,
        )

    assert execute_frame.call_count == 2
    scheduler._covariance_reducer.reduce_frame_map_output.assert_has_calls(
        [
            call(shared_data, frame_out0),
            call(shared_data, frame_out1),
        ]
    )
    progress.update.assert_has_calls(
        [
            call(task_id, title="Frame 0"),
            call(task_id, title="Frame 1"),
        ]
    )
    assert progress.advance.call_count == 2


def test_make_frame_chunk_tasks_uses_policy_chunk_size():
    policy = ExecutionPolicy(target_frame_chunks_per_worker=1)
    scheduler = _scheduler(policy=policy)
    shared_data = {"args": SimpleNamespace(dask_workers=2, hpc=False)}

    tasks = scheduler._make_frame_chunk_tasks(shared_data, [0, 1, 2, 3, 4])

    assert tasks == [
        FrameChunkTask(chunk_index=0, frame_indices=(0, 1, 2)),
        FrameChunkTask(chunk_index=1, frame_indices=(3, 4)),
    ]


def test_run_dask_scatters_worker_shared_once_and_reduces_in_chunk_order():
    policy = MagicMock()
    policy.max_frame_in_flight_tasks.return_value = 2
    scheduler = _scheduler(policy=policy)

    frame_tasks = [
        FrameChunkTask(chunk_index=0, frame_indices=(0,)),
        FrameChunkTask(chunk_index=1, frame_indices=(1,)),
    ]
    scheduler._make_frame_chunk_tasks = MagicMock(return_value=frame_tasks)
    scheduler._covariance_reducer.merge_chunk_partial = MagicMock()

    shared_data = {
        "groups": {0: [0]},
        "args": SimpleNamespace(dask_workers=2, hpc=False),
        "force_covariances": "parent-only",
        "frame_source": "kept",
    }

    worker_future = MagicMock(name="worker_shared_future")
    future_zero = MagicMock(name="future_zero")
    future_one = MagicMock(name="future_one")
    future_zero.result.return_value = _chunk_result(0, (0,))
    future_one.result.return_value = _chunk_result(1, (1,))

    client = MagicMock()
    client.scatter.return_value = [worker_future]
    client.submit.side_effect = [future_zero, future_one]

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.wait = MagicMock(return_value=({future_one, future_zero}, set()))

    with (
        patch.dict(sys.modules, {"distributed": fake_distributed}),
        patch(
            "CodeEntropy.levels.execution.scheduler.execute_frame_chunk_worker"
        ) as worker_func,
        patch(
            "CodeEntropy.levels.execution.scheduler.NeighborReducer.merge_chunk_partial"
        ) as merge_neighbors,
    ):
        scheduler._run_dask(
            shared_data,
            frame_indices=[0, 1],
            client=client,
            progress=None,
            task=None,
        )

    client.scatter.assert_called_once()
    scattered_payload = client.scatter.call_args.args[0]
    assert isinstance(scattered_payload, list)
    assert len(scattered_payload) == 1
    assert "force_covariances" not in scattered_payload[0]
    assert scattered_payload[0]["frame_source"] == "kept"

    client.submit.assert_has_calls(
        [
            call(
                worker_func,
                frame_tasks[0],
                worker_future,
                scheduler._universe_operations,
                pure=False,
            ),
            call(
                worker_func,
                frame_tasks[1],
                worker_future,
                scheduler._universe_operations,
                pure=False,
            ),
        ]
    )

    scheduler._covariance_reducer.merge_chunk_partial.assert_has_calls(
        [
            call(shared_data, future_zero.result.return_value.covariance_partial),
            call(shared_data, future_one.result.return_value.covariance_partial),
        ]
    )
    merge_neighbors.assert_has_calls(
        [
            call(shared_data, {0: 1}, {0: 1}),
            call(shared_data, {0: 1}, {0: 1}),
        ]
    )

    future_zero.release.assert_called_once()
    future_one.release.assert_called_once()
    worker_future.release.assert_called_once()


def test_run_dask_submits_more_tasks_as_futures_complete():
    policy = MagicMock()
    policy.max_frame_in_flight_tasks.return_value = 1
    scheduler = _scheduler(policy=policy)

    frame_tasks = [
        FrameChunkTask(chunk_index=0, frame_indices=(0,)),
        FrameChunkTask(chunk_index=1, frame_indices=(1,)),
    ]
    scheduler._make_frame_chunk_tasks = MagicMock(return_value=frame_tasks)
    scheduler._covariance_reducer.merge_chunk_partial = MagicMock()

    worker_future = MagicMock()
    future_zero = MagicMock()
    future_one = MagicMock()
    future_zero.result.return_value = _chunk_result(0, (0,))
    future_one.result.return_value = _chunk_result(1, (1,))

    client = MagicMock()
    client.scatter.return_value = [worker_future]
    client.submit.side_effect = [future_zero, future_one]

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.wait = MagicMock(
        side_effect=[
            ({future_zero}, set()),
            ({future_one}, set()),
        ]
    )

    progress = MagicMock()

    with (
        patch.dict(sys.modules, {"distributed": fake_distributed}),
        patch(
            "CodeEntropy.levels.execution.scheduler.NeighborReducer.merge_chunk_partial"
        ),
    ):
        scheduler._run_dask(
            {"groups": {0: [0]}, "args": SimpleNamespace(dask_workers=1, hpc=False)},
            frame_indices=[0, 1],
            client=client,
            progress=progress,
            task="task-id",
        )

    assert client.submit.call_count == 2
    assert progress.advance.call_count == 2
    worker_future.release.assert_called_once()


def test_run_dask_cancels_active_futures_and_releases_scattered_data_on_error():
    policy = MagicMock()
    policy.max_frame_in_flight_tasks.return_value = 1
    scheduler = _scheduler(policy=policy)

    task = FrameChunkTask(chunk_index=0, frame_indices=(0,))
    scheduler._make_frame_chunk_tasks = MagicMock(return_value=[task])

    worker_future = MagicMock()
    failed_future = MagicMock()
    failed_future.result.side_effect = RuntimeError("worker failed")

    client = MagicMock()
    client.scatter.return_value = [worker_future]
    client.submit.return_value = failed_future

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.wait = MagicMock(return_value=({failed_future}, set()))

    with patch.dict(sys.modules, {"distributed": fake_distributed}):
        with pytest.raises(RuntimeError, match="worker failed"):
            scheduler._run_dask(
                {
                    "groups": {0: [0]},
                    "args": SimpleNamespace(dask_workers=1, hpc=False),
                },
                frame_indices=[0],
                client=client,
                progress=None,
                task=None,
            )

    client.cancel.assert_called_once()
    worker_future.release.assert_called_once()


def test_run_dask_raises_when_distributed_is_missing():
    scheduler = _scheduler()
    client = MagicMock()

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "distributed":
            raise ImportError("No module named distributed")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="requires dask.distributed"):
            scheduler._run_dask(
                {
                    "groups": {0: [0]},
                    "args": SimpleNamespace(dask_workers=1, hpc=False),
                },
                frame_indices=[0],
                client=client,
                progress=None,
                task=None,
            )


def test_run_dask_raises_if_completed_frame_count_mismatches():
    policy = MagicMock()
    policy.max_frame_in_flight_tasks.return_value = 1
    scheduler = _scheduler(policy=policy)

    task = FrameChunkTask(chunk_index=0, frame_indices=(0,))
    scheduler._make_frame_chunk_tasks = MagicMock(return_value=[task])
    scheduler._covariance_reducer.merge_chunk_partial = MagicMock()

    worker_future = MagicMock()
    future = MagicMock()
    future.result.return_value = _chunk_result(0, (0,))

    client = MagicMock()
    client.scatter.return_value = [worker_future]
    client.submit.return_value = future

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.wait = MagicMock(return_value=({future}, set()))

    with patch.dict(sys.modules, {"distributed": fake_distributed}):
        with pytest.raises(
            RuntimeError,
            match="Parallel frame execution completed 1 frames, but expected 2",
        ):
            scheduler._run_dask(
                {
                    "groups": {0: [0]},
                    "args": SimpleNamespace(dask_workers=1, hpc=False),
                },
                frame_indices=[0, 1],
                client=client,
                progress=None,
                task=None,
            )

    worker_future.release.assert_called_once()
