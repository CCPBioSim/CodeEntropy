"""Serial and Dask schedulers for frame-chunk map-reduce execution."""

from __future__ import annotations

from typing import Any

from rich.progress import TaskID

from CodeEntropy.levels.execution.chunks import chunk_frame_indices
from CodeEntropy.levels.execution.policy import ExecutionPolicy
from CodeEntropy.levels.execution.reducers import CovarianceReducer, NeighborReducer
from CodeEntropy.levels.execution.tasks import (
    FrameChunkResult,
    FrameChunkTask,
    execute_frame_chunk_worker,
    execute_frame_map_output,
    make_frame_worker_shared_data,
)
from CodeEntropy.levels.frame_dag import FrameGraph
from CodeEntropy.levels.neighbors import Neighbors
from CodeEntropy.results.reporter import _RichProgressSink


class FrameScheduler:
    """Execute frame-local MAP work serially or through Dask.

    Dask futures may complete in any order, but completed chunks are reduced by
    chunk index. This keeps parent-side floating-point reductions deterministic.
    """

    def __init__(
        self,
        *,
        frame_dag: FrameGraph,
        policy: ExecutionPolicy,
        universe_operations: Any | None = None,
    ) -> None:
        self._frame_dag = frame_dag
        self._policy = policy
        self._universe_operations = universe_operations
        self._covariance_reducer = CovarianceReducer()

    def execute(
        self,
        shared_data: dict[str, Any],
        *,
        frame_indices: list[int],
        progress: _RichProgressSink | None = None,
    ) -> None:
        """Execute frame-local MAP work and reduce it into ``shared_data``."""
        task: TaskID | None = None
        if progress is not None:
            task = progress.add_task(
                "[green]Frame processing",
                total=len(frame_indices),
                title="Initializing frame stage",
            )

        client = shared_data.get("dask_client")
        parallel_frames = bool(shared_data.get("parallel_frames", client is not None))

        if parallel_frames and client is not None and len(frame_indices) > 1:
            self._run_dask(
                shared_data,
                frame_indices=frame_indices,
                client=client,
                progress=progress,
                task=task,
            )
            return

        self._run_serial(
            shared_data,
            frame_indices=frame_indices,
            progress=progress,
            task=task,
        )

    def _run_serial(
        self,
        shared_data: dict[str, Any],
        *,
        frame_indices: list[int],
        progress: _RichProgressSink | None,
        task: TaskID | None,
    ) -> None:
        """Execute frame-local MAP work serially and reduce immediately."""
        neighbor_helper = Neighbors()

        for frame_index in frame_indices:
            if progress is not None and task is not None:
                progress.update(task, title=f"Frame {frame_index}")

            frame_out = execute_frame_map_output(
                shared_data=shared_data,
                frame_index=frame_index,
                frame_dag=self._frame_dag,
                neighbor_helper=neighbor_helper,
            )
            self._covariance_reducer.reduce_frame_map_output(shared_data, frame_out)

            if progress is not None and task is not None:
                progress.advance(task)

    def _run_dask(
        self,
        shared_data: dict[str, Any],
        *,
        frame_indices: list[int],
        client: Any,
        progress: _RichProgressSink | None,
        task: TaskID | None,
    ) -> None:
        """Execute frame-chunk MAP tasks with bounded deterministic reduction."""
        try:
            from distributed import wait
        except ImportError as exc:
            raise RuntimeError(
                "Parallel frame execution requires dask.distributed to be installed."
            ) from exc

        frame_tasks = self._make_frame_chunk_tasks(shared_data, frame_indices)
        max_in_flight = self._policy.max_frame_in_flight_tasks(
            shared_data,
            n_chunks=len(frame_tasks),
        )
        worker_shared = make_frame_worker_shared_data(shared_data)
        worker_shared_future = client.scatter(
            [worker_shared],
            broadcast=True,
            hash=False,
        )[0]

        frame_task_iter = iter(frame_tasks)
        active_futures: set[Any] = set()
        submitted = 0
        completed = 0
        next_reduce_index = 0
        pending_results: dict[int, FrameChunkResult] = {}

        def submit_next() -> bool:
            nonlocal submitted
            try:
                frame_task = next(frame_task_iter)
            except StopIteration:
                return False

            future = client.submit(
                execute_frame_chunk_worker,
                frame_task,
                worker_shared_future,
                self._universe_operations,
                pure=False,
            )
            active_futures.add(future)
            submitted += 1
            return True

        def reduce_ready_results() -> None:
            nonlocal completed, next_reduce_index
            while next_reduce_index in pending_results:
                chunk_result = pending_results.pop(next_reduce_index)

                self._covariance_reducer.merge_chunk_partial(
                    shared_data,
                    chunk_result.covariance_partial,
                )
                NeighborReducer.merge_chunk_partial(
                    shared_data,
                    chunk_result.neighbor_totals,
                    chunk_result.neighbor_samples,
                )

                completed += len(chunk_result.frame_indices)
                next_reduce_index += 1

                if progress is not None and task is not None:
                    progress.advance(task, len(chunk_result.frame_indices))

        try:
            for _ in range(min(max_in_flight, len(frame_tasks))):
                submit_next()

            if progress is not None and task is not None:
                progress.update(
                    task,
                    title=f"Submitted {submitted} of {len(frame_tasks)} frame chunks",
                )

            while active_futures:
                if progress is not None and task is not None and completed == 0:
                    progress.update(task, title="Waiting for first frame chunk")

                done, not_done = wait(
                    active_futures,
                    return_when="FIRST_COMPLETED",
                )
                active_futures = set(not_done)

                for future in done:
                    chunk_result = future.result()
                    pending_results[chunk_result.chunk_index] = chunk_result
                    future.release()

                    if submit_next() and progress is not None and task is not None:
                        progress.update(
                            task,
                            title=(
                                f"Submitted {submitted} of {len(frame_tasks)} "
                                "frame chunks"
                            ),
                        )

                reduce_ready_results()

            reduce_ready_results()

            if completed != len(frame_indices):
                raise RuntimeError(
                    f"Parallel frame execution completed {completed} frames, "
                    f"but expected {len(frame_indices)}."
                )

        except Exception:
            client.cancel(list(active_futures))
            raise
        finally:
            worker_shared_future.release()

    def _make_frame_chunk_tasks(
        self,
        shared_data: dict[str, Any],
        frame_indices: list[int],
    ) -> list[FrameChunkTask]:
        """Build explicit frame-chunk MAP tasks."""
        chunk_size = self._policy.frame_chunk_size(
            shared_data,
            n_frames=len(frame_indices),
        )
        frame_chunks = chunk_frame_indices(frame_indices, chunk_size)

        return [
            FrameChunkTask(
                chunk_index=chunk_index,
                frame_indices=chunk,
            )
            for chunk_index, chunk in enumerate(frame_chunks)
        ]
