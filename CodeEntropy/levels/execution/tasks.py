"""Task and worker-side helpers for frame-chunk execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from CodeEntropy.levels.frame_dag import FrameGraph
from CodeEntropy.levels.neighbors import Neighbors

FRAME_WORKER_EXCLUDED_SHARED_KEYS = {
    "force_covariances",
    "torque_covariances",
    "forcetorque_covariances",
    "frame_counts",
    "forcetorque_counts",
    "neighbor_totals",
    "neighbor_samples",
    "n_frames",
    "entropy_manager",
    "run_manager",
    "reporter",
    "dask_client",
}


@dataclass(frozen=True)
class FrameChunkTask:
    """MAP-stage input for a chunk of selected trajectory frames."""

    chunk_index: int
    frame_indices: tuple[int, ...]


@dataclass
class CovarianceChunkPartial:
    """Compact, mergeable covariance partial for one frame chunk."""

    force: dict[str, dict[Any, Any]] = field(
        default_factory=lambda: {"ua": {}, "res": {}, "poly": {}}
    )
    torque: dict[str, dict[Any, Any]] = field(
        default_factory=lambda: {"ua": {}, "res": {}, "poly": {}}
    )
    frame_counts: dict[str, dict[Any, int]] = field(
        default_factory=lambda: {"ua": {}, "res": {}, "poly": {}}
    )
    forcetorque: dict[str, dict[Any, Any]] = field(
        default_factory=lambda: {"res": {}, "poly": {}}
    )
    forcetorque_counts: dict[str, dict[Any, int]] = field(
        default_factory=lambda: {"res": {}, "poly": {}}
    )


@dataclass(frozen=True)
class FrameChunkResult:
    """MAP-stage output for a completed frame chunk."""

    chunk_index: int
    covariance_partial: CovarianceChunkPartial
    neighbor_totals: dict[int, int]
    neighbor_samples: dict[int, int]
    frame_indices: tuple[int, ...]


def make_frame_worker_shared_data(shared_data: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of shared data required by frame workers."""
    return {
        key: value
        for key, value in shared_data.items()
        if key not in FRAME_WORKER_EXCLUDED_SHARED_KEYS
    }


def incremental_mean_value(old: Any, new: Any, n: int) -> Any:
    """Compute a running mean value for worker-side chunk partials."""
    if old is None:
        return new.copy() if hasattr(new, "copy") else new
    return old + (new - old) / float(n)


def reduce_frame_covariance_into_partial(
    partial: CovarianceChunkPartial,
    frame_out: dict[str, Any],
) -> None:
    """Reduce one frame covariance payload into a chunk-local partial."""
    f_frame = frame_out["force"]
    t_frame = frame_out["torque"]

    for key, force_matrix in f_frame["ua"].items():
        partial.frame_counts["ua"][key] = partial.frame_counts["ua"].get(key, 0) + 1
        n = partial.frame_counts["ua"][key]
        partial.force["ua"][key] = incremental_mean_value(
            partial.force["ua"].get(key),
            force_matrix,
            n,
        )

    for key, torque_matrix in t_frame["ua"].items():
        if key not in partial.frame_counts["ua"]:
            partial.frame_counts["ua"][key] = partial.frame_counts["ua"].get(key, 0) + 1
        n = partial.frame_counts["ua"][key]
        partial.torque["ua"][key] = incremental_mean_value(
            partial.torque["ua"].get(key),
            torque_matrix,
            n,
        )

    for group_id, force_matrix in f_frame["res"].items():
        partial.frame_counts["res"][group_id] = (
            partial.frame_counts["res"].get(group_id, 0) + 1
        )
        n = partial.frame_counts["res"][group_id]
        partial.force["res"][group_id] = incremental_mean_value(
            partial.force["res"].get(group_id),
            force_matrix,
            n,
        )

    for group_id, torque_matrix in t_frame["res"].items():
        if group_id not in partial.frame_counts["res"]:
            partial.frame_counts["res"][group_id] = (
                partial.frame_counts["res"].get(group_id, 0) + 1
            )
        n = partial.frame_counts["res"][group_id]
        partial.torque["res"][group_id] = incremental_mean_value(
            partial.torque["res"].get(group_id),
            torque_matrix,
            n,
        )

    for group_id, force_matrix in f_frame["poly"].items():
        partial.frame_counts["poly"][group_id] = (
            partial.frame_counts["poly"].get(group_id, 0) + 1
        )
        n = partial.frame_counts["poly"][group_id]
        partial.force["poly"][group_id] = incremental_mean_value(
            partial.force["poly"].get(group_id),
            force_matrix,
            n,
        )

    for group_id, torque_matrix in t_frame["poly"].items():
        if group_id not in partial.frame_counts["poly"]:
            partial.frame_counts["poly"][group_id] = (
                partial.frame_counts["poly"].get(group_id, 0) + 1
            )
        n = partial.frame_counts["poly"][group_id]
        partial.torque["poly"][group_id] = incremental_mean_value(
            partial.torque["poly"].get(group_id),
            torque_matrix,
            n,
        )

    if "forcetorque" not in frame_out:
        return

    ft_frame = frame_out["forcetorque"]
    for group_id, matrix in ft_frame.get("res", {}).items():
        partial.forcetorque_counts["res"][group_id] = (
            partial.forcetorque_counts["res"].get(group_id, 0) + 1
        )
        n = partial.forcetorque_counts["res"][group_id]
        partial.forcetorque["res"][group_id] = incremental_mean_value(
            partial.forcetorque["res"].get(group_id),
            matrix,
            n,
        )

    for group_id, matrix in ft_frame.get("poly", {}).items():
        partial.forcetorque_counts["poly"][group_id] = (
            partial.forcetorque_counts["poly"].get(group_id, 0) + 1
        )
        n = partial.forcetorque_counts["poly"][group_id]
        partial.forcetorque["poly"][group_id] = incremental_mean_value(
            partial.forcetorque["poly"].get(group_id),
            matrix,
            n,
        )


def execute_frame_map_output(
    *,
    shared_data: dict[str, Any],
    frame_index: int,
    frame_dag: FrameGraph,
    neighbor_helper: Neighbors | None = None,
) -> dict[str, Any]:
    """Execute frame-local MAP operations for one frame."""
    frame_index = int(frame_index)
    frame_out: dict[str, Any] = {
        "covariance": frame_dag.execute_frame(shared_data, frame_index),
    }

    if neighbor_helper is None:
        neighbor_helper = Neighbors()

    universe = shared_data.get("reduced_universe", shared_data.get("universe"))
    frame_out["neighbors"] = neighbor_helper.get_frame_neighbor_counts(
        universe=universe,
        levels=shared_data["levels"],
        groups=shared_data["groups"],
        frame_source=shared_data["frame_source"],
        frame_index=frame_index,
        search_type=shared_data["args"].search_type,
    )

    return frame_out


def execute_frame_chunk_worker(
    task: FrameChunkTask,
    worker_shared_data: dict[str, Any],
    universe_operations: Any | None = None,
) -> FrameChunkResult:
    """Execute one frame chunk on a Dask worker and return compact partials."""
    frame_dag = FrameGraph(universe_operations=universe_operations).build()
    neighbor_helper = Neighbors()

    covariance_partial = CovarianceChunkPartial()
    neighbor_totals = {group_id: 0 for group_id in worker_shared_data["groups"].keys()}
    neighbor_samples = {group_id: 0 for group_id in worker_shared_data["groups"].keys()}

    for frame_index in task.frame_indices:
        frame_index = int(frame_index)
        frame_covariance = frame_dag.execute_frame(worker_shared_data, frame_index)
        reduce_frame_covariance_into_partial(covariance_partial, frame_covariance)

        universe = worker_shared_data.get(
            "reduced_universe",
            worker_shared_data.get("universe"),
        )
        frame_neighbors = neighbor_helper.get_frame_neighbor_counts(
            universe=universe,
            levels=worker_shared_data["levels"],
            groups=worker_shared_data["groups"],
            frame_source=worker_shared_data["frame_source"],
            frame_index=frame_index,
            search_type=worker_shared_data["args"].search_type,
        )

        for group_id, (count, sample_count) in frame_neighbors.items():
            neighbor_totals[group_id] = neighbor_totals.get(group_id, 0) + int(count)
            neighbor_samples[group_id] = neighbor_samples.get(group_id, 0) + int(
                sample_count
            )

    return FrameChunkResult(
        chunk_index=task.chunk_index,
        covariance_partial=covariance_partial,
        neighbor_totals=neighbor_totals,
        neighbor_samples=neighbor_samples,
        frame_indices=task.frame_indices,
    )
