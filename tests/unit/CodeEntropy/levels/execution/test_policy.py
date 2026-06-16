"""Unit tests for internal frame execution policy."""

from __future__ import annotations

from types import SimpleNamespace

from CodeEntropy.levels.execution.policy import ExecutionPolicy


def test_requested_worker_count_uses_dask_workers_when_provided():
    policy = ExecutionPolicy()
    shared_data = {
        "args": SimpleNamespace(
            dask_workers=8,
            hpc=True,
            hpc_nodes=2,
            hpc_processes=4,
        )
    }

    assert policy.requested_worker_count(shared_data) == 8


def test_requested_worker_count_clamps_dask_workers_to_at_least_one():
    policy = ExecutionPolicy()
    shared_data = {"args": SimpleNamespace(dask_workers=0, hpc=False)}

    assert policy.requested_worker_count(shared_data) == 1


def test_requested_worker_count_uses_hpc_nodes_and_processes_without_local_workers():
    policy = ExecutionPolicy()
    shared_data = {
        "args": SimpleNamespace(
            dask_workers=None,
            hpc=True,
            hpc_nodes=3,
            hpc_processes=2,
        )
    }

    assert policy.requested_worker_count(shared_data) == 6


def test_requested_worker_count_clamps_hpc_values_to_at_least_one():
    policy = ExecutionPolicy()
    shared_data = {
        "args": SimpleNamespace(
            dask_workers=None,
            hpc=True,
            hpc_nodes=0,
            hpc_processes=0,
        )
    }

    assert policy.requested_worker_count(shared_data) == 1


def test_requested_worker_count_defaults_to_one_without_args():
    assert ExecutionPolicy().requested_worker_count({}) == 1


def test_requested_worker_count_defaults_to_one_for_non_parallel_run():
    policy = ExecutionPolicy()
    shared_data = {"args": SimpleNamespace(dask_workers=None, hpc=False)}

    assert policy.requested_worker_count(shared_data) == 1


def test_frame_chunk_size_is_deterministic_and_clamped():
    policy = ExecutionPolicy(
        target_frame_chunks_per_worker=2,
        min_frame_chunk_size=2,
        max_frame_chunk_size=10,
    )
    shared_data = {"args": SimpleNamespace(dask_workers=4, hpc=False)}

    assert policy.frame_chunk_size(shared_data, n_frames=100) == 10
    assert policy.frame_chunk_size(shared_data, n_frames=3) == 2


def test_frame_chunk_size_treats_zero_frames_as_one():
    policy = ExecutionPolicy(min_frame_chunk_size=1, max_frame_chunk_size=32)
    shared_data = {"args": SimpleNamespace(dask_workers=1, hpc=False)}

    assert policy.frame_chunk_size(shared_data, n_frames=0) == 1


def test_max_frame_in_flight_tasks_is_bounded_by_chunk_count():
    policy = ExecutionPolicy(max_frame_in_flight_multiplier=2)
    shared_data = {"args": SimpleNamespace(dask_workers=4, hpc=False)}

    assert policy.max_frame_in_flight_tasks(shared_data, n_chunks=3) == 3
    assert policy.max_frame_in_flight_tasks(shared_data, n_chunks=20) == 8
