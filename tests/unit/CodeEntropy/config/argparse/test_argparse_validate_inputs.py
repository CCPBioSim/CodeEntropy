import logging

import pytest


def test_validate_inputs_valid_does_not_raise(resolver, dummy_universe, make_args):
    args = make_args()

    resolver.validate_inputs(dummy_universe, args)


def test_check_input_start_raises_when_start_exceeds_trajectory(resolver, make_args):
    u = type("U", (), {"trajectory": [None] * 10})()
    args = make_args(start=11)

    with pytest.raises(ValueError):
        resolver._check_input_start(u, args)


def test_check_input_end_raises_when_end_exceeds_trajectory(resolver, make_args):
    u = type("U", (), {"trajectory": [None] * 10})()
    args = make_args(end=11)

    with pytest.raises(ValueError):
        resolver._check_input_end(u, args)


def test_check_input_step_negative_logs_warning(resolver, make_args, caplog):
    args = make_args(step=-1)

    with caplog.at_level(logging.WARNING):
        resolver._check_input_step(args)

    assert "Negative 'step' value" in caplog.text


@pytest.mark.parametrize("bin_width", [-1, 361])
def test_check_input_bin_width_out_of_range_raises(resolver, make_args, bin_width):
    args = make_args(bin_width=bin_width)

    with pytest.raises(ValueError):
        resolver._check_input_bin_width(args)


def test_check_input_temperature_negative_raises(resolver, make_args):
    args = make_args(temperature=-0.1)

    with pytest.raises(ValueError):
        resolver._check_input_temperature(args)


def test_check_input_force_partitioning_non_default_logs_warning(
    resolver, make_args, caplog
):
    args = make_args(force_partitioning=0.7)

    with caplog.at_level(logging.WARNING):
        resolver._check_input_force_partitioning(args)

    assert "differs from the default" in caplog.text


def test_check_parallel_frame_options_valid_local_dask_does_not_raise(
    resolver, make_args
):
    args = make_args(
        dask_workers=2,
        dask_threads_per_worker=1,
        hpc=False,
        submit=False,
    )

    resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_allows_dask_workers_none(resolver, make_args):
    args = make_args(
        dask_workers=None,
        dask_threads_per_worker=1,
        hpc=False,
        submit=False,
    )

    resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_valid_hpc_does_not_raise(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args()

    resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_valid_hpc_submit_does_not_raise(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(submit=True)

    resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_dask_workers_less_than_one(
    resolver, make_args
):
    args = make_args(
        dask_workers=0,
        dask_threads_per_worker=1,
        hpc=False,
        submit=False,
    )

    with pytest.raises(
        ValueError,
        match="'dask_workers' must be at least 1 if provided.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_dask_threads_less_than_one(
    resolver, make_args
):
    args = make_args(
        dask_workers=None,
        dask_threads_per_worker=0,
        hpc=False,
        submit=False,
    )

    with pytest.raises(
        ValueError,
        match="'dask_threads_per_worker' must be at least 1.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_submit_without_hpc(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        hpc=False,
        submit=True,
    )

    with pytest.raises(
        ValueError,
        match="'submit' requires 'hpc' to be enabled.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_hpc_queue_missing(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        hpc_queue=None,
    )

    with pytest.raises(
        ValueError,
        match="'hpc_queue' must be provided when using HPC Dask.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_hpc_nodes_less_than_one(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        hpc_nodes=0,
    )

    with pytest.raises(
        ValueError,
        match="'hpc_nodes' must be at least 1.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_hpc_cores_less_than_one(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        hpc_cores=0,
    )

    with pytest.raises(
        ValueError,
        match="'hpc_cores' must be at least 1.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_hpc_processes_less_than_one(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        hpc_processes=0,
    )

    with pytest.raises(
        ValueError,
        match="'hpc_processes' must be at least 1.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_hpc_memory_missing(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        hpc_memory=None,
    )

    with pytest.raises(
        ValueError,
        match="'hpc_memory' must be provided when using HPC Dask.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_hpc_walltime_missing(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        hpc_walltime=None,
    )

    with pytest.raises(
        ValueError,
        match="'hpc_walltime' must be provided when using HPC Dask.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_conda_env_missing(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        conda_env=None,
    )

    with pytest.raises(
        ValueError,
        match="'conda_env' must be provided when using HPC Dask.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_conda_path_missing(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        conda_path=None,
    )

    with pytest.raises(
        ValueError,
        match="'conda_path' must be provided when using HPC Dask.",
    ):
        resolver._check_parallel_frame_options(args)


def test_check_parallel_frame_options_raises_when_conda_exec_missing(
    resolver, make_valid_hpc_args
):
    args = make_valid_hpc_args(
        conda_exec=None,
    )

    with pytest.raises(
        ValueError,
        match="'conda_exec' must be provided when using HPC Dask.",
    ):
        resolver._check_parallel_frame_options(args)
