from types import SimpleNamespace

import pytest

from CodeEntropy.config.argparse import ConfigResolver


class DummyUniverse:
    """Minimal MDAnalysis-like Universe stub for validate_inputs tests."""

    def __init__(self, length: int):
        self.trajectory = [None] * length


@pytest.fixture()
def resolver():
    return ConfigResolver()


@pytest.fixture()
def dummy_universe():
    # default length used in many tests
    return DummyUniverse(length=100)


@pytest.fixture()
def make_args():
    """Factory to build an args-like object with defaults used by validation checks."""

    def _make(**overrides):
        base = dict(
            start=0,
            end=10,
            step=1,
            bin_width=30,
            temperature=298.0,
            force_partitioning=0.5,
        )
        base.update(overrides)
        # validation functions only require attribute access; SimpleNamespace is ideal
        return SimpleNamespace(**base)

    return _make


@pytest.fixture()
def make_valid_hpc_args(make_args):
    """Factory to build a valid HPC/Dask args object for validation tests."""

    def _make(**overrides):
        base = dict(
            dask_workers=None,
            dask_threads_per_worker=1,
            hpc=True,
            submit=False,
            hpc_queue="standard",
            hpc_nodes=1,
            hpc_cores=1,
            hpc_processes=1,
            hpc_memory="4GB",
            hpc_walltime="01:00:00",
            conda_env="codeentropy",
            conda_path="conda",
            conda_exec="conda",
        )
        base.update(overrides)
        return make_args(**base)

    return _make


@pytest.fixture()
def empty_cli_args(resolver):
    """Argparse Namespace with all parser defaults."""
    parser = resolver.build_parser()
    return parser.parse_args([])
