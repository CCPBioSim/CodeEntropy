"""Tests for CodeEntropy HPC/Dask SLURM cluster helpers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from unittest import mock

import pytest

from CodeEntropy.core.dask_clusters import HPCDaskManager


@pytest.fixture(autouse=True)
def isolate_generated_scripts(tmp_path, monkeypatch):
    """Run each test in an isolated temporary working directory.

    HPCDaskManager writes generated SLURM scripts to the current working
    directory, so tests must not share CodeEntropy-master-submit.sh or
    dask-cluster-submit.sh.
    """
    monkeypatch.chdir(tmp_path)


def args_helper(args_list):
    """Build test args for HPCDaskManager."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--hpc-account", type=str, default="")
    parser.add_argument("--hpc-constraint", type=str, default="")
    parser.add_argument("--hpc-qos", type=str, default="")
    parser.add_argument("--hpc-queue", type=str, default="standard")
    parser.add_argument("--hpc-cores", type=int, default=20)
    parser.add_argument("--hpc-memory", type=str, default="16GB")
    parser.add_argument("--hpc-nodes", type=int, default=4)
    parser.add_argument("--hpc-processes", type=int, default=20)
    parser.add_argument("--hpc-walltime", type=str, default="24:00:00")
    parser.add_argument("--hpc-modules", nargs="+", default=None)

    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument("--conda-exec", type=str, default=None)
    parser.add_argument("--conda-path", type=str, default=None)

    return parser.parse_args(args_list)


def expected_slurm_cleanup_prologue():
    """Expected SLURM environment cleanup commands."""
    return [
        "unset SLURM_MEM_PER_CPU",
        "unset SLURM_MEM_PER_GPU",
        "unset SLURM_MEM_PER_NODE",
        "unset SLURM_CPU_BIND",
    ]


def test_check_slurm_env_removes_inherited_slurm_variables():
    args = args_helper([])
    manager = HPCDaskManager(args)

    os.environ["SLURM_CPU_BIND"] = "1"
    os.environ["SLURM_MEM_PER_CPU"] = "1000"
    os.environ["SLURM_MEM_PER_GPU"] = "1000"
    os.environ["SLURM_MEM_PER_NODE"] = "16G"

    manager.check_slurm_env()

    assert "SLURM_CPU_BIND" not in os.environ
    assert "SLURM_MEM_PER_CPU" not in os.environ
    assert "SLURM_MEM_PER_GPU" not in os.environ
    assert "SLURM_MEM_PER_NODE" not in os.environ


def test_resolve_conda_settings_uses_existing_args_when_present():
    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
        ]
    )
    manager = HPCDaskManager(args)

    manager.resolve_conda_settings()

    assert args.conda_env == "codeentropy"
    assert args.conda_exec == "conda"
    assert args.conda_path == "/path/to/conda"


def test_resolve_conda_settings_auto_detects_conda(monkeypatch):
    args = args_helper([])

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "CodeEntropy312")
    monkeypatch.setenv("CONDA_EXE", "/path/to/real/conda")
    monkeypatch.delenv("MAMBA_EXE", raising=False)

    manager = HPCDaskManager(args)

    manager.resolve_conda_settings()

    assert args.conda_env == "CodeEntropy312"
    assert args.conda_exec == "conda"
    assert args.conda_path == "/path/to/real/conda"


def test_resolve_conda_settings_auto_detects_mamba(monkeypatch):
    args = args_helper([])

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "CodeEntropy312")
    monkeypatch.setenv("CONDA_EXE", "/path/to/real/conda")
    monkeypatch.setenv("MAMBA_EXE", "/path/to/mamba")

    manager = HPCDaskManager(args)

    manager.resolve_conda_settings()

    assert args.conda_env == "CodeEntropy312"
    assert args.conda_exec == "mamba"
    assert args.conda_path == "/path/to/real/conda"


def test_resolve_conda_settings_raises_without_env(monkeypatch):
    args = args_helper([])

    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    manager = HPCDaskManager(args)

    with pytest.raises(SystemExit):
        manager.resolve_conda_settings()


def test_slurm_directives_account():
    args = args_helper(["--hpc-account", "c01"])
    manager = HPCDaskManager(args)

    extra, skip = manager.slurm_directives()

    assert extra == ["--account=c01"]
    assert skip == ["--mem"]


def test_slurm_directives_constraint():
    args = args_helper(["--hpc-constraint", "intel25"])
    manager = HPCDaskManager(args)

    extra, _skip = manager.slurm_directives()

    assert extra == ["--constraint=intel25"]


def test_slurm_directives_qos():
    args = args_helper(["--hpc-qos", "standard"])
    manager = HPCDaskManager(args)

    extra, _skip = manager.slurm_directives()

    assert extra == ["--qos=standard"]


def test_slurm_directives_all():
    args = args_helper(
        [
            "--hpc-account",
            "c01",
            "--hpc-qos",
            "standard",
            "--hpc-constraint",
            "intel25",
        ]
    )
    manager = HPCDaskManager(args)

    extra, skip = manager.slurm_directives()

    assert extra == [
        "--account=c01",
        "--qos=standard",
        "--constraint=intel25",
    ]
    assert skip == ["--mem"]


def test_slurm_prologues_conda():
    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
        ]
    )
    manager = HPCDaskManager(args)

    prologue = manager.slurm_prologues()

    assert prologue == [
        *expected_slurm_cleanup_prologue(),
        'eval "$(/path/to/conda shell.bash hook)"',
        "conda activate codeentropy",
        "export SLURM_CPU_FREQ_REQ=2250000",
    ]


def test_slurm_prologues_mamba():
    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "mamba",
            "--conda-path",
            "/path/to/conda",
        ]
    )
    manager = HPCDaskManager(args)

    prologue = manager.slurm_prologues()

    assert prologue == [
        *expected_slurm_cleanup_prologue(),
        'eval "$(/path/to/conda shell.bash hook)"',
        'eval "$(mamba shell hook --shell bash)"',
        "mamba activate codeentropy",
        "export SLURM_CPU_FREQ_REQ=2250000",
    ]


def test_slurm_prologues_includes_hpc_modules():
    args = args_helper(
        [
            "--hpc-modules",
            "apps/binapps/conda/miniforge3/25.9.1",
            "gcc/12.2.0",
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
        ]
    )
    manager = HPCDaskManager(args)

    prologue = manager.slurm_prologues()

    assert prologue == [
        "module load apps/binapps/conda/miniforge3/25.9.1",
        "module load gcc/12.2.0",
        *expected_slurm_cleanup_prologue(),
        'eval "$(/path/to/conda shell.bash hook)"',
        "conda activate codeentropy",
        "export SLURM_CPU_FREQ_REQ=2250000",
    ]


@mock.patch("psutil.net_if_addrs")
def test_system_network_interface_prefers_bond0(net_if_addrs):
    net_if_addrs.return_value = {"bond0": [], "ib0": [], "eth0": []}

    args = args_helper([])
    manager = HPCDaskManager(args)

    assert manager.system_network_interface() == "bond0"


@mock.patch("psutil.net_if_addrs")
def test_system_network_interface_prefers_ib0_when_bond0_missing(net_if_addrs):
    net_if_addrs.return_value = {"ib0": [], "eth0": []}

    args = args_helper([])
    manager = HPCDaskManager(args)

    assert manager.system_network_interface() == "ib0"


@mock.patch("psutil.net_if_addrs")
def test_system_network_interface_prefers_hsn0_when_bond0_and_ib0_missing(
    net_if_addrs,
):
    net_if_addrs.return_value = {"hsn0": [], "eth0": []}

    args = args_helper([])
    manager = HPCDaskManager(args)

    assert manager.system_network_interface() == "hsn0"


@mock.patch("psutil.net_if_addrs")
def test_system_network_interface_prefers_eth0_when_only_eth0_known_interface(
    net_if_addrs,
):
    net_if_addrs.return_value = {"eth0": [], "eno1": []}

    args = args_helper([])
    manager = HPCDaskManager(args)

    assert manager.system_network_interface() == "eth0"


@mock.patch("psutil.net_if_addrs")
def test_system_network_interface_raises_without_known_hpc_interface(net_if_addrs):
    net_if_addrs.return_value = {"lo": [], "docker0": [], "eno1": []}

    args = args_helper([])
    manager = HPCDaskManager(args)

    with pytest.raises(RuntimeError, match="Could not find a known HPC network"):
        manager.system_network_interface()


@mock.patch("subprocess.check_output")
def test_submit_master_writes_expected_script_conda(check_output):
    check_output.return_value = b"Submitted batch job 12345\n"

    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
            "--hpc-account",
            "c01-bio",
            "--hpc-qos",
            "standard",
            "--hpc-queue",
            "standard",
            "--hpc-walltime",
            "24:00:00",
        ]
    )
    manager = HPCDaskManager(args)

    cli = [
        "CodeEntropy",
        "--top_traj_file",
        "topology.tpr",
        "trajectory.trr",
        "--start",
        "0",
        "--end",
        "512",
        "--step",
        "1",
        "--hpc",
        "true",
        "--hpc_nodes",
        "4",
        "--submit",
        "true",
    ]

    with mock.patch.object(sys, "argv", cli):
        manager.submit_master()

    with open("CodeEntropy-master-submit.sh", encoding="utf-8") as file:
        script = file.read()

    assert script
    assert "#SBATCH --job-name=codeentropy-master" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --output=CodeEntropy-master-%j.out" in script
    assert "#SBATCH --error=CodeEntropy-master-%j.err" in script
    assert "#SBATCH --account=c01-bio" in script
    assert "#SBATCH --qos=standard" in script

    assert "unset SLURM_MEM_PER_CPU" in script
    assert "unset SLURM_MEM_PER_GPU" in script
    assert "unset SLURM_MEM_PER_NODE" in script
    assert "unset SLURM_CPU_BIND" in script

    assert 'eval "$(/path/to/conda shell.bash hook)"' in script
    assert "conda activate codeentropy" in script
    assert "export SLURM_CPU_FREQ_REQ=2250000" in script
    assert "export CODEENTROPY_SUBMITTED_JOB=1" in script

    assert "srun CodeEntropy" in script
    assert "--submit" not in script
    assert " --submit " not in script
    assert not script.rstrip().endswith(" true")

    check_output.assert_called_once_with(["sbatch", "CodeEntropy-master-submit.sh"])


@mock.patch("subprocess.check_output")
def test_submit_master_writes_expected_script_mamba(check_output):
    check_output.return_value = b"Submitted batch job 12345\n"

    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "mamba",
            "--conda-path",
            "/path/to/conda",
            "--hpc-account",
            "c01-bio",
            "--hpc-qos",
            "standard",
            "--hpc-queue",
            "standard",
        ]
    )
    manager = HPCDaskManager(args)

    cli = [
        "CodeEntropy",
        "--top_traj_file",
        "topology.tpr",
        "trajectory.trr",
        "--hpc",
        "true",
        "--submit",
        "true",
    ]

    with mock.patch.object(sys, "argv", cli):
        manager.submit_master()

    with open("CodeEntropy-master-submit.sh", encoding="utf-8") as file:
        script = file.read()

    assert script
    assert 'eval "$(/path/to/conda shell.bash hook)"' in script
    assert 'eval "$(mamba shell hook --shell bash)"' in script
    assert "mamba activate codeentropy" in script
    assert "export CODEENTROPY_SUBMITTED_JOB=1" in script
    assert "srun CodeEntropy" in script
    assert "--submit" not in script

    check_output.assert_called_once_with(["sbatch", "CodeEntropy-master-submit.sh"])


@mock.patch("subprocess.check_output")
def test_submit_master_writes_hpc_modules(check_output):
    check_output.return_value = b"Submitted batch job 12345\n"

    args = args_helper(
        [
            "--hpc-modules",
            "apps/binapps/conda/miniforge3/25.9.1",
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
            "--hpc-queue",
            "standard",
        ]
    )
    manager = HPCDaskManager(args)

    cli = [
        "CodeEntropy",
        "--top_traj_file",
        "topology.tpr",
        "trajectory.trr",
        "--hpc",
        "true",
        "--submit",
        "true",
    ]

    with mock.patch.object(sys, "argv", cli):
        manager.submit_master()

    with open("CodeEntropy-master-submit.sh", encoding="utf-8") as file:
        script = file.read()

    assert script
    assert "module load apps/binapps/conda/miniforge3/25.9.1" in script
    assert 'eval "$(/path/to/conda shell.bash hook)"' in script
    assert "conda activate codeentropy" in script
    assert "srun CodeEntropy" in script

    check_output.assert_called_once_with(["sbatch", "CodeEntropy-master-submit.sh"])


@mock.patch("CodeEntropy.core.dask_clusters.Client")
@mock.patch("CodeEntropy.core.dask_clusters.SLURMCluster")
@mock.patch.object(HPCDaskManager, "system_network_interface")
def test_configure_cluster_writes_job_script(
    system_network_interface,
    slurm_cluster,
    client,
):
    system_network_interface.return_value = "ib0"

    cluster_instance = mock.MagicMock()
    cluster_instance.job_script.return_value = "#!/bin/bash\n# dask worker script\n"
    slurm_cluster.return_value = cluster_instance

    client_instance = mock.MagicMock()
    client.return_value = client_instance

    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
            "--hpc-account",
            "c01-bio",
            "--hpc-qos",
            "standard",
            "--hpc-queue",
            "standard",
            "--hpc-cores",
            "8",
            "--hpc-processes",
            "1",
            "--hpc-memory",
            "16GB",
            "--hpc-nodes",
            "4",
            "--hpc-walltime",
            "02:00:00",
        ]
    )
    manager = HPCDaskManager(args)

    returned_client = manager.configure_cluster()

    assert returned_client is client_instance

    slurm_cluster.assert_called_once()
    _, kwargs = slurm_cluster.call_args

    assert kwargs["cores"] == 8
    assert kwargs["processes"] == 1
    assert kwargs["memory"] == "16GB"
    assert kwargs["queue"] == "standard"
    assert kwargs["job_directives_skip"] == ["--mem"]
    assert kwargs["job_extra_directives"] == [
        "--account=c01-bio",
        "--qos=standard",
    ]
    assert kwargs["python"] == "srun python"
    assert kwargs["walltime"] == "02:00:00"
    assert kwargs["shebang"] == "#!/bin/bash --login"
    assert kwargs["local_directory"] == "$PWD"
    assert kwargs["interface"] == "ib0"
    assert "scheduler_options" not in kwargs

    assert kwargs["job_script_prologue"] == [
        *expected_slurm_cleanup_prologue(),
        'eval "$(/path/to/conda shell.bash hook)"',
        "conda activate codeentropy",
        "export SLURM_CPU_FREQ_REQ=2250000",
    ]

    cluster_instance.scale.assert_called_once_with(jobs=4)
    client.assert_called_once_with(cluster_instance)

    with open("dask-cluster-submit.sh", encoding="utf-8") as file:
        script = file.read()

    assert script == "#!/bin/bash\n# dask worker script\n"


@mock.patch("subprocess.check_output")
def test_submit_master_prints_called_process_error_output(check_output, capsys):
    error_output = b"sbatch: error: invalid partition\n"

    check_output.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["sbatch", "CodeEntropy-master-submit.sh"],
        output=error_output,
    )

    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
            "--hpc-account",
            "c01-bio",
            "--hpc-qos",
            "standard",
            "--hpc-queue",
            "standard",
            "--hpc-walltime",
            "24:00:00",
        ]
    )
    manager = HPCDaskManager(args)

    cli = [
        "CodeEntropy",
        "--top_traj_file",
        "topology.tpr",
        "trajectory.trr",
        "--hpc",
        "true",
        "--submit",
        "true",
    ]

    with mock.patch.object(sys, "argv", cli):
        with pytest.raises(subprocess.CalledProcessError):
            manager.submit_master()

    captured = capsys.readouterr()

    assert "sbatch: error: invalid partition" in captured.out
    check_output.assert_called_once_with(["sbatch", "CodeEntropy-master-submit.sh"])


def test_conda_exec_raises_without_conda_or_mamba(monkeypatch):
    args = args_helper([])
    manager = HPCDaskManager(args)

    monkeypatch.delenv("MAMBA_EXE", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)

    with pytest.raises(SystemExit):
        manager._conda_exec()


def test_conda_path_raises_without_conda_exe(monkeypatch):
    args = args_helper([])
    manager = HPCDaskManager(args)

    monkeypatch.delenv("CONDA_EXE", raising=False)

    with pytest.raises(SystemExit):
        manager._conda_path()


@mock.patch("subprocess.check_output")
def test_submit_master_reraises_called_process_error(check_output):
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["sbatch", "CodeEntropy-master-submit.sh"],
        output=b"sbatch: error: invalid partition\n",
    )
    check_output.side_effect = error

    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
            "--hpc-queue",
            "standard",
            "--hpc-walltime",
            "24:00:00",
        ]
    )
    manager = HPCDaskManager(args)

    cli = [
        "CodeEntropy",
        "--top_traj_file",
        "topology.tpr",
        "trajectory.trr",
        "--hpc",
        "true",
        "--submit",
        "true",
    ]

    with mock.patch.object(sys, "argv", cli):
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            manager.submit_master()

    assert exc_info.value is error
    check_output.assert_called_once_with(["sbatch", "CodeEntropy-master-submit.sh"])


@mock.patch("subprocess.check_output")
def test_submit_master_writes_hpc_constraint(check_output):
    check_output.return_value = b"Submitted batch job 12345\n"

    args = args_helper(
        [
            "--conda-env",
            "codeentropy",
            "--conda-exec",
            "conda",
            "--conda-path",
            "/path/to/conda",
            "--hpc-queue",
            "standard",
            "--hpc-constraint",
            "intel25",
        ]
    )
    manager = HPCDaskManager(args)

    cli = [
        "CodeEntropy",
        "--top_traj_file",
        "topology.tpr",
        "trajectory.trr",
        "--hpc",
        "true",
        "--submit",
        "true",
    ]

    with mock.patch.object(sys, "argv", cli):
        manager.submit_master()

    with open("CodeEntropy-master-submit.sh", encoding="utf-8") as file:
        script = file.read()

    assert "#SBATCH --constraint=intel25" in script

    check_output.assert_called_once_with(["sbatch", "CodeEntropy-master-submit.sh"])
