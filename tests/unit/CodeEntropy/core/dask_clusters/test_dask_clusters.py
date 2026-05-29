"""Tests for CodeEntropy HPC/Dask SLURM cluster helpers."""

import argparse
import os
import subprocess
import sys
from unittest import mock

from CodeEntropy.core.dask_clusters import HPCDaskManager


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

    parser.add_argument("--conda-env", type=str, default="codeentropy")
    parser.add_argument("--conda-exec", type=str, default="conda")
    parser.add_argument("--conda-path", type=str, default="/path/to/conda")

    return parser.parse_args(args_list)


def test_check_slurm_env_removes_cpu_bind():
    args = args_helper([])
    manager = HPCDaskManager(args)

    os.environ["SLURM_CPU_BIND"] = "1"
    assert os.environ["SLURM_CPU_BIND"] == "1"

    manager.check_slurm_env()

    assert "SLURM_CPU_BIND" not in os.environ


def test_slurm_directives_account():
    args = args_helper(["--hpc-account", "c01"])
    manager = HPCDaskManager(args)

    extra, skip = manager.slurm_directives()

    assert extra == ['--account="c01"']
    assert skip == ["--mem"]


def test_slurm_directives_constraint():
    args = args_helper(["--hpc-constraint", "intel25"])
    manager = HPCDaskManager(args)

    extra, _skip = manager.slurm_directives()

    assert extra == ['--constraint="intel25"']


def test_slurm_directives_qos():
    args = args_helper(["--hpc-qos", "standard"])
    manager = HPCDaskManager(args)

    extra, _skip = manager.slurm_directives()

    assert extra == ['--qos="standard"']


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
        '--account="c01"',
        '--qos="standard"',
        '--constraint="intel25"',
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
        'eval "$(/path/to/conda shell.bash hook)"',
        'eval "$(mamba shell hook --shell bash)"',
        "mamba activate codeentropy",
        "export SLURM_CPU_FREQ_REQ=2250000",
    ]


@mock.patch("psutil.net_if_addrs")
def test_system_network_interface_prefers_ib0(net_if_addrs):
    net_if_addrs.return_value = {"ib0": [], "eth0": []}

    args = args_helper([])
    manager = HPCDaskManager(args)

    assert manager.system_network_interface() == "ib0"


@mock.patch("psutil.net_if_addrs")
def test_system_network_interface_fallback(net_if_addrs):
    net_if_addrs.return_value = {"lo": [], "docker0": []}

    args = args_helper([])
    manager = HPCDaskManager(args)

    assert manager.system_network_interface() == "lo"


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

    assert "#SBATCH --job-name=codeentropy-master" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --account=c01-bio" in script
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --qos=standard" in script
    assert 'eval "$(/path/to/conda shell.bash hook)"' in script
    assert "conda activate codeentropy" in script
    assert "srun CodeEntropy" in script
    assert "--submit" not in script
    assert "srun CodeEntropy" in script
    assert " --submit " not in script
    assert not script.rstrip().endswith(" true")

    os.remove("CodeEntropy-master-submit.sh")


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

    assert 'eval "$(/path/to/conda shell.bash hook)"' in script
    assert 'eval "$(mamba shell hook --shell bash)"' in script
    assert "mamba activate codeentropy" in script
    assert "srun CodeEntropy" in script
    assert "--submit" not in script

    os.remove("CodeEntropy-master-submit.sh")


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
    cluster_instance.scale.assert_called_once_with(jobs=4)
    client.assert_called_once_with(cluster_instance)

    with open("dask-cluster-submit.sh", encoding="utf-8") as file:
        script = file.read()

    assert script == "#!/bin/bash\n# dask worker script\n"

    os.remove("dask-cluster-submit.sh")


@mock.patch("subprocess.check_output")
def test_submit_master_prints_called_process_error_output(check_output, capsys):
    error_output = b"sbatch: error: invalid partition\n"

    check_output.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["bash", "-c", "sbatch CodeEntropy-master-submit.sh"],
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

    try:
        with mock.patch.object(sys, "argv", cli):
            manager.submit_master()

        captured = capsys.readouterr()

        assert "sbatch: error: invalid partition" in captured.out
        check_output.assert_called_once_with(
            ["bash", "-c", "sbatch CodeEntropy-master-submit.sh"]
        )

    finally:
        if os.path.exists("CodeEntropy-master-submit.sh"):
            os.remove("CodeEntropy-master-submit.sh")
