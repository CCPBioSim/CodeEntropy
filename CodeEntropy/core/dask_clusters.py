"""
Helpers for setting up Dask clusters on HPC using SLURM.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys

import psutil
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

logger = logging.getLogger(__name__)


class HPCDaskManager:
    """
    Manage SLURM-backed Dask clusters and submission utilities for HPC environments.
    """

    def __init__(self, args):
        """
        Initialise HPCDaskManager with runtime arguments.

        Args:
            args: Parsed CLI arguments containing HPC and conda configuration.
        """
        self.args = args

    def _conda_env(self) -> str:
        """Determine the activated conda/mamba environment."""
        try:
            return os.environ["CONDA_DEFAULT_ENV"]
        except KeyError:
            logger.error("Please activate your conda/mamba environment.")
            raise SystemExit(1) from None

    def _conda_exec(self) -> str:
        """Determine whether conda or mamba should be used for activation."""
        if os.environ.get("MAMBA_EXE"):
            return "mamba"

        if os.environ.get("CONDA_EXE"):
            return "conda"

        logger.error(
            "Cannot determine your conda executable. "
            "Please make sure conda or mamba has been initialised."
        )
        raise SystemExit(1)

    def _conda_path(self) -> str:
        """Determine the path to the conda executable used for shell initialisation."""
        conda_exe = os.environ.get("CONDA_EXE")

        if conda_exe:
            return conda_exe

        logger.error("Please make sure conda is set up correctly.")
        raise SystemExit(1)

    def resolve_conda_settings(self) -> None:
        """
        Fill missing conda/mamba settings from the active environment.

        Explicit user-provided values are preserved. Auto-detection is only used
        when values are missing.
        """
        args = self.args

        if not getattr(args, "conda_env", None):
            args.conda_env = self._conda_env()

        if not getattr(args, "conda_exec", None):
            args.conda_exec = self._conda_exec()

        if not getattr(args, "conda_path", None) or args.conda_path == "conda":
            args.conda_path = self._conda_path()

    def check_slurm_env(self) -> None:
        """
        Remove inherited SLURM environment variables that can break nested srun calls.

        This is important when the master CodeEntropy process itself is already
        running inside a SLURM allocation and then launches Dask worker jobs.
        """
        for variable in (
            "SLURM_CPU_BIND",
            "SLURM_MEM_PER_CPU",
            "SLURM_MEM_PER_GPU",
            "SLURM_MEM_PER_NODE",
        ):
            os.environ.pop(variable, None)

    def system_network_interface(self) -> str:
        """
        Get the best candidate for the HPC network interface.

        This deliberately follows the WaterEntropy-style behaviour and only
        selects from known HPC-safe interfaces. It avoids selecting arbitrary
        interfaces such as eno1, which may exist on the master node but not on
        worker nodes.
        """
        hpc_nics = ["bond0", "ib0", "hsn0", "eth0"]
        interfaces = list(psutil.net_if_addrs().keys())

        for iface in hpc_nics:
            if iface in interfaces:
                return iface

        raise RuntimeError(
            "Could not find a known HPC network interface. "
            f"Available interfaces: {interfaces}. "
            "Expected one of: bond0, ib0, hsn0, eth0."
        )

    def slurm_directives(self) -> tuple[list[str], list[str]]:
        """
        Process additional SLURM directives and directives to skip.

        Returns:
            Tuple containing extra directives and skipped directives.
        """
        args = self.args
        extra: list[str] = []

        if args.hpc_account:
            extra.append(f"--account={args.hpc_account}")
        if args.hpc_qos:
            extra.append(f"--qos={args.hpc_qos}")
        if args.hpc_constraint:
            extra.append(f"--constraint={args.hpc_constraint}")

        skip = ["--mem"]

        return extra, skip

    def slurm_prologues(self) -> list[str]:
        """
        Build environment setup commands for the SLURM worker job script.

        Returns:
            List of shell commands executed before the Dask worker starts.
        """
        args = self.args
        prologue: list[str] = []

        for module_name in getattr(args, "hpc_modules", None) or []:
            prologue.append(f"module load {module_name}")

        prologue.append("unset SLURM_MEM_PER_CPU")
        prologue.append("unset SLURM_MEM_PER_GPU")
        prologue.append("unset SLURM_MEM_PER_NODE")
        prologue.append("unset SLURM_CPU_BIND")

        prologue.append(f'eval "$({args.conda_path} shell.bash hook)"')

        if args.conda_exec == "mamba":
            prologue.append(f'eval "$({args.conda_exec} shell hook --shell bash)"')

        prologue.append(f"{args.conda_exec} activate {args.conda_env}")
        prologue.append("export SLURM_CPU_FREQ_REQ=2250000")

        return prologue

    def configure_cluster(self) -> Client:
        """
        Configure a SLURM-backed Dask cluster.

        Returns:
            Dask distributed client connected to the SLURMCluster.
        """
        args = self.args

        self.resolve_conda_settings()

        extra, skip = self.slurm_directives()
        prologue = self.slurm_prologues()
        iface = self.system_network_interface()

        self.check_slurm_env()

        cluster = SLURMCluster(
            cores=args.hpc_cores,
            processes=args.hpc_processes,
            memory=args.hpc_memory,
            queue=args.hpc_queue,
            job_directives_skip=skip,
            job_extra_directives=extra,
            python="srun python",
            walltime=args.hpc_walltime,
            shebang="#!/bin/bash --login",
            local_directory="$PWD",
            interface=iface,
            job_script_prologue=prologue,
        )

        cluster.scale(jobs=args.hpc_nodes)

        client = Client(cluster)

        with open("dask-cluster-submit.sh", "w", encoding="utf-8") as f:
            f.write(cluster.job_script())

        return client

    def submit_master(self) -> None:
        """
        Submit a SLURM job that runs the master CodeEntropy process.

        This generates a temporary SLURM script and submits it via sbatch.
        """
        self.resolve_conda_settings()

        cli = list(sys.argv[1:])

        if "--submit" in cli:
            idx = cli.index("--submit")
            cli.pop(idx)

            if idx < len(cli) and str(cli[idx]).lower() in {"true", "false"}:
                cli.pop(idx)

        script_name = "CodeEntropy-master-submit.sh"

        with open(script_name, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash --login\n\n")
            f.write("#SBATCH --job-name=codeentropy-master\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --ntasks=1\n")
            f.write("#SBATCH --cpus-per-task=2\n")
            f.write(f"#SBATCH --time={self.args.hpc_walltime}\n")
            f.write(f"#SBATCH --partition={self.args.hpc_queue}\n")
            f.write("#SBATCH --output=CodeEntropy-master-%j.out\n")
            f.write("#SBATCH --error=CodeEntropy-master-%j.err\n")

            if self.args.hpc_account:
                f.write(f"#SBATCH --account={self.args.hpc_account}\n")

            if self.args.hpc_qos:
                f.write(f"#SBATCH --qos={self.args.hpc_qos}\n")

            if self.args.hpc_constraint:
                f.write(f"#SBATCH --constraint={self.args.hpc_constraint}\n")

            f.write("\n")

            for module_name in getattr(self.args, "hpc_modules", None) or []:
                f.write(f"module load {module_name}\n")

            f.write("unset SLURM_MEM_PER_CPU\n")
            f.write("unset SLURM_MEM_PER_GPU\n")
            f.write("unset SLURM_MEM_PER_NODE\n")
            f.write("unset SLURM_CPU_BIND\n")

            f.write(f'eval "$({self.args.conda_path} shell.bash hook)"\n')

            if self.args.conda_exec == "mamba":
                f.write(f'eval "$({self.args.conda_exec} shell hook --shell bash)"\n')

            f.write(f"{self.args.conda_exec} activate {self.args.conda_env}\n")
            f.write("export SLURM_CPU_FREQ_REQ=2250000\n")
            f.write("export CODEENTROPY_SUBMITTED_JOB=1\n\n")

            command = " ".join(["srun", "CodeEntropy", shlex.join(cli)])
            f.write(f"{command}\n")

        self.check_slurm_env()

        try:
            result = subprocess.check_output(["sbatch", script_name])
            print(result.decode("utf-8"))
        except subprocess.CalledProcessError as exc:
            print(exc.output.decode("utf-8", errors="replace"))
            raise
