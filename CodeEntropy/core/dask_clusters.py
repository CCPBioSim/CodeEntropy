"""
Helpers for setting up Dask clusters on HPC using SLURM.
"""

import os
import subprocess
import sys

import psutil
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


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

    def check_slurm_env(self) -> None:
        """
        Remove SLURM_CPU_BIND from environment if present.

        Some HPC systems require this variable to be unset for correct CPU binding.
        """
        os.environ.pop("SLURM_CPU_BIND", None)

    def system_network_interface(self) -> str:
        """
        Select the most appropriate network interface for HPC communication.

        If args.hpc_interface is provided, that value is used directly. Otherwise,
        commonly used HPC interfaces are preferred. Loopback and container interfaces
        are avoided because Dask workers on other nodes cannot connect to a scheduler
        advertised on 127.0.0.1.

        Returns:
            str: Name of selected network interface.

        Raises:
            RuntimeError: If no suitable non-loopback interface can be found.
        """
        configured = getattr(self.args, "hpc_interface", None)
        if configured:
            return configured

        preferred_nics = ["bond0", "ib0", "hsn0", "eth0"]
        interfaces = list(psutil.net_if_addrs().keys())

        for iface in preferred_nics:
            if iface in interfaces:
                return iface

        for iface in interfaces:
            if not iface.startswith(("lo", "docker", "veth")):
                return iface

        raise RuntimeError(
            "Could not find a non-loopback network interface for Dask workers. "
            f"Available interfaces: {interfaces}. Set 'hpc_interface' in config.yaml."
        )

    def slurm_directives(self) -> tuple[list[str], list[str]]:
        """
        Build SLURM job directives and skip list.

        Returns:
            Tuple[List[str], List[str]]:
                - Extra SLURM directives
                - Directives to skip
        """
        args = self.args
        extra: list[str] = []
        skip: list[str] = ["--mem"]

        if args.hpc_account:
            extra.append(f'--account="{args.hpc_account}"')
        if args.hpc_qos:
            extra.append(f'--qos="{args.hpc_qos}"')
        if args.hpc_constraint:
            extra.append(f'--constraint="{args.hpc_constraint}"')

        return extra, skip

    def slurm_prologues(self) -> list[str]:
        """
        Build SLURM job prologue commands for environment setup.

        Returns:
            List[str]: Shell commands executed before job start.
        """
        args = self.args
        prologue: list[str] = []

        for module_name in getattr(args, "hpc_modules", None) or []:
            prologue.append(f"module load {module_name}")

        prologue.append(f'eval "$({args.conda_path} shell.bash hook)"')

        if args.conda_exec == "mamba":
            prologue.append(f'eval "$({args.conda_exec} shell hook --shell bash)"')

        prologue.append(f"{args.conda_exec} activate {args.conda_env}")
        prologue.append("export SLURM_CPU_FREQ_REQ=2250000")

        return prologue

    def configure_cluster(self) -> Client:
        """
        Configure and launch a SLURM-backed Dask cluster.

        Returns:
            Client: Dask distributed client connected to cluster.
        """
        args = self.args

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
            scheduler_options={"interface": iface},
            job_script_prologue=prologue,
        )

        cluster.scale(jobs=args.hpc_nodes)

        client = Client(cluster)

        with open("dask-cluster-submit.sh", "w", encoding="utf-8") as f:
            f.write(cluster.job_script())

        return client

    def submit_master(self) -> None:
        """
        Submit a SLURM job that runs a master Dask orchestration process.

        This generates a temporary SLURM script and submits it via `sbatch`.
        """
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

            if self.args.hpc_account:
                f.write(f"#SBATCH --account={self.args.hpc_account}\n")

            f.write(f"#SBATCH --partition={self.args.hpc_queue}\n")

            if self.args.hpc_qos:
                f.write(f"#SBATCH --qos={self.args.hpc_qos}\n")

            f.write("\n")

            for module_name in getattr(self.args, "hpc_modules", None) or []:
                f.write(f"module load {module_name}\n")

            f.write(f'eval "$({self.args.conda_path} shell.bash hook)"\n')

            if self.args.conda_exec == "mamba":
                f.write(f'eval "$({self.args.conda_exec} shell hook --shell bash)"\n')

            f.write(f"{self.args.conda_exec} activate {self.args.conda_env}\n\n")
            f.write(f"srun CodeEntropy {' '.join(cli)}")

        self.check_slurm_env()

        try:
            result = subprocess.check_output(["bash", "-c", f"sbatch {script_name}"])
            print(result.decode("utf-8"))
        except subprocess.CalledProcessError as e:
            print(e.output)
