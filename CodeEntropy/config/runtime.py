"""Run orchestration for CodeEntropy.

This module provides the RunManager, which is responsible for:
- Creating a new job folder for each run
- Loading YAML configuration and merging it with CLI arguments
- Setting up logging and displaying a Rich splash screen
- Building the MDAnalysis Universe (including optional force merging)
- Wiring dependencies and executing the EntropyManager workflow
- Providing physical-constants helpers used by entropy calculations

Notes on design:
- RunManager focuses on orchestration and simple utilities only.
- Computational logic lives in EntropyManager and the level/entropy DAG modules.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Optional

import MDAnalysis as mda
import requests
import yaml
from art import text2art
from rich.align import Align
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from CodeEntropy.config.argparse import ConfigManager
from CodeEntropy.core.logging import LoggingConfig
from CodeEntropy.entropy.manager import EntropyManager
from CodeEntropy.levels.dihedrals import DihedralAnalysis
from CodeEntropy.levels.mda import UniverseOperations
from CodeEntropy.molecules.grouping import GroupMolecules
from CodeEntropy.results.reporter import DataLogger

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class RunManager:
    """Coordinate setup and execution of entropy analysis runs.

    Responsibilities:
      - Bootstrapping: job folder, logging, splash screen
      - Configuration: YAML loading + CLI parsing + merge and validation
      - Universe creation: MDAnalysis Universe (optionally merging forces)
      - Dependency wiring and execution: EntropyManager
      - Utilities used by downstream modules: constants and unit conversions

    Attributes:
        folder: Working directory for the current job (e.g., job001).
    """

    _N_AVOGADRO = 6.0221415e23
    _DEF_TEMPER = 298

    def __init__(self, folder: str) -> None:
        """Initialize a RunManager for a given working folder.

        This sets up configuration helpers, data logging, and logging configuration.
        It also defines physical constants used in entropy calculations.

        Args:
            folder: Job folder path where logs and outputs will be written.
        """
        self.folder = folder
        self._config_manager = ConfigManager()
        self._data_logger = DataLogger()
        self._logging_config = LoggingConfig(folder)

    @property
    def N_AVOGADRO(self) -> float:
        """Return Avogadro's number used in entropy calculations."""
        return self._N_AVOGADRO

    @property
    def DEF_TEMPER(self) -> float:
        """Return the default temperature (K) used in the analysis."""
        return self._DEF_TEMPER

    @staticmethod
    def create_job_folder() -> str:
        """Create a new job folder (job###) in the current working directory.

        The method searches existing folders that start with "job" and picks the next
        integer suffix. If none exist, it creates job001.

        Returns:
            The full path to the newly created job folder.
        """
        current_dir = os.getcwd()
        existing_folders = [f for f in os.listdir(current_dir) if f.startswith("job")]

        job_numbers = []
        for folder in existing_folders:
            try:
                job_numbers.append(int(folder[3:]))
            except ValueError:
                continue

        next_job_number = 1 if not job_numbers else max(job_numbers) + 1
        new_job_folder = f"job{next_job_number:03d}"
        new_folder_path = os.path.join(current_dir, new_job_folder)
        os.makedirs(new_folder_path, exist_ok=True)

        return new_folder_path

    def load_citation_data(self) -> Optional[Dict[str, Any]]:
        """Load CITATION.cff from GitHub.

        If the request fails (offline, blocked, etc.), returns None.

        Returns:
            Parsed CITATION.cff content as a dict, or None if unavailable.
        """
        url = (
            "https://raw.githubusercontent.com/CCPBioSim/"
            "CodeEntropy/refs/heads/main/CITATION.cff"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return yaml.safe_load(response.text)
        except requests.exceptions.RequestException:
            return None

    def show_splash(self) -> None:
        """Render a Rich splash screen with optional citation metadata."""
        citation = self.load_citation_data()

        if citation:
            ascii_title = text2art(citation.get("title", "CodeEntropy"))
            ascii_render = Align.center(Text(ascii_title, style="bold white"))

            version = citation.get("version", "?")
            release_date = citation.get("date-released", "?")
            url = citation.get("url", citation.get("repository-code", ""))

            version_text = Align.center(
                Text(f"Version {version} | Released {release_date}", style="green")
            )
            url_text = Align.center(Text(url, style="blue underline"))

            abstract = citation.get("abstract", "No description available.")
            description_title = Align.center(
                Text("Description", style="bold magenta underline")
            )
            description_body = Align.center(
                Padding(Text(abstract, style="white", justify="left"), (0, 4))
            )

            contributors_title = Align.center(
                Text("Contributors", style="bold magenta underline")
            )

            author_table = Table(
                show_header=True, header_style="bold yellow", box=None, pad_edge=False
            )
            author_table.add_column("Name", style="bold", justify="center")
            author_table.add_column("Affiliation", justify="center")

            for author in citation.get("authors", []):
                name = (
                    f"{author.get('given-names', '')} {author.get('family-names', '')}"
                ).strip()
                affiliation = author.get("affiliation", "")
                author_table.add_row(name, affiliation)

            contributors_table = Align.center(Padding(author_table, (0, 4)))

            splash_content = Group(
                ascii_render,
                Rule(style="cyan"),
                version_text,
                url_text,
                Text(),
                description_title,
                description_body,
                Text(),
                contributors_title,
                contributors_table,
            )
        else:
            ascii_title = text2art("CodeEntropy")
            ascii_render = Align.center(Text(ascii_title, style="bold white"))
            splash_content = Group(ascii_render)

        splash_panel = Panel(
            splash_content,
            title="[bold bright_cyan]Welcome to CodeEntropy",
            title_align="center",
            border_style="bright_cyan",
            padding=(1, 4),
            expand=True,
        )

        console.print(splash_panel)

    def print_args_table(self, args: Any) -> None:
        """Print a Rich table of the run configuration arguments.

        Args:
            args: argparse Namespace or object with attributes for configuration.
        """
        table = Table(title="Run Configuration", expand=True)
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for arg in vars(args):
            table.add_row(arg, str(getattr(args, arg)))

        console.print(table)

    def run_entropy_workflow(self) -> None:
        """Run the end-to-end entropy workflow.

        This method:
          - Sets up logging and prints the splash screen
          - Loads YAML config from CWD and parses CLI args
          - Merges args with YAML per-run config
          - Builds the MDAnalysis Universe (with optional force merging)
          - Validates user parameters
          - Constructs dependencies and executes EntropyManager
          - Saves recorded console output to a log file

        Raises:
            Exception: Re-raises any exception after logging with traceback.
        """
        try:
            run_logger = self._logging_config.setup_logging()
            self.show_splash()

            current_directory = os.getcwd()
            config = self._config_manager.load_config(current_directory)

            parser = self._config_manager.setup_argparse()
            args, _ = parser.parse_known_args()
            args.output_file = os.path.join(self.folder, args.output_file)

            for run_name, run_config in config.items():
                if not isinstance(run_config, dict):
                    run_logger.warning(
                        "Run configuration for %s is not a dictionary.", run_name
                    )
                    continue

                args = self._config_manager.merge_configs(args, run_config)

                log_level = (
                    logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
                )
                self._logging_config.update_logging_level(log_level)

                command = " ".join(os.sys.argv)
                logging.getLogger("commands").info(command)

                self._validate_required_args(args)

                self.print_args_table(args)

                universe_operations = UniverseOperations()
                u = self._build_universe(args, universe_operations)

                self._config_manager.input_parameters_validation(u, args)

                group_molecules = GroupMolecules()
                dihedral_analysis = DihedralAnalysis(
                    universe_operations=universe_operations
                )

                entropy_manager = EntropyManager(
                    run_manager=self,
                    args=args,
                    universe=u,
                    data_logger=self._data_logger,
                    group_molecules=group_molecules,
                    dihedral_analysis=dihedral_analysis,
                    universe_operations=universe_operations,
                )
                entropy_manager.execute()

            self._logging_config.save_console_log()

        except Exception as e:
            logger.error("RunManager encountered an error: %s", e, exc_info=True)
            raise

    @staticmethod
    def _validate_required_args(args: Any) -> None:
        """Validate presence of required arguments.

        Args:
            args: argparse Namespace or similar.

        Raises:
            ValueError: If required arguments are missing.
        """
        if not getattr(args, "top_traj_file", None):
            raise ValueError("Missing 'top_traj_file' argument.")
        if not getattr(args, "selection_string", None):
            raise ValueError("Missing 'selection_string' argument.")

    @staticmethod
    def _build_universe(
        args: Any, universe_operations: UniverseOperations
    ) -> mda.Universe:
        """Create an MDAnalysis Universe from args.

        Args:
            args: Parsed arguments containing topology/trajectory and force settings.
            universe_operations: UniverseOperations utility instance.

        Returns:
            An MDAnalysis Universe ready for analysis.
        """
        tprfile = args.top_traj_file[0]
        trrfile = args.top_traj_file[1:]
        forcefile = args.force_file
        fileformat = args.file_format
        kcal_units = args.kcal_force_units

        if forcefile is None:
            logger.debug("Loading Universe with %s and %s", tprfile, trrfile)
            return mda.Universe(tprfile, trrfile, format=fileformat)

        return universe_operations.merge_forces(
            tprfile, trrfile, forcefile, fileformat, kcal_units
        )

    def write_universe(self, u: mda.Universe, name: str = "default") -> str:
        """Write a universe to disk as a pickle.

        Parameters
        ----------
        u : MDAnalyse.Universe
            A Universe object will all topology, dihedrals,coordinates and force
            information
        name : str, Optional. default: 'default'
            The name of file with sub file name .pkl

        Returns
        -------
            name : str
                filename of saved universe
        """
        filename = f"{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(u, f)
        return name

    def read_universe(self, path: str) -> mda.Universe:
        """Read a universe from disk (pickle).

        Parameters
        ----------
        path : str
            The path to file.

        Returns
        -------
            u : MDAnalysis.Universe
                A Universe object will all topology, dihedrals,coordinates and force
                information.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def change_lambda_units(self, arg_lambdas: Any) -> Any:
        """Unit of lambdas : kJ2 mol-2 A-2 amu-1
        change units of lambda to J/s2"""
        # return arg_lambdas * N_AVOGADRO * N_AVOGADRO * AMU2KG * 1e-26
        return arg_lambdas * 1e29 / self.N_AVOGADRO

    def get_KT2J(self, arg_temper: float) -> float:
        """A temperature dependent KT to Joule conversion"""
        return 4.11e-21 * arg_temper / self.DEF_TEMPER
