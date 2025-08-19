import logging
import os
import pickle

import MDAnalysis as mda
import requests
import yaml
from art import text2art
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from rich.align import Align
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from CodeEntropy.config.arg_config_manager import ConfigManager
from CodeEntropy.config.data_logger import DataLogger
from CodeEntropy.config.logging_config import LoggingConfig
from CodeEntropy.entropy import EntropyManager
from CodeEntropy.group_molecules import GroupMolecules
from CodeEntropy.levels import LevelManager

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class RunManager:
    """
    Handles the setup and execution of entropy analysis runs, including configuration
    loading, logging, and access to physical constants used in calculations.
    """

    def __init__(self, folder):
        """
        Initializes the RunManager with the working folder and sets up configuration,
        data logging, and logging systems. Also defines physical constants used in
        entropy calculations.
        """
        self.folder = folder
        self._config_manager = ConfigManager()
        self._data_logger = DataLogger()
        self._logging_config = LoggingConfig(folder)
        self._N_AVOGADRO = 6.0221415e23
        self._DEF_TEMPER = 298

    @property
    def N_AVOGADRO(self):
        """Returns Avogadro's number used in entropy calculations."""
        return self._N_AVOGADRO

    @property
    def DEF_TEMPER(self):
        """Returns the default temperature (in Kelvin) used in the analysis."""
        return self._DEF_TEMPER

    @staticmethod
    def create_job_folder():
        """
        Create a new job folder with an incremented job number based on existing
        folders.
        """
        # Get the current working directory
        current_dir = os.getcwd()

        # Get a list of existing folders that start with "job"
        existing_folders = [f for f in os.listdir(current_dir) if f.startswith("job")]

        # Extract numbers from existing folder names
        job_numbers = []
        for folder in existing_folders:
            try:
                # Assuming folder names are in the format "jobXXX"
                job_number = int(folder[3:])  # Get the number part after "job"
                job_numbers.append(job_number)
            except ValueError:
                continue  # Ignore any folder names that don't follow the pattern

        # If no folders exist, start with job001
        if not job_numbers:
            next_job_number = 1
        else:
            next_job_number = max(job_numbers) + 1

        # Create the new job folder name
        new_job_folder = f"job{next_job_number:03d}"

        # Create the full path to the new folder
        new_folder_path = os.path.join(current_dir, new_job_folder)

        # Create the directory
        os.makedirs(new_folder_path, exist_ok=True)

        # Return the path of the newly created folder
        return new_folder_path

    def load_citation_data(self):
        """
        Load CITATION.cff from GitHub into memory.
        Return empty dict if offline.
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

    def show_splash(self):
        """Render splash screen with optional citation metadata."""
        citation = self.load_citation_data()

        if citation:
            # ASCII Title
            ascii_title = text2art(citation.get("title", "CodeEntropy"))
            ascii_render = Align.center(Text(ascii_title, style="bold white"))

            # Metadata
            version = citation.get("version", "?")
            release_date = citation.get("date-released", "?")
            url = citation.get("url", citation.get("repository-code", ""))

            version_text = Align.center(
                Text(f"Version {version} | Released {release_date}", style="green")
            )
            url_text = Align.center(Text(url, style="blue underline"))

            # Description block
            abstract = citation.get("abstract", "No description available.")
            description_title = Align.center(
                Text("Description", style="bold magenta underline")
            )
            description_body = Align.center(
                Padding(Text(abstract, style="white", justify="left"), (0, 4))
            )

            # Contributors table
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

            # Full layout
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
            # ASCII Title
            ascii_title = text2art("CodeEntropy")
            ascii_render = Align.center(Text(ascii_title, style="bold white"))

            splash_content = Group(
                ascii_render,
            )

        splash_panel = Panel(
            splash_content,
            title="[bold bright_cyan]Welcome to CodeEntropy",
            title_align="center",
            border_style="bright_cyan",
            padding=(1, 4),
            expand=True,
        )

        console.print(splash_panel)

    def print_args_table(self, args):
        table = Table(title="Run Configuration", expand=True)

        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for arg in vars(args):
            table.add_row(arg, str(getattr(args, arg)))

        console.print(table)

    def run_entropy_workflow(self):
        """
        Runs the entropy analysis workflow by setting up logging, loading configuration
        files, parsing arguments, and executing the analysis for each configured run.
        Initializes the MDAnalysis Universe and supporting managers, and logs all
        relevant inputs and commands.
        """
        try:
            logger = self._logging_config.setup_logging()
            self.show_splash()

            current_directory = os.getcwd()

            config = self._config_manager.load_config(current_directory)
            parser = self._config_manager.setup_argparse()
            args, _ = parser.parse_known_args()
            args.output_file = os.path.join(self.folder, args.output_file)

            for run_name, run_config in config.items():
                if not isinstance(run_config, dict):
                    logger.warning(
                        f"Run configuration for {run_name} is not a dictionary."
                    )
                    continue

                args = self._config_manager.merge_configs(args, run_config)

                log_level = logging.DEBUG if args.verbose else logging.INFO
                self._logging_config.update_logging_level(log_level)

                command = " ".join(os.sys.argv)
                logging.getLogger("commands").info(command)

                if not getattr(args, "top_traj_file", None):
                    raise ValueError("Missing 'top_traj_file' argument.")
                if not getattr(args, "selection_string", None):
                    raise ValueError("Missing 'selection_string' argument.")

                self.print_args_table(args)

                # Load MDAnalysis Universe
                tprfile = args.top_traj_file[0]
                trrfile = args.top_traj_file[1:]
                logger.debug(f"Loading Universe with {tprfile} and {trrfile}")
                u = mda.Universe(tprfile, trrfile)

                self._config_manager.input_parameters_validation(u, args)

                # Create LevelManager instance
                level_manager = LevelManager()

                # Create GroupMolecules instance
                group_molecules = GroupMolecules()

                # Inject all dependencies into EntropyManager
                entropy_manager = EntropyManager(
                    run_manager=self,
                    args=args,
                    universe=u,
                    data_logger=self._data_logger,
                    level_manager=level_manager,
                    group_molecules=group_molecules,
                )

                entropy_manager.execute()

            self._logging_config.save_console_log()

        except Exception as e:
            logger.error(f"RunManager encountered an error: {e}", exc_info=True)
            raise

    def new_U_select_frame(self, u, start=None, end=None, step=1):
        """Create a reduced universe by dropping frames according to user selection

        Parameters
        ----------
        u : MDAnalyse.Universe
            A Universe object will all topology, dihedrals,coordinates and force
            information
        start : int or None, Optional, default: None
            Frame id to start analysis. Default None will start from frame 0
        end : int or None, Optional, default: None
            Frame id to end analysis. Default None will end at last frame
        step : int, Optional, default: 1
            Steps between frame.

        Returns
        -------
            u2 : MDAnalysis.Universe
                reduced universe
        """
        if start is None:
            start = 0
        if end is None:
            end = len(u.trajectory)
        select_atom = u.select_atoms("all", updating=True)
        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )
        dimensions = (
            AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )
        u2 = mda.Merge(select_atom)
        u2.load_new(
            coordinates, format=MemoryReader, forces=forces, dimensions=dimensions
        )
        logger.debug(f"MDAnalysis.Universe - reduced universe: {u2}")
        return u2

    def new_U_select_atom(self, u, select_string="all"):
        """Create a reduced universe by dropping atoms according to user selection

        Parameters
        ----------
        u : MDAnalyse.Universe
            A Universe object will all topology, dihedrals,coordinates and force
            information
        select_string : str, Optional, default: 'all'
            MDAnalysis.select_atoms selection string.

        Returns
        -------
            u2 : MDAnalysis.Universe
                reduced universe

        """
        select_atom = u.select_atoms(select_string, updating=True)
        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        dimensions = (
            AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        u2 = mda.Merge(select_atom)
        u2.load_new(
            coordinates, format=MemoryReader, forces=forces, dimensions=dimensions
        )
        logger.debug(f"MDAnalysis.Universe - reduced universe: {u2}")
        return u2

    def write_universe(self, u, name="default"):
        """Write a universe to working directories as pickle

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
        pickle.dump(u, open(filename, "wb"))
        return name

    def read_universe(self, path):
        """read a universe to working directories as pickle

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
        u = pickle.load(open(path, "rb"))
        return u

    def change_lambda_units(self, arg_lambdas):
        """Unit of lambdas : kJ2 mol-2 A-2 amu-1
        change units of lambda to J/s2"""
        # return arg_lambdas * N_AVOGADRO * N_AVOGADRO * AMU2KG * 1e-26
        return arg_lambdas * 1e29 / self.N_AVOGADRO

    def get_KT2J(self, arg_temper):
        """A temperature dependent KT to Joule conversion"""
        return 4.11e-21 * arg_temper / self.DEF_TEMPER
