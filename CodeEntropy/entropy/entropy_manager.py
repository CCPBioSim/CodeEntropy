import logging
import math
from collections import defaultdict

import pandas as pd

from CodeEntropy.config.logging_config import LoggingConfig
from CodeEntropy.entropy.entropy_graph import EntropyGraph
from CodeEntropy.levels.hierarchy_graph import LevelDAG
from CodeEntropy.levels.level_hierarchy import LevelHierarchy

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class EntropyManager:
    """
    Manages entropy calculations at multiple molecular levels, based on a
    molecular dynamics trajectory.
    """

    def __init__(
        self,
        run_manager,
        args,
        universe,
        data_logger,
        group_molecules,
        dihedral_analysis,
        universe_operations,
    ):
        """
        Initializes the EntropyManager with required components.

        Args:
            run_manager: Manager for universe and selection operations.
            args: Argument namespace containing user parameters.
            universe: MDAnalysis universe representing the simulation system.
            data_logger: Logger for storing and exporting entropy data.
            group_molecules: includes the grouping functions for averaging over
            molecules.
        """
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._data_logger = data_logger
        self._group_molecules = group_molecules
        self._dihedral_analysis = dihedral_analysis
        self._universe_operations = universe_operations
        self._GAS_CONST = 8.3144598484848

    def execute(self):
        start, end, step = self._get_trajectory_bounds()
        n_frames = self._get_number_frames(start, end, step)

        console.print(f"Analyzing a total of {n_frames} frames in this calculation.")

        reduced_universe = self._get_reduced_universe()

        level_hierarchy = LevelHierarchy()
        number_molecules, levels = level_hierarchy.select_levels(reduced_universe)

        groups = self._group_molecules.grouping_molecules(
            reduced_universe, self._args.grouping
        )
        logger.info(f"Number of molecule groups: {len(groups)}")

        water_atoms = self._universe.select_atoms("water")
        water_resids = {res.resid for res in water_atoms.residues}

        water_groups = {
            gid: g
            for gid, g in groups.items()
            if any(
                res.resid in water_resids
                for mol in [self._universe.atoms.fragments[i] for i in g]
                for res in mol.residues
            )
        }
        nonwater_groups = {
            gid: g for gid, g in groups.items() if gid not in water_groups
        }

        if self._args.water_entropy and water_groups:
            self._handle_water_entropy(start, end, step, water_groups)
        else:
            nonwater_groups.update(water_groups)

        shared_data = {
            "entropy_manager": self,
            "run_manager": self._run_manager,
            "data_logger": self._data_logger,
            "args": self._args,
            "universe": self._universe,
            "reduced_universe": reduced_universe,
            "levels": levels,
            "groups": nonwater_groups,
            "start": start,
            "end": end,
            "step": step,
            "n_frames": n_frames,
        }

        logger.info(f"shared_data: {shared_data}")

        LevelDAG(self._universe_operations).build().execute(shared_data)

        entropy_results = EntropyGraph().build().execute(shared_data)
        shared_data.update(entropy_results)

        logger.info(f"entropy_results: {entropy_results}")

        self._finalize_molecule_results()
        self._data_logger.log_tables()

    def _handle_water_entropy(self, start, end, step, water_groups):
        """
        Compute water entropy for each water group, log data, and update selection
        string to exclude water from further analysis.

        Args:
            start (int): Start frame index
            end (int): End frame index
            step (int): Step size
            water_groups (dict): {group_id: [atom indices]} for water
        """
        if not water_groups or not self._args.water_entropy:
            return

        for group_id, atom_indices in water_groups.items():

            self._calculate_water_entropy(
                universe=self._universe,
                start=start,
                end=end,
                step=step,
                group_id=group_id,
            )

        self._args.selection_string = (
            self._args.selection_string + " and not water"
            if self._args.selection_string != "all"
            else "not water"
        )

        logger.debug(f"WaterEntropy: molecule_data: {self._data_logger.molecule_data}")
        logger.debug(f"WaterEntropy: residue_data: {self._data_logger.residue_data}")

    def _initialize_molecules(self):
        """
        Prepare the reduced universe and determine molecule-level configurations.

        Returns:
            tuple: A tuple containing:
                - reduced_atom (Universe): The reduced atom selection.
                - number_molecules (int): Number of molecules in the system.
                - levels (list): List of entropy levels per molecule.
                - groups (dict): Groups for averaging over molecules.
        """
        # Based on the selection string, create a new MDAnalysis universe
        reduced_atom = self._get_reduced_universe()

        level_hierarchy = LevelHierarchy()

        # Count the molecules and identify the length scale levels for each one
        number_molecules, levels = level_hierarchy.select_levels(reduced_atom)

        # Group the molecules for averaging
        grouping = self._args.grouping
        groups = self._group_molecules.grouping_molecules(reduced_atom, grouping)

        return reduced_atom, number_molecules, levels, groups

    def _get_trajectory_bounds(self):
        """
        Returns the start, end, and step frame indices based on input arguments.

        Returns:
            Tuple of (start, end, step) frame indices.
        """
        start = self._args.start or 0
        end = len(self._universe.trajectory) if self._args.end == -1 else self._args.end
        step = self._args.step or 1

        return start, end, step

    def _get_number_frames(self, start, end, step):
        """
        Calculates the total number of trajectory frames used in the calculation.

        Args:
            start (int): Start frame index.
            end (int): End frame index. If -1, it refers to the end of the trajectory.
            step (int): Frame step size.

        Returns:
            int: Total number of frames considered.
        """
        return math.floor((end - start) / step)

    def _get_reduced_universe(self):
        """
        Applies atom selection based on the user's input.

        Returns:
            MDAnalysis.Universe: Selected subset of the system.
        """
        # If selection string is "all" the universe does not change
        if self._args.selection_string == "all":
            return self._universe

        # Otherwise create a new (smaller) universe based on the selection
        u = self._universe
        selection_string = self._args.selection_string
        reduced = self._universe_operations.new_U_select_atom(u, selection_string)
        name = f"{len(reduced.trajectory)}_frame_dump_atom_selection"
        self._run_manager.write_universe(reduced, name)

        return reduced

    def _finalize_molecule_results(self):
        """
        Aggregates and logs total entropy and frame counts per molecule.
        """
        entropy_by_molecule = defaultdict(float)
        for (
            mol_id,
            level,
            entropy_type,
            result,
        ) in self._data_logger.molecule_data:
            if level != "Group Total":
                try:
                    entropy_by_molecule[mol_id] += float(result)
                except ValueError:
                    logger.warning(f"Skipping invalid entry: {mol_id}, {result}")

        for mol_id in entropy_by_molecule.keys():
            total_entropy = entropy_by_molecule[mol_id]

            self._data_logger.molecule_data.append(
                (
                    mol_id,
                    "Group Total",
                    "Group Total Entropy",
                    total_entropy,
                )
            )

        self._data_logger.save_dataframes_as_json(
            pd.DataFrame(
                self._data_logger.molecule_data,
                columns=[
                    "Group ID",
                    "Level",
                    "Type",
                    "Result (J/mol/K)",
                ],
            ),
            pd.DataFrame(
                self._data_logger.residue_data,
                columns=[
                    "Group ID",
                    "Residue Name",
                    "Level",
                    "Type",
                    "Frame Count",
                    "Result (J/mol/K)",
                ],
            ),
            self._args.output_file,
        )
