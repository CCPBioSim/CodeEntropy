import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from CodeEntropy.config.logging_config import LoggingConfig
from CodeEntropy.entropy.entropy_graph import EntropyGraph
from CodeEntropy.levels.hierarchy_graph import LevelDAG

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
        level_manager,
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
            level_manager: Provides level-specific data such as matrices and dihedrals.
            group_molecules: includes the grouping functions for averaging over
            molecules.
        """
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._data_logger = data_logger
        self._level_manager = level_manager
        self._group_molecules = group_molecules
        self._dihedral_analysis = dihedral_analysis
        self._universe_operations = universe_operations
        self._GAS_CONST = 8.3144598484848

    def execute(self):
        """
        Run the full entropy computation workflow using the DAG system.

        Workflow:
            1. Parse trajectory frame bounds
            2. Build reduced universe
            3. Detect molecules + levels
            4. Run LEVEL DAG
            5. Run ENTROPY DAG
            6. Log and store results
        """
        start, end, step = self._get_trajectory_bounds()
        number_frames = self._get_number_frames(start, end, step)

        console.print(
            f"Analyzing a total of {number_frames} frames in this calculation."
        )

        reduced_atom, number_molecules, levels, groups = self._initialize_molecules()
        logger.debug(f"[EntropyManager] Reduced universe loaded: {reduced_atom}")

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
            "universe": self._universe,
            "reduced_universe": reduced_atom,
            "levels": levels,
            "groups": nonwater_groups,
            "args": self._args,
            "start": start,
            "end": end,
            "step": step,
            "n_frames": number_frames,
        }

        level_results = LevelDAG().build().execute(shared_data)
        entropy_results = EntropyGraph().build().execute(level_results)

        self._finalize_outputs(entropy_results)
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

        # Count the molecules and identify the length scale levels for each one
        number_molecules, levels = self._level_manager.select_levels(reduced_atom)

        # Group the molecules for averaging
        grouping = self._args.grouping
        groups = self._group_molecules.grouping_molecules(reduced_atom, grouping)

        return reduced_atom, number_molecules, levels, groups

    def _compute_entropies(
        self,
        reduced_atom,
        levels,
        groups,
        force_matrices,
        torque_matrices,
        states_ua,
        states_res,
        frame_counts,
        number_frames,
        ve,
        ce,
    ):
        """
        Compute vibrational and conformational entropies for all molecules and levels.

        This method iterates over each molecule and its associated entropy levels
        (united_atom, residue, polymer), computing the corresponding entropy
        contributions using force/torque matrices and dihedral conformations.

        For each level:
        - "united_atom": Computes per-residue conformational states and entropy.
        - "residue": Computes molecule-level conformational and vibrational entropy.
        - "polymer": Computes only vibrational entropy.

        Parameters:
            reduced_atom (Universe): The reduced atom selection from the trajectory.
            levels (list): List of entropy levels per molecule.
            groups (dict): Groups for averaging over molecules.
            force_matrices (dict): Precomputed force covariance matrices.
            torque_matrices (dict): Precomputed torque covariance matrices.
            states_ua (dict): Dictionary to store united-atom conformational states.
            states_res (list): List to store residue-level conformational states.
            frames_count (dict): Dictionary to store the frame counts
            number_frames (int): Total number of trajectory frames to process.
            ve: Vibrational Entropy object
            ce: Conformational Entropy object
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[title]}", justify="right"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
        ) as progress:

            task = progress.add_task(
                "[green]Calculating Entropy...",
                total=len(groups),
                title="Starting...",
            )

            for group_id in groups.keys():
                mol = self._universe_operations.get_molecule_container(
                    reduced_atom, groups[group_id][0]
                )

                residue_group = "_".join(
                    sorted(set(res.resname for res in mol.residues))
                )
                group_residue_count = len(groups[group_id])
                group_atom_count = 0
                for mol_id in groups[group_id]:
                    each_mol = self._universe_operations.get_molecule_container(
                        reduced_atom, mol_id
                    )
                    group_atom_count += len(each_mol.atoms)
                self._data_logger.add_group_label(
                    group_id, residue_group, group_residue_count, group_atom_count
                )

                resname = mol.atoms[0].resname
                resid = mol.atoms[0].resid
                segid = mol.atoms[0].segid

                mol_label = f"{resname}_{resid} (segid {segid})"

                for level in levels[groups[group_id][0]]:
                    progress.update(
                        task,
                        title=f"Calculating entropy values | "
                        f"Molecule: {mol_label} | "
                        f"Level: {level}",
                    )
                    highest = level == levels[groups[group_id][0]][-1]

                    if level == "united_atom":
                        self._process_united_atom_entropy(
                            group_id,
                            mol,
                            ve,
                            ce,
                            level,
                            force_matrices["ua"],
                            torque_matrices["ua"],
                            states_ua,
                            frame_counts["ua"],
                            highest,
                            number_frames,
                        )

                    elif level == "residue":
                        self._process_vibrational_entropy(
                            group_id,
                            mol,
                            number_frames,
                            ve,
                            level,
                            force_matrices["res"][group_id],
                            torque_matrices["res"][group_id],
                            highest,
                        )

                        self._process_conformational_entropy(
                            group_id,
                            mol,
                            ce,
                            level,
                            states_res,
                            number_frames,
                        )

                    elif level == "polymer":
                        self._process_vibrational_entropy(
                            group_id,
                            mol,
                            number_frames,
                            ve,
                            level,
                            force_matrices["poly"][group_id],
                            torque_matrices["poly"][group_id],
                            highest,
                        )

                progress.advance(task)

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

    def _process_united_atom_entropy(
        self,
        group_id,
        mol_container,
        ve,
        ce,
        level,
        force_matrix,
        torque_matrix,
        states,
        frame_counts,
        highest,
        number_frames,
    ):
        """
        Calculates translational, rotational, and conformational entropy at the
        united-atom level.

        Args:
            group_id (int): ID of the group.
            mol_container (Universe): Universe for the selected molecule.
            ve: VibrationalEntropy object.
            ce: ConformationalEntropy object.
            level (str): Granularity level (should be 'united_atom').
            start, end, step (int): Trajectory frame parameters.
            n_frames (int): Number of trajectory frames.
            frame_counts: Number of frames counted
            highest (bool): Whether this is the highest level of resolution for
             the molecule.
            number_frames (int): The number of frames analysed.
        """
        S_trans, S_rot, S_conf = 0, 0, 0

        # The united atom entropy is calculated separately for each residue
        # This is to allow residue by residue information
        # and prevents the matrices from becoming too large
        for residue_id, residue in enumerate(mol_container.residues):

            key = (group_id, residue_id)

            # Find the relevant force and torque matrices and tidy them up
            # by removing rows and columns that are all zeros
            f_matrix = force_matrix[key]
            f_matrix = self._level_manager.filter_zero_rows_columns(f_matrix)

            t_matrix = torque_matrix[key]
            t_matrix = self._level_manager.filter_zero_rows_columns(t_matrix)

            # Calculate the vibrational entropy
            S_trans_res = ve.vibrational_entropy_calculation(
                f_matrix, "force", self._args.temperature, highest
            )
            S_rot_res = ve.vibrational_entropy_calculation(
                t_matrix, "torque", self._args.temperature, highest
            )

            # Get the relevant conformational states
            values = states[key]
            # Check if there is information in the states array
            contains_non_empty_states = (
                np.any(values) if isinstance(values, np.ndarray) else any(values)
            )

            # Calculate the conformational entropy
            # If there are no conformational states (i.e. no dihedrals)
            # then the conformational entropy is zero
            S_conf_res = (
                ce.conformational_entropy_calculation(values)
                if contains_non_empty_states
                else 0
            )

            # Add the data to the united atom level entropy
            S_trans += S_trans_res
            S_rot += S_rot_res
            S_conf += S_conf_res

            # Print out the data for each residue
            self._data_logger.add_residue_data(
                group_id,
                residue.resname,
                level,
                "Transvibrational",
                frame_counts[key],
                S_trans_res,
            )
            self._data_logger.add_residue_data(
                group_id,
                residue.resname,
                level,
                "Rovibrational",
                frame_counts[key],
                S_rot_res,
            )
            self._data_logger.add_residue_data(
                group_id,
                residue.resname,
                level,
                "Conformational",
                frame_counts[key],
                S_conf_res,
            )

        # Print the total united atom level data for the molecule group
        self._data_logger.add_results_data(group_id, level, "Transvibrational", S_trans)
        self._data_logger.add_results_data(group_id, level, "Rovibrational", S_rot)
        self._data_logger.add_results_data(group_id, level, "Conformational", S_conf)

        residue_group = "_".join(
            sorted(set(res.resname for res in mol_container.residues))
        )

        logger.debug(f"residue_group {residue_group}")

    def _process_vibrational_entropy(
        self,
        group_id,
        mol_container,
        number_frames,
        ve,
        level,
        force_matrix,
        torque_matrix,
        highest,
    ):
        """
        Calculates vibrational entropy.

        Args:
            group_id (int): Group ID.
            ve: VibrationalEntropy object.
            level (str): Current granularity level.
            force_matrix : Force covariance matrix
            torque_matrix : Torque covariance matrix
            frame_count:
            highest (bool): Flag indicating if this is the highest granularity
            level.
        """
        # Find the relevant force and torque matrices and tidy them up
        # by removing rows and columns that are all zeros

        force_matrix = self._level_manager.filter_zero_rows_columns(force_matrix)

        torque_matrix = self._level_manager.filter_zero_rows_columns(torque_matrix)

        # Calculate the vibrational entropy
        S_trans = ve.vibrational_entropy_calculation(
            force_matrix, "force", self._args.temperature, highest
        )
        S_rot = ve.vibrational_entropy_calculation(
            torque_matrix, "torque", self._args.temperature, highest
        )

        # Print the vibrational entropy for the molecule group
        self._data_logger.add_results_data(group_id, level, "Transvibrational", S_trans)
        self._data_logger.add_results_data(group_id, level, "Rovibrational", S_rot)

        residue_group = "_".join(
            sorted(set(res.resname for res in mol_container.residues))
        )
        residue_count = len(mol_container.residues)
        atom_count = len(mol_container.atoms)
        self._data_logger.add_group_label(
            group_id, residue_group, residue_count, atom_count
        )

    def _process_conformational_entropy(
        self, group_id, mol_container, ce, level, states, number_frames
    ):
        """
        Computes conformational entropy at the residue level (whole-molecule dihedral
        analysis).

        Args:
            mol_id (int): ID of the molecule.
            mol_container (Universe): Selected molecule's universe.
            ce: ConformationalEntropy object.
            level (str): Level name (should be 'residue').
            states (array): The conformational states.
            number_frames (int): Number of frames used.
        """
        # Get the relevant conformational states
        # Check if there is information in the states array
        group_states = states[group_id] if group_id < len(states) else None

        if group_states is not None:
            contains_state_data = (
                group_states.any()
                if isinstance(group_states, np.ndarray)
                else any(group_states)
            )
        else:
            contains_state_data = False

        # Calculate the conformational entropy
        # If there are no conformational states (i.e. no dihedrals)
        # then the conformational entropy is zero
        S_conf = (
            ce.conformational_entropy_calculation(group_states)
            if contains_state_data
            else 0
        )
        self._data_logger.add_results_data(group_id, level, "Conformational", S_conf)

        residue_group = "_".join(
            sorted(set(res.resname for res in mol_container.residues))
        )
        residue_count = len(mol_container.residues)
        atom_count = len(mol_container.atoms)
        self._data_logger.add_group_label(
            group_id, residue_group, residue_count, atom_count
        )

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
