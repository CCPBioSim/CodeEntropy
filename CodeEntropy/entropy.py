import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import waterEntropy.recipes.interfacial_solvent as GetSolvent
from numpy import linalg as la
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from CodeEntropy.config.logging_config import LoggingConfig

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class EntropyManager:
    """
    Manages entropy calculations at multiple molecular levels, based on a
    molecular dynamics trajectory.
    """

    def __init__(
        self, run_manager, args, universe, data_logger, level_manager, group_molecules
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
        self._GAS_CONST = 8.3144598484848

    def execute(self):
        """
        Run the full entropy computation workflow.

        This method orchestrates the entire entropy analysis pipeline, including:
        - Handling water entropy if present.
        - Initializing molecular structures and levels.
        - Building force and torque covariance matrices.
        - Computing vibrational and conformational entropies.
        - Finalizing and logging results.
        """
        # Set up initial information
        start, end, step = self._get_trajectory_bounds()
        number_frames = self._get_number_frames(start, end, step)

        console.print(
            f"Analyzing a total of {number_frames} frames in this calculation."
        )

        ve = VibrationalEntropy(
            self._run_manager,
            self._args,
            self._universe,
            self._data_logger,
            self._level_manager,
            self._group_molecules,
        )
        ce = ConformationalEntropy(
            self._run_manager,
            self._args,
            self._universe,
            self._data_logger,
            self._level_manager,
            self._group_molecules,
        )

        reduced_atom, number_molecules, levels, groups = self._initialize_molecules()

        water_atoms = self._universe.select_atoms("water")
        water_resids = set(res.resid for res in water_atoms.residues)

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

        force_matrices, torque_matrices, frame_counts = (
            self._level_manager.build_covariance_matrices(
                self,
                reduced_atom,
                levels,
                nonwater_groups,
                start,
                end,
                step,
                number_frames,
            )
        )

        # Identify the conformational states from dihedral angles for the
        # conformational entropy calculations
        states_ua, states_res = self._level_manager.build_conformational_states(
            self,
            reduced_atom,
            levels,
            nonwater_groups,
            start,
            end,
            step,
            number_frames,
            self._args.bin_width,
            ce,
        )

        # Complete the entropy calculations
        self._compute_entropies(
            reduced_atom,
            levels,
            nonwater_groups,
            force_matrices,
            torque_matrices,
            states_ua,
            states_res,
            frame_counts,
            number_frames,
            ve,
            ce,
        )

        # Print the results in a nicely formated way
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
                mol = self._get_molecule_container(reduced_atom, groups[group_id][0])

                residue_group = "_".join(
                    sorted(set(res.resname for res in mol.residues))
                )
                group_residue_count = len(groups[group_id])
                group_atom_count = 0
                for mol_id in groups[group_id]:
                    each_mol = self._get_molecule_container(reduced_atom, mol_id)
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
        reduced = self._run_manager.new_U_select_atom(
            self._universe, self._args.selection_string
        )
        name = f"{len(reduced.trajectory)}_frame_dump_atom_selection"
        self._run_manager.write_universe(reduced, name)
        return reduced

    def _get_molecule_container(self, universe, molecule_id):
        """
        Extracts the atom group corresponding to a single molecule from the universe.

        Args:
            universe (MDAnalysis.Universe): The reduced universe.
            molecule_id (int): Index of the molecule to extract.

        Returns:
            MDAnalysis.Universe: Universe containing only the selected molecule.
        """
        # Identify the atoms in the molecule
        frag = universe.atoms.fragments[molecule_id]
        selection_string = f"index {frag.indices[0]}:{frag.indices[-1]}"

        # Build a new universe with only the one molecule
        return self._run_manager.new_U_select_atom(universe, selection_string)

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
                ce.conformational_entropy_calculation(values, number_frames)
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
            ce.conformational_entropy_calculation(group_states, number_frames)
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

    def _calculate_water_entropy(self, universe, start, end, step, group_id=None):
        """
        Calculate and aggregate the entropy of water molecules in a simulation.

        This function computes orientational, translational, and rotational
        entropy components for all water molecules, aggregates them per residue,
        and maps all waters to a single group ID. It also logs the total results
        and labels the water group in the data logger.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            The simulation universe containing water molecules.
        start : int
            The starting frame for analysis.
        end : int
            The ending frame for analysis.
        step : int
            Frame interval for analysis.
        group_id : int or str, optional
            The group ID to which all water molecules will be assigned.
        """
        Sorient_dict, covariances, vibrations, _, water_count = (
            GetSolvent.get_interfacial_water_orient_entropy(
                universe, start, end, step, self._args.temperature, parallel=True
            )
        )

        self._calculate_water_orientational_entropy(Sorient_dict, group_id)
        self._calculate_water_vibrational_translational_entropy(
            vibrations, group_id, covariances
        )
        self._calculate_water_vibrational_rotational_entropy(
            vibrations, group_id, covariances
        )

        water_selection = universe.select_atoms("resname WAT")
        actual_water_residues = len(water_selection.residues)
        residue_names = {
            resname
            for res_dict in Sorient_dict.values()
            for resname in res_dict.keys()
            if resname.upper() in water_selection.residues.resnames
        }

        residue_group = "_".join(sorted(residue_names)) if residue_names else "WAT"
        self._data_logger.add_group_label(
            group_id, residue_group, actual_water_residues, len(water_selection.atoms)
        )

    def _calculate_water_orientational_entropy(self, Sorient_dict, group_id):
        """
        Aggregate orientational entropy for all water molecules into a single group.

        Parameters
        ----------
        Sorient_dict : dict
            Dictionary containing orientational entropy values per residue.
        group_id : int or str
            The group ID to which the water residues belong.
        covariances : object
            Covariance object.
        """
        for resid, resname_dict in Sorient_dict.items():
            for resname, values in resname_dict.items():
                if isinstance(values, list) and len(values) == 2:
                    Sor, count = values
                    self._data_logger.add_residue_data(
                        group_id, resname, "Water", "Orientational", count, Sor
                    )

    def _calculate_water_vibrational_translational_entropy(
        self, vibrations, group_id, covariances
    ):
        """
        Aggregate translational vibrational entropy for all water molecules.

        Parameters
        ----------
        vibrations : object
            Object containing translational entropy data (vibrations.translational_S).
        group_id : int or str
            The group ID for the water residues.
        covariances : object
            Covariance object.
        """

        for (solute_id, _), entropy in vibrations.translational_S.items():
            if isinstance(entropy, (list, np.ndarray)):
                entropy = float(np.sum(entropy))

            count = covariances.counts.get((solute_id, "WAT"), 1)
            resname = solute_id.rsplit("_", 1)[0] if "_" in solute_id else solute_id
            self._data_logger.add_residue_data(
                group_id, resname, "Water", "Transvibrational", count, entropy
            )

    def _calculate_water_vibrational_rotational_entropy(
        self, vibrations, group_id, covariances
    ):
        """
        Aggregate rotational vibrational entropy for all water molecules.

        Parameters
        ----------
        vibrations : object
            Object containing rotational entropy data (vibrations.rotational_S).
        group_id : int or str
            The group ID for the water residues.
        covariances : object
            Covariance object.
        """
        for (solute_id, _), entropy in vibrations.rotational_S.items():
            if isinstance(entropy, (list, np.ndarray)):
                entropy = float(np.sum(entropy))

            count = covariances.counts.get((solute_id, "WAT"), 1)

            resname = solute_id.rsplit("_", 1)[0] if "_" in solute_id else solute_id
            self._data_logger.add_residue_data(
                group_id, resname, "Water", "Rovibrational", count, entropy
            )


class VibrationalEntropy(EntropyManager):
    """
    Performs vibrational entropy calculations using molecular trajectory data.
    Extends the base EntropyManager with constants and logic specific to
    vibrational modes and thermodynamic properties.
    """

    def __init__(
        self, run_manager, args, universe, data_logger, level_manager, group_molecules
    ):
        """
        Initializes the VibrationalEntropy manager with all required components and
        defines physical constants used in vibrational entropy calculations.
        """
        super().__init__(
            run_manager, args, universe, data_logger, level_manager, group_molecules
        )
        self._PLANCK_CONST = 6.62607004081818e-34

    def frequency_calculation(self, lambdas, temp):
        """
        Function to calculate an array of vibrational frequencies from the eigenvalues
        of the covariance matrix.

        Calculated from eq. (3) in Higham, S.-Y. Chou, F. Gräter and  R. H. Henchman,
        Molecular Physics, 2018, 116, 1965–1976//eq. (3) in A. Chakravorty, J. Higham
        and R. H. Henchman, J. Chem. Inf. Model., 2020, 60, 5540–5551

        frequency=sqrt(λ/kT)/2π

        Args:
            lambdas : array of floats - eigenvalues of the covariance matrix
            temp: float - temperature

        Returns:
            frequencies : array of floats - corresponding vibrational frequencies
        """
        pi = np.pi
        # get kT in Joules from given temperature
        kT = self._run_manager.get_KT2J(temp)
        logger.debug(f"Temperature: {temp}, kT: {kT}")

        lambdas = np.array(lambdas)  # Ensure input is a NumPy array
        logger.debug(f"Eigenvalues (lambdas): {lambdas}")

        # Filter out lambda values that are negative or imaginary numbers
        # As these will produce supurious entropy results that can crash
        # the calculation
        lambdas = np.real_if_close(lambdas, tol=1000)
        valid_mask = (
            np.isreal(lambdas) & (lambdas > 0) & (~np.isclose(lambdas, 0, atol=1e-07))
        )

        # If any lambdas were removed by the filter, warn the user
        # as this will suggest insufficient sampling in the simulation data
        if len(lambdas) > np.count_nonzero(valid_mask):
            logger.warning(
                f"{len(lambdas) - np.count_nonzero(valid_mask)} "
                f"invalid eigenvalues excluded (complex, non-positive, or near-zero)."
            )

        lambdas = lambdas[valid_mask].real

        # Compute frequencies safely
        frequencies = 1 / (2 * pi) * np.sqrt(lambdas / kT)
        logger.debug(f"Calculated frequencies: {frequencies}")

        return frequencies

    def vibrational_entropy_calculation(self, matrix, matrix_type, temp, highest_level):
        """
        Function to calculate the vibrational entropy for each level calculated from
        eq. (4) in J. Higham, S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular
        Physics, 2018, 116, 1965–1976 / eq. (2) in A. Chakravorty, J. Higham and
        R. H. Henchman, J. Chem. Inf. Model., 2020, 60, 5540–5551.

        Args:
            matrix : matrix - force/torque covariance matrix
            matrix_type: string
            temp: float - temperature
            highest_level: bool - is this the highest level of the heirarchy

        Returns:
            S_vib_total : float - transvibrational/rovibrational entropy
        """
        # N beads at a level => 3N x 3N covariance matrix => 3N eigenvalues
        # Get eigenvalues of the given matrix and change units to SI units
        lambdas = la.eigvals(matrix)
        logger.debug(f"Eigenvalues (lambdas) before unit change: {lambdas}")

        lambdas = self._run_manager.change_lambda_units(lambdas)
        logger.debug(f"Eigenvalues (lambdas) after unit change: {lambdas}")

        # Calculate frequencies from the eigenvalues
        frequencies = self.frequency_calculation(lambdas, temp)
        logger.debug(f"Calculated frequencies: {frequencies}")

        # Sort frequencies lowest to highest
        frequencies = np.sort(frequencies)
        logger.debug(f"Sorted frequencies: {frequencies}")

        kT = self._run_manager.get_KT2J(temp)
        logger.debug(f"Temperature: {temp}, kT: {kT}")
        exponent = self._PLANCK_CONST * frequencies / kT
        logger.debug(f"Exponent values: {exponent}")
        power_positive = np.power(np.e, exponent)
        power_negative = np.power(np.e, -exponent)
        logger.debug(f"Power positive values: {power_positive}")
        logger.debug(f"Power negative values: {power_negative}")
        S_components = exponent / (power_positive - 1) - np.log(1 - power_negative)
        S_components = (
            S_components * self._GAS_CONST
        )  # multiply by R - get entropy in J mol^{-1} K^{-1}
        logger.debug(f"Entropy components: {S_components}")
        # N beads at a level => 3N x 3N covariance matrix => 3N eigenvalues
        if matrix_type == "force":  # force covariance matrix
            if (
                highest_level
            ):  # whole molecule level - we take all frequencies into account
                S_vib_total = sum(S_components)

            # discard the 6 lowest frequencies to discard translation and rotation of
            # the whole unit the overall translation and rotation of a unit is an
            # internal motion of the level above
            else:
                S_vib_total = sum(S_components[6:])

        else:  # torque covariance matrix - we always take all values into account
            S_vib_total = sum(S_components)

        logger.debug(f"Total vibrational entropy: {S_vib_total}")

        return S_vib_total


class ConformationalEntropy(EntropyManager):
    """
    Performs conformational entropy calculations based on molecular dynamics data.
    Inherits from EntropyManager and includes constants specific to conformational
    analysis using statistical mechanics principles.
    """

    def __init__(
        self, run_manager, args, universe, data_logger, level_manager, group_molecules
    ):
        """
        Initializes the ConformationalEntropy manager with all required components and
        sets the gas constant used in conformational entropy calculations.
        """
        super().__init__(
            run_manager, args, universe, data_logger, level_manager, group_molecules
        )

    def assign_conformation(
        self, data_container, dihedral, number_frames, bin_width, start, end, step
    ):
        """
        Create a state vector, showing the state in which the input dihedral is
        as a function of time. The function creates a histogram from the timeseries of
        the dihedral angle values and identifies points of dominant occupancy
        (called CONVEX TURNING POINTS).
        Based on the identified TPs, states are assigned to each configuration of the
        dihedral.

        Args:
            data_container (MDAnalysis Universe): data for the molecule/residue unit
            dihedral (array): The dihedral angles in the unit
            number_frames (int): number of frames in the trajectory
            bin_width (int): the width of the histogram bit, default 30 degrees
            start (int): starting frame, will default to 0
            end (int): ending frame, will default to -1 (last frame in trajectory)
            step (int): spacing between frames, will default to 1

        Returns:
            conformations (array): A timeseries with integer labels describing the
            state at each point in time.

        """
        conformations = np.zeros(number_frames)
        phi = np.zeros(number_frames)

        # get the values of the angle for the dihedral
        # dihedral angle values have a range from -180 to 180
        indices = list(range(number_frames))
        for timestep_index, _ in zip(
            indices, data_container.trajectory[start:end:step]
        ):
            timestep_index = timestep_index
            value = dihedral.value()
            # we want postive values in range 0 to 360 to make the peak assignment
            # works using the fact that dihedrals have circular symetry
            # (i.e. -15 degrees = +345 degrees)
            if value < 0:
                value += 360
            phi[timestep_index] = value

        # create a histogram using numpy
        number_bins = int(360 / bin_width)
        popul, bin_edges = np.histogram(a=phi, bins=number_bins, range=(0, 360))
        bin_value = [
            0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(0, len(popul))
        ]

        # identify "convex turning-points" and populate a list of peaks
        # peak : a bin whose neighboring bins have smaller population
        # NOTE might have problems if the peak is wide with a flat or sawtooth
        # top in which case check you have a sensible bin width
        peak_values = []

        for bin_index in range(number_bins):
            # if there is no dihedrals in a bin then it cannot be a peak
            if popul[bin_index] == 0:
                pass
            # being careful of the last bin
            # (dihedrals have circular symmetry, the histogram does not)
            elif (
                bin_index == number_bins - 1
            ):  # the -1 is because the index starts with 0 not 1
                if (
                    popul[bin_index] >= popul[bin_index - 1]
                    and popul[bin_index] >= popul[0]
                ):
                    peak_values.append(bin_value[bin_index])
            else:
                if (
                    popul[bin_index] >= popul[bin_index - 1]
                    and popul[bin_index] >= popul[bin_index + 1]
                ):
                    peak_values.append(bin_value[bin_index])

        # go through each frame again and assign conformation state
        for frame in range(number_frames):
            # find the TP that the snapshot is least distant from
            distances = [abs(phi[frame] - peak) for peak in peak_values]
            conformations[frame] = np.argmin(distances)

        logger.debug(f"Final conformations: {conformations}")

        return conformations

    def conformational_entropy_calculation(self, states, number_frames):
        """
        Function to calculate conformational entropies using eq. (7) in Higham,
        S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116,
        1965–1976 / eq. (4) in A. Chakravorty, J. Higham and R. H. Henchman,
        J. Chem. Inf. Model., 2020, 60, 5540–5551.

        Uses the adaptive enumeration method (AEM).

        Args:
           states (array): Conformational states in the molecule
           number_frames (int): The number of frames analysed

        Returns:
            S_conf_total (float) : conformational entropy
        """

        S_conf_total = 0

        # Count how many times each state occurs, then use the probability
        # to get the entropy
        # entropy = sum over states p*ln(p)
        values, counts = np.unique(states, return_counts=True)
        for state in range(len(values)):
            logger.debug(f"Unique states: {values}")
            logger.debug(f"Counts: {counts}")
            count = counts[state]
            probability = count / number_frames
            entropy = probability * np.log(probability)
            S_conf_total += entropy

        # multiply by gas constant to get the units J/mol/K
        S_conf_total *= -1 * self._GAS_CONST

        logger.debug(f"Total conformational entropy: {S_conf_total}")

        return S_conf_total


class OrientationalEntropy(EntropyManager):
    """
    Performs orientational entropy calculations using molecular dynamics data.
    Inherits from EntropyManager and includes constants relevant to rotational
    and orientational degrees of freedom.
    """

    def __init__(
        self, run_manager, args, universe, data_logger, level_manager, group_molecules
    ):
        """
        Initializes the OrientationalEntropy manager with all required components and
        sets the gas constant used in orientational entropy calculations.
        """
        super().__init__(
            run_manager, args, universe, data_logger, level_manager, group_molecules
        )

    def orientational_entropy_calculation(self, neighbours_dict):
        """
        Function to calculate orientational entropies from eq. (10) in J. Higham,
        S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116,
        3 1965–1976. Number of orientations, Ω, is calculated using eq. (8) in
        J. Higham, S.-Y. Chou, F. Gräter and R. H. Henchman,  Molecular Physics,
        2018, 116, 3 1965–1976.

        σ is assumed to be 1 for the molecules we're concerned with and hence,
        max {1, (Nc^3*π)^(1/2)} will always be (Nc^3*π)^(1/2).

        TODO future release - function for determing symmetry and symmetry numbers
        maybe?

        Input
        -----
        neighbours_dict :  dictionary - dictionary of neighbours for the molecule -
            should contain the type of neighbour molecule and the number of neighbour
            molecules of that species

        Returns
        -------
        S_or_total : float - orientational entropy
        """

        # Replaced molecule with neighbour as this is what the for loop uses
        S_or_total = 0
        for neighbour in neighbours_dict:  # we are going through neighbours
            if neighbour in ["H2O"]:  # water molecules - call POSEIDON functions
                pass  # TODO temporary until function is written
            else:
                # the bound ligand is always going to be a neighbour
                omega = np.sqrt((neighbours_dict[neighbour] ** 3) * math.pi)
                logger.debug(f"Omega for neighbour {neighbour}: {omega}")
                # orientational entropy arising from each neighbouring species
                # - we know the species is going to be a neighbour
                S_or_component = math.log(omega)
                logger.debug(
                    f"S_or_component (log(omega)) for neighbour {neighbour}: "
                    f"{S_or_component}"
                )
                S_or_component *= self._GAS_CONST
                logger.debug(
                    f"S_or_component after multiplying by GAS_CONST for neighbour "
                    f"{neighbour}: {S_or_component}"
                )
                S_or_total += S_or_component
                logger.debug(
                    f"S_or_total after adding component for neighbour {neighbour}: "
                    f"{S_or_total}"
                )
        # TODO for future releases
        # implement a case for molecules with hydrogen bonds but to a lesser
        # extent than water

        logger.debug(f"Final total orientational entropy: {S_or_total}")

        return S_or_total
