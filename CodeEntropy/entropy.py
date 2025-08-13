import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import waterEntropy.recipes.interfacial_solvent as GetSolvent
from numpy import linalg as la

logger = logging.getLogger(__name__)


class EntropyManager:
    """
    Manages entropy calculations at multiple molecular levels, based on a
    molecular dynamics trajectory.
    """

    def __init__(self, run_manager, args, universe, data_logger, level_manager):
        """
        Initializes the EntropyManager with required components.

        Args:
            run_manager: Manager for universe and selection operations.
            args: Argument namespace containing user parameters.
            universe: MDAnalysis universe representing the simulation system.
            data_logger: Logger for storing and exporting entropy data.
            level_manager: Provides level-specific data such as matrices and dihedrals.
        """
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._data_logger = data_logger
        self._level_manager = level_manager
        self._GAS_CONST = 8.3144598484848

    def execute(self):
        """
        Executes the full entropy computation workflow over selected molecules and
        levels. This includes both vibrational and conformational entropy, recorded
        per molecule and residue.
        """
        start, end, step = self._get_trajectory_bounds()
        number_frames = self._get_number_frames(start, end, step)

        has_water = self._universe.select_atoms("water").n_atoms > 0
        if has_water and self._args.water_entropy:
            self._calculate_water_entropy(self._universe, start, end, step)

            if self._args.selection_string != "all":
                self._args.selection_string += " and not water"
            else:
                self._args.selection_string = "not water"

            logger.debug(
                "WaterEntropy: molecule_data: %s",
                self._data_logger.molecule_data,
            )
            logger.debug(
                "WaterEntropy: residue_data: %s",
                self._data_logger.residue_data,
            )

        reduced_atom = self._get_reduced_universe()
        number_molecules, levels = self._level_manager.select_levels(reduced_atom)

        ve = VibrationalEntropy(
            self._run_manager,
            self._args,
            self._universe,
            self._data_logger,
            self._level_manager,
        )
        ce = ConformationalEntropy(
            self._run_manager,
            self._args,
            self._universe,
            self._data_logger,
            self._level_manager,
        )

        for molecule_id in range(number_molecules):
            mol_container = self._get_molecule_container(reduced_atom, molecule_id)

            for level in levels[molecule_id]:
                highest_level = level == levels[molecule_id][-1]
                if level == "united_atom":
                    self._process_united_atom_level(
                        molecule_id,
                        mol_container,
                        ve,
                        ce,
                        level,
                        start,
                        end,
                        step,
                        number_frames,
                        highest_level,
                    )

                    logger.debug(
                        "%s level: molecule_data: %s",
                        level,
                        self._data_logger.molecule_data,
                    )
                    logger.debug(
                        "%s level: residue_data: %s",
                        level,
                        self._data_logger.residue_data,
                    )

                elif level in ("polymer", "residue"):
                    self._process_vibrational_only_levels(
                        molecule_id,
                        mol_container,
                        ve,
                        level,
                        start,
                        end,
                        step,
                        number_frames,
                        highest_level,
                    )

                    logger.debug(
                        "%s level: molecule_data: %s",
                        level,
                        self._data_logger.molecule_data,
                    )
                    logger.debug(
                        "%s level: residue_data: %s",
                        level,
                        self._data_logger.residue_data,
                    )

                if level == "residue":
                    self._process_conformational_residue_level(
                        molecule_id,
                        mol_container,
                        ce,
                        level,
                        start,
                        end,
                        step,
                        number_frames,
                    )

                    logger.debug(
                        "%s level: molecule_data: %s",
                        level,
                        self._data_logger.molecule_data,
                    )
                    logger.debug(
                        "%s level: residue_data: %s",
                        level,
                        self._data_logger.residue_data,
                    )

        self._finalize_molecule_results()

        self._data_logger.log_tables()

    def _get_trajectory_bounds(self):
        """
        Returns the start, end, and step frame indices based on input arguments.

        Returns:
            Tuple of (start, end, step) frame indices.
        """
        start = self._args.start or 0
        end = self._args.end or -1
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
        trajectory_length = len(self._universe.trajectory)

        if start == 0 and end == -1 and step == 1:
            return trajectory_length

        if end == -1:
            end = trajectory_length
        else:
            end += 1

        return math.floor((end - start) / step)

    def _get_reduced_universe(self):
        """
        Applies atom selection based on the user's input.

        Returns:
            MDAnalysis.Universe: Selected subset of the system.
        """
        if self._args.selection_string == "all":
            return self._universe
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
        frag = universe.atoms.fragments[molecule_id]
        selection_string = f"index {frag.indices[0]}:{frag.indices[-1]}"
        return self._run_manager.new_U_select_atom(universe, selection_string)

    def _process_united_atom_level(
        self, mol_id, mol_container, ve, ce, level, start, end, step, n_frames, highest
    ):
        """
        Calculates translational, rotational, and conformational entropy at the
        united-atom level.

        Args:
            mol_id (int): ID of the molecule.
            mol_container (Universe): Universe for the selected molecule.
            ve: VibrationalEntropy object.
            ce: ConformationalEntropy object.
            level (str): Granularity level (should be 'united_atom').
            start, end, step (int): Trajectory frame parameters.
            n_frames (int): Number of trajectory frames.
            highest (bool): Whether this is the highest level of resolution for
            the molecule.
        """
        bin_width = self._args.bin_width
        S_trans, S_rot, S_conf = 0, 0, 0
        for residue_id, residue in enumerate(mol_container.residues):
            res_container = self._run_manager.new_U_select_atom(
                mol_container,
                f"index {residue.atoms.indices[0]}:{residue.atoms.indices[-1]}",
            )
            heavy_res = self._run_manager.new_U_select_atom(
                res_container, "not name H*"
            )

            force_matrix, torque_matrix = self._level_manager.get_matrices(
                res_container, level, start, end, step, n_frames, highest
            )

            S_trans_res = ve.vibrational_entropy_calculation(
                force_matrix, "force", self._args.temperature, highest
            )
            S_rot_res = ve.vibrational_entropy_calculation(
                torque_matrix, "torque", self._args.temperature, highest
            )

            dihedrals = self._level_manager.get_dihedrals(heavy_res, level)
            S_conf_res = ce.conformational_entropy_calculation(
                heavy_res, dihedrals, bin_width, start, end, step, n_frames
            )

            S_trans += S_trans_res
            S_rot += S_rot_res
            S_conf += S_conf_res

            self._data_logger.add_residue_data(
                mol_id, residue.resname, level, "Transvibrational", S_trans_res
            )
            self._data_logger.add_residue_data(
                mol_id, residue.resname, level, "Rovibrational", S_rot_res
            )
            self._data_logger.add_residue_data(
                mol_id, residue.resname, level, "Conformational", S_conf_res
            )

        self._data_logger.add_results_data(
            residue.resname, level, "Transvibrational", S_trans
        )
        self._data_logger.add_results_data(
            residue.resname, level, "Rovibrational", S_rot
        )
        self._data_logger.add_results_data(
            residue.resname, level, "Conformational", S_conf
        )

    def _process_vibrational_only_levels(
        self, mol_id, mol_container, ve, level, start, end, step, n_frames, highest
    ):
        """
        Calculates vibrational entropy at levels where conformational entropy is
        not considered.

        Args:
            mol_id (int): Molecule ID.
            mol_container (Universe): Selected molecule's universe.
            ve: VibrationalEntropy object.
            level (str): Current granularity level ('polymer' or 'residue').
            start, end, step (int): Trajectory frame parameters.
            n_frames (int): Number of trajectory frames.
            highest (bool): Flag indicating if this is the highest granularity
            level.
        """
        force_matrix, torque_matrix = self._level_manager.get_matrices(
            mol_container, level, start, end, step, n_frames, highest
        )
        S_trans = ve.vibrational_entropy_calculation(
            force_matrix, "force", self._args.temperature, highest
        )
        S_rot = ve.vibrational_entropy_calculation(
            torque_matrix, "torque", self._args.temperature, highest
        )
        residue = mol_container.residues[mol_id]
        self._data_logger.add_results_data(
            residue.resname, level, "Transvibrational", S_trans
        )
        self._data_logger.add_results_data(
            residue.resname, level, "Rovibrational", S_rot
        )

    def _process_conformational_residue_level(
        self, mol_id, mol_container, ce, level, start, end, step, n_frames
    ):
        """
        Computes conformational entropy at the residue level (whole-molecule dihedral
        analysis).

        Args:
            mol_id (int): ID of the molecule.
            mol_container (Universe): Selected molecule's universe.
            ce: ConformationalEntropy object.
            level (str): Level name (should be 'residue').
            start, end, step (int): Frame bounds.
            n_frames (int): Number of frames used.
        """
        bin_width = self._args.bin_width
        dihedrals = self._level_manager.get_dihedrals(mol_container, level)
        S_conf = ce.conformational_entropy_calculation(
            mol_container, dihedrals, bin_width, start, end, step, n_frames
        )
        residue = mol_container.residues[mol_id]
        self._data_logger.add_results_data(
            residue.resname, level, "Conformational", S_conf
        )

    def _finalize_molecule_results(self):
        """
        Aggregates and logs total entropy per molecule using residue_data grouped by
        resid.
        """
        entropy_by_molecule = defaultdict(float)

        for mol_id, level, entropy_type, result in self._data_logger.molecule_data:
            if level != "Molecule Total":
                try:
                    entropy_by_molecule[mol_id] += float(result)
                except ValueError:
                    logger.warning(f"Skipping invalid entry: {mol_id}, {result}")

        for mol_id, total_entropy in entropy_by_molecule.items():
            self._data_logger.molecule_data.append(
                (mol_id, "Molecule Total", "Molecule Total Entropy", total_entropy)
            )

        # Save to file
        self._data_logger.save_dataframes_as_json(
            pd.DataFrame(
                self._data_logger.molecule_data,
                columns=["Molecule ID", "Level", "Type", "Result (J/mol/K)"],
            ),
            pd.DataFrame(
                self._data_logger.residue_data,
                columns=[
                    "Residue ID",
                    "Residue Name",
                    "Level",
                    "Type",
                    "Result (J/mol/K)",
                ],
            ),
            self._args.output_file,
        )

    def _calculate_water_entropy(self, universe, start, end, step):
        """
        Calculates orientational and vibrational entropy for water molecules.

        Args:
            universe: MDAnalysis Universe object.
            start (int): Start frame.
            end (int): End frame.
            step (int): Step size.
        """
        Sorient_dict, _, vibrations, _, _ = (
            GetSolvent.get_interfacial_water_orient_entropy(
                universe, start, end, step, self._args.temperature, parallel=True
            )
        )

        # Log per-residue entropy using helper functions
        self._calculate_water_orientational_entropy(Sorient_dict)
        self._calculate_water_vibrational_translational_entropy(vibrations)
        self._calculate_water_vibrational_rotational_entropy(vibrations)

        # Aggregate entropy components per molecule
        results = {}

        for row in self._data_logger.residue_data:
            mol_id = row[1]
            entropy_type = row[3].split()[0]
            value = float(row[4])

            if mol_id not in results:
                results[mol_id] = {
                    "Orientational": 0.0,
                    "Transvibrational": 0.0,
                    "Rovibrational": 0.0,
                }

            results[mol_id][entropy_type] += value

        # Log per-molecule entropy components and total
        for mol_id, components in results.items():
            total = 0.0
            for entropy_type in ["Orientational", "Transvibrational", "Rovibrational"]:
                S_component = components[entropy_type]
                self._data_logger.add_results_data(
                    mol_id, "water", entropy_type, S_component
                )
                total += S_component

    def _calculate_water_orientational_entropy(self, Sorient_dict):
        """
        Logs orientational entropy values directly from Sorient_dict.
        """
        for resid, resname_dict in Sorient_dict.items():
            for resname, values in resname_dict.items():
                if isinstance(values, list) and len(values) == 2:
                    Sor, count = values
                    self._data_logger.add_residue_data(
                        resid, resname, "Water", "Orientational", Sor
                    )

    def _calculate_water_vibrational_translational_entropy(self, vibrations):
        """
        Logs summed translational entropy values per residue-solvent pair.
        """
        for (solute_id, _), entropy in vibrations.translational_S.items():
            if isinstance(entropy, (list, np.ndarray)):
                entropy = float(np.sum(entropy))

            if "_" in solute_id:
                resname, resid_str = solute_id.rsplit("_", 1)
                try:
                    resid = int(resid_str)
                except ValueError:
                    resid = -1
            else:
                resname = solute_id
                resid = -1

            self._data_logger.add_residue_data(
                resid, resname, "Water", "Transvibrational", entropy
            )

    def _calculate_water_vibrational_rotational_entropy(self, vibrations):
        """
        Logs summed rotational entropy values per residue-solvent pair.
        """
        for (solute_id, _), entropy in vibrations.rotational_S.items():
            if isinstance(entropy, (list, np.ndarray)):
                entropy = float(np.sum(entropy))

            if "_" in solute_id:
                resname, resid_str = solute_id.rsplit("_", 1)
                try:
                    resid = int(resid_str)
                except ValueError:
                    resid = -1
            else:
                resname = solute_id
                resid = -1

            self._data_logger.add_residue_data(
                resid, resname, "Water", "Rovibrational", entropy
            )


class VibrationalEntropy(EntropyManager):
    """
    Performs vibrational entropy calculations using molecular trajectory data.
    Extends the base EntropyManager with constants and logic specific to
    vibrational modes and thermodynamic properties.
    """

    def __init__(self, run_manager, args, universe, data_logger, level_manager):
        """
        Initializes the VibrationalEntropy manager with all required components and
        defines physical constants used in vibrational entropy calculations.
        """
        super().__init__(run_manager, args, universe, data_logger, level_manager)
        self._PLANCK_CONST = 6.62607004081818e-34

    def frequency_calculation(self, lambdas, temp):
        """
        Function to calculate an array of vibrational frequencies from the eigenvalues
        of the covariance matrix.

        Calculated from eq. (3) in Higham, S.-Y. Chou, F. Gräter and  R. H. Henchman,
        Molecular Physics, 2018, 116, 1965–1976//eq. (3) in A. Chakravorty, J. Higham
        and R. H. Henchman, J. Chem. Inf. Model., 2020, 60, 5540–5551

        frequency=sqrt(λ/kT)/2π

        Input
        -----
        lambdas : array of floats - eigenvalues of the covariance matrix
        temp: float - temperature

        Returns
        -------
        frequencies : array of floats - corresponding vibrational frequencies
        """
        pi = np.pi
        # get kT in Joules from given temperature
        kT = self._run_manager.get_KT2J(temp)
        logger.debug(f"Temperature: {temp}, kT: {kT}")

        lambdas = np.array(lambdas)  # Ensure input is a NumPy array
        logger.debug(f"Eigenvalues (lambdas): {lambdas}")

        # Check for negatives and raise an error if any are found
        if np.any(lambdas < 0):
            logger.error(f"Negative eigenvalues encountered: {lambdas[lambdas < 0]}")
            raise ValueError(
                f"Negative eigenvalues encountered: {lambdas[lambdas < 0]}"
            )

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

        Input
        -----
        matrix : matrix - force/torque covariance matrix
        matrix_type: string
        temp: float - temperature
        highest_level: bool - is this the highest level of the heirarchy

        Returns
        -------
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

    def __init__(self, run_manager, args, universe, data_logger, level_manager):
        """
        Initializes the ConformationalEntropy manager with all required components and
        sets the gas constant used in conformational entropy calculations.
        """
        super().__init__(run_manager, args, universe, data_logger, level_manager)

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

        Input
        -----
        dihedral_atom_group : the group of 4 atoms defining the dihedral
        number_frames : number of frames in the trajectory
        bin_width : the width of the histogram bit, default 30 degrees
        start : int, starting frame, will default to 0
        end : int, ending frame, will default to -1 (last frame in trajectory)
        step : int, spacing between frames, will default to 1

        Return
        ------
        A timeseries with integer labels describing the state at each point in time.

        """
        conformations = np.zeros(number_frames)
        phi = np.zeros(number_frames)

        # get the values of the angle for the dihedral
        # dihedral angle values have a range from -180 to 180
        for timestep in data_container.trajectory[start:end:step]:
            timestep_index = timestep.frame - start
            value = dihedral.value()
            # we want postive values in range 0 to 360 to make the peak assignment
            # work using the fact that dihedrals have circular symetry
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
        # NOTE might have problems if the peak is wide with a flat or sawtooth top
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

    def conformational_entropy_calculation(
        self, data_container, dihedrals, bin_width, start, end, step, number_frames
    ):
        """
        Function to calculate conformational entropies using eq. (7) in Higham,
        S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116,
        1965–1976 / eq. (4) in A. Chakravorty, J. Higham and R. H. Henchman,
        J. Chem. Inf. Model., 2020, 60, 5540–5551.

        Uses the adaptive enumeration method (AEM).

        Input
        -----
        dihedrals : array - array of dihedrals in the molecule
        Returns
        -------
        S_conf_total : float - conformational entropy
        """

        S_conf_total = 0

        # For each dihedral, identify the conformation in each frame
        num_dihedrals = len(dihedrals)
        conformation = np.zeros((num_dihedrals, number_frames))
        index = 0
        for dihedral in dihedrals:
            conformation[index] = self.assign_conformation(
                data_container, dihedral, number_frames, bin_width, start, end, step
            )
            index += 1

        logger.debug(f"Conformation matrix: {conformation}")

        # For each frame, convert the conformation of all dihedrals into a
        # state string
        states = ["" for x in range(number_frames)]
        for frame_index in range(number_frames):
            for index in range(num_dihedrals):
                states[frame_index] += str(conformation[index][frame_index])

        logger.debug(f"States: {states}")

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

    def __init__(self, run_manager, args, universe, data_logger, level_manager):
        """
        Initializes the OrientationalEntropy manager with all required components and
        sets the gas constant used in orientational entropy calculations.
        """
        super().__init__(run_manager, args, universe, data_logger, level_manager)

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
