import logging
import math

import numpy as np
import pandas as pd
from numpy import linalg as la

logger = logging.getLogger(__name__)


class EntropyManager:
    """
    Base class for running entropy calculations. Coordinates the main analysis workflow
    by integrating configuration, trajectory data, logging, and level management.
    Stores and organizes results at both the molecule and residue level.
    """

    def __init__(self, run_manager, args, universe, data_logger, level_manager):
        """
        Initializes the EntropyManager with all required components for running the
        entropy calculation, including configuration, trajectory data, logging, and
        level handling. Sets up internal dataframes for storing results.
        """
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._data_logger = data_logger
        self._level_manager = level_manager
        self._GAS_CONST = 8.3144598484848

        self._results_df = pd.DataFrame(
            columns=["Molecule ID", "Level", "Type", "Result"]
        )
        self._residue_results_df = pd.DataFrame(
            columns=["Molecule ID", "Residue", "Type", "Result"]
        )

    @property
    def results_df(self):
        """Returns the dataframe containing molecule-level entropy results."""
        return self._results_df

    @property
    def residue_results_df(self):
        """Returns the dataframe containing residue-level entropy results."""
        return self._residue_results_df

    def execute(self):
        """
        Runs the main entropy calculation workflow. This method should be implemented
        by subclasses to perform specific analysis steps and populate the results
        dataframes.
        """
        # Define bin_width for histogram from inputs
        bin_width = self._args.bin_width

        # Define trajectory slicing from inputs
        start = self._args.start
        if start is None:
            start = 0
        end = self._args.end
        if end is None:
            end = -1
        step = self._args.step
        if step is None:
            step = 1
        # Count number of frames, easy if not slicing
        if start == 0 and end == -1 and step == 1:
            number_frames = len(self._universe.trajectory)
        elif end == -1:
            end = len(self._universe.trajectory)
            number_frames = math.floor((end - start) / step) + 1
        else:
            number_frames = math.floor((end - start) / step) + 1
        logger.debug(f"Number of Frames: {number_frames}")

        # Create pandas data frame for results
        results_df = pd.DataFrame(columns=["Molecule ID", "Level", "Type", "Result"])
        residue_results_df = pd.DataFrame(
            columns=["Molecule ID", "Residue", "Type", "Result"]
        )

        # Reduce number of atoms in MDA universe to selection_string arg
        # (default all atoms included)
        if self._args.selection_string == "all":
            reduced_atom = self._universe
        else:
            reduced_atom = self._run_manager.new_U_select_atom(
                self._universe, self._args.selection_string
            )
            reduced_atom_name = (
                f"{len(reduced_atom.trajectory)}_frame_dump_atom_selection"
            )
            self._run_manager.write_universe(reduced_atom, reduced_atom_name)

        # Scan system for molecules and select levels (united atom, residue, polymer)
        # for each
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
        # oe = OrientationalEntropy(
        #     self._run_manager,
        #     self._args,
        #     self._universe,
        #     self._data_logger,
        #     self._level_manager,
        # )

        # Loop over molecules
        for molecule in range(number_molecules):
            # molecule data container of MDAnalysis Universe type for internal degrees
            # of freedom getting indices of first and last atoms in the molecule
            # assuming atoms are numbered consecutively and all atoms in a given
            # molecule are together
            index1 = reduced_atom.atoms.fragments[molecule].indices[0]
            index2 = reduced_atom.atoms.fragments[molecule].indices[-1]
            selection_string = f"index {index1}:{index2}"
            molecule_container = self._run_manager.new_U_select_atom(
                reduced_atom, selection_string
            )

            # Calculate entropy for each relevent level
            for level in levels[molecule]:
                if level == levels[molecule][-1]:
                    highest_level = True
                else:
                    highest_level = False

                if level == "united_atom":
                    # loop over residues, report results per residue + total united atom
                    # level. This is done per residue to reduce the size of the matrices
                    # - amino acid resiudes have tens of united atoms but a whole
                    # protein could have thousands. Doing the calculation per residue
                    # allows for comparisons of contributions from different residues
                    num_residues = len(molecule_container.residues)
                    S_trans = 0
                    S_rot = 0
                    S_conf = 0

                    for residue in range(num_residues):
                        # molecule data container of MDAnalysis Universe type for
                        # internal degrees of freedom getting indices of first and
                        # last atoms in the molecule assuming atoms are numbered
                        # consecutively and all atoms in a given molecule are together
                        index1 = molecule_container.residues[residue].atoms.indices[0]
                        index2 = molecule_container.residues[residue].atoms.indices[-1]
                        selection_string = f"index {index1}:{index2}"
                        residue_container = self._run_manager.new_U_select_atom(
                            molecule_container, selection_string
                        )
                        residue_heavy_atoms_container = (
                            self._run_manager.new_U_select_atom(
                                residue_container, "not name H*"
                            )
                        )  # only heavy atom dihedrals are relevant

                        # Vibrational entropy at every level
                        # Get the force and torque matrices for the beads at the
                        # relevant level

                        force_matrix, torque_matrix = self._level_manager.get_matrices(
                            residue_container,
                            level,
                            start,
                            end,
                            step,
                            number_frames,
                            highest_level,
                        )

                        # Calculate the entropy from the diagonalisation of the matrices
                        S_trans_residue = ve.vibrational_entropy_calculation(
                            force_matrix, "force", self._args.temperature, highest_level
                        )
                        S_trans += S_trans_residue
                        logger.debug(f"S_trans_{level}_{residue} = {S_trans_residue}")
                        new_row = pd.DataFrame(
                            {
                                "Molecule ID": [molecule],
                                "Residue": [residue],
                                "Type": ["Transvibrational (J/mol/K)"],
                                "Result": [S_trans_residue],
                            }
                        )
                        residue_results_df = pd.concat(
                            [residue_results_df, new_row], ignore_index=True
                        )
                        self._data_logger.add_residue_data(
                            molecule, residue, "Transvibrational", S_trans_residue
                        )

                        S_rot_residue = ve.vibrational_entropy_calculation(
                            torque_matrix,
                            "torque",
                            self._args.temperature,
                            highest_level,
                        )
                        S_rot += S_rot_residue
                        logger.debug(f"S_rot_{level}_{residue} = {S_rot_residue}")
                        new_row = pd.DataFrame(
                            {
                                "Molecule ID": [molecule],
                                "Residue": [residue],
                                "Type": ["Rovibrational (J/mol/K)"],
                                "Result": [S_rot_residue],
                            }
                        )
                        residue_results_df = pd.concat(
                            [residue_results_df, new_row], ignore_index=True
                        )
                        self._data_logger.add_residue_data(
                            molecule, residue, "Rovibrational", S_rot_residue
                        )

                        # Conformational entropy based on atom dihedral angle
                        # distribution. Gives entropy of conformations within
                        # each residue

                        # Get dihedral angle distribution
                        dihedrals = self._level_manager.get_dihedrals(
                            residue_heavy_atoms_container, level
                        )

                        # Calculate conformational entropy
                        S_conf_residue = ce.conformational_entropy_calculation(
                            residue_heavy_atoms_container,
                            dihedrals,
                            bin_width,
                            start,
                            end,
                            step,
                            number_frames,
                        )
                        S_conf += S_conf_residue
                        logger.debug(f"S_conf_{level}_{residue} = {S_conf_residue}")
                        new_row = pd.DataFrame(
                            {
                                "Molecule ID": [molecule],
                                "Residue": [residue],
                                "Type": ["Conformational (J/mol/K)"],
                                "Result": [S_conf_residue],
                            }
                        )
                        residue_results_df = pd.concat(
                            [residue_results_df, new_row], ignore_index=True
                        )
                        self._data_logger.add_residue_data(
                            molecule, residue, "Conformational", S_conf_residue
                        )

                    # Print united atom level results summed over all residues
                    logger.debug(f"S_trans_{level} = {S_trans}")
                    new_row = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Level": [level],
                            "Type": ["Transvibrational (J/mol/K)"],
                            "Result": [S_trans],
                        }
                    )

                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                    self._data_logger.add_results_data(
                        molecule, level, "Transvibrational", S_trans
                    )

                    logger.debug(f"S_rot_{level} = {S_rot}")
                    new_row = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Level": [level],
                            "Type": ["Rovibrational (J/mol/K)"],
                            "Result": [S_rot],
                        }
                    )
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                    self._data_logger.add_results_data(
                        molecule, level, "Rovibrational", S_rot
                    )
                    logger.debug(f"S_conf_{level} = {S_conf}")

                    new_row = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Level": [level],
                            "Type": ["Conformational (J/mol/K)"],
                            "Result": [S_conf],
                        }
                    )
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                    self._data_logger.add_results_data(
                        molecule, level, "Conformational", S_conf
                    )

                if level in ("polymer", "residue"):
                    # Vibrational entropy at every level
                    # Get the force and torque matrices for the beads at the relevant
                    # level
                    force_matrix, torque_matrix = self._level_manager.get_matrices(
                        molecule_container,
                        level,
                        start,
                        end,
                        step,
                        number_frames,
                        highest_level,
                    )

                    # Calculate the entropy from the diagonalisation of the matrices
                    S_trans = ve.vibrational_entropy_calculation(
                        force_matrix, "force", self._args.temperature, highest_level
                    )
                    logger.debug(f"S_trans_{level} = {S_trans}")

                    # Create new row as a DataFrame for Transvibrational
                    new_row_trans = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Level": [level],
                            "Type": ["Transvibrational (J/mol/K)"],
                            "Result": [S_trans],
                        }
                    )

                    # Concatenate the new row to the DataFrame
                    results_df = pd.concat(
                        [results_df, new_row_trans], ignore_index=True
                    )

                    # Calculate the entropy for Rovibrational
                    S_rot = ve.vibrational_entropy_calculation(
                        torque_matrix, "torque", self._args.temperature, highest_level
                    )
                    logger.debug(f"S_rot_{level} = {S_rot}")

                    # Create new row as a DataFrame for Rovibrational
                    new_row_rot = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Level": [level],
                            "Type": ["Rovibrational (J/mol/K)"],
                            "Result": [S_rot],
                        }
                    )

                    # Concatenate the new row to the DataFrame
                    results_df = pd.concat([results_df, new_row_rot], ignore_index=True)

                    self._data_logger.add_results_data(
                        molecule, level, "Transvibrational", S_trans
                    )
                    self._data_logger.add_results_data(
                        molecule, level, "Rovibrational", S_rot
                    )

                    # Note: conformational entropy is not calculated at the polymer
                    # level, because there is at most one polymer bead per molecule
                    # so no dihedral angles.

                if level == "residue":
                    # Conformational entropy based on distributions of dihedral angles
                    # of residues. Gives conformational entropy of secondary structure

                    # Get dihedral angle distribution
                    dihedrals = self._level_manager.get_dihedrals(
                        molecule_container, level
                    )
                    # Calculate conformational entropy
                    S_conf = ce.conformational_entropy_calculation(
                        molecule_container,
                        dihedrals,
                        bin_width,
                        start,
                        end,
                        step,
                        number_frames,
                    )
                    logger.debug(f"S_conf_{level} = {S_conf}")
                    new_row = pd.DataFrame(
                        {
                            "Molecule ID": [molecule],
                            "Level": [level],
                            "Type": ["Conformational (J/mol/K)"],
                            "Result": [S_conf],
                        }
                    )
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    self._data_logger.add_results_data(
                        molecule, level, "Conformational", S_conf
                    )

                # Orientational entropy based on network of neighbouring molecules,
                #  only calculated at the highest level (whole molecule)
                # if highest_level:
                #     neigbours = self._level_manager.get_neighbours(
                #         reduced_atom, molecule
                #     )
                #     S_orient = EF.orientational_entropy(neighbours)
                #     print(f"S_orient_{level} = {S_orient}")
                #     new_row = pd.DataFrame(
                #         {
                #             "Molecule ID": [molecule],
                #             "Level": [level],
                #             "Type": ["Orientational (J/mol/K)"],
                #             "Result": [S_orient],
                #         }
                #     )
                #     results_df = pd.concat([results_df, new_row], ignore_index=True)
                #     with open(self._args.output_file, "a") as out:
                #         print(
                #             molecule,
                #             "\t",
                #             level,
                #             "\tOrientational\t",
                #             S_orient,
                #             file=out,
                #         )

            # Report total entropy for the molecule
            S_molecule = results_df[results_df["Molecule ID"] == molecule][
                "Result"
            ].sum()
            logger.debug(f"S_molecule = {S_molecule}")
            new_row = pd.DataFrame(
                {
                    "Molecule ID": [molecule],
                    "Level": ["Molecule Total"],
                    "Type": ["Molecule Total Entropy "],
                    "Result": [S_molecule],
                }
            )
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            self._data_logger.add_results_data(
                molecule, level, "Molecule Total Entropy", S_molecule
            )
            self._data_logger.save_dataframes_as_json(
                results_df, residue_results_df, self._args.output_file
            )

        logger.info("Molecules:")
        self._data_logger.log_tables()


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
            if neighbour in []:  # water molecules - call POSEIDON functions
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
                S_or_component *= self.GAS_CONST
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
