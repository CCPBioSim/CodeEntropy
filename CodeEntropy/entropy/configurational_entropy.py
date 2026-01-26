import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConformationalEntropy:
    def __init__(
        self, run_manager, args, universe, data_logger, level_manager, group_molecules
    ):
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._data_logger = data_logger
        self._level_manager = level_manager
        self._group_molecules = group_molecules

        self._GAS_CONST = 8.3144598484848

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
