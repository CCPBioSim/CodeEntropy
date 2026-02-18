import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConformationalEntropy:
    def __init__(self, run_manager, args, universe, data_logger, group_molecules):
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._data_logger = data_logger
        self._group_molecules = group_molecules

        self._GAS_CONST = 8.3144598484848

    def assign_conformation(
        self, data_container, dihedral, number_frames, bin_width, start, end, step
    ):
        """
        Build a conformation/state time series for ONE dihedral using the same
        logic as the procedural approach (histogram peaks -> nearest peak index),
        but with correct handling of start/end/step.

        NOTE: `number_frames` is ignored for sizing; we size to the slice length
        to avoid mismatches that cause invalid probabilities later.
        """
        traj_slice = data_container.trajectory[start:end:step]
        n = len(traj_slice)

        if n <= 0:
            return np.array([], dtype=int)

        phi = np.zeros(n, dtype=float)

        k = 0
        for _ts in traj_slice:
            value = float(dihedral.value())
            if value < 0:
                value += 360.0
            phi[k] = value
            k += 1

        number_bins = int(360 / bin_width)
        popul, bin_edges = np.histogram(phi, bins=number_bins, range=(0.0, 360.0))
        bin_value = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        peak_values = []
        for bin_index in range(number_bins):
            if popul[bin_index] == 0:
                continue

            if bin_index == number_bins - 1:
                if (
                    popul[bin_index] >= popul[bin_index - 1]
                    and popul[bin_index] >= popul[0]
                ):
                    peak_values.append(float(bin_value[bin_index]))
            else:
                if (
                    popul[bin_index] >= popul[bin_index - 1]
                    and popul[bin_index] >= popul[bin_index + 1]
                ):
                    peak_values.append(float(bin_value[bin_index]))

        if not peak_values:
            return np.zeros(n, dtype=int)

        peak_values = np.asarray(peak_values, dtype=float)

        conformations = np.zeros(n, dtype=int)
        for i in range(n):
            distances = np.abs(phi[i] - peak_values)
            conformations[i] = int(np.argmin(distances))

        logger.debug(f"Final conformations: {conformations}")
        return conformations

    def conformational_entropy_calculation(self, states, number_frames):
        """
        Procedural parity:
        - probabilities are computed using total_count = len(states)
        - number_frames is NOT used as the denominator (it is only metadata)
        """
        if states is None:
            return 0.0

        if isinstance(states, np.ndarray):
            states = states.reshape(-1)

        try:
            if len(states) == 0:
                return 0.0
        except TypeError:
            return 0.0

        try:
            if not any(states):
                return 0.0
        except TypeError:
            pass

        values, counts = np.unique(states, return_counts=True)
        total_count = int(np.sum(counts))
        if total_count <= 0:
            return 0.0

        S_conf_total = 0.0
        for c in counts:
            p = float(c) / float(total_count)
            S_conf_total += p * np.log(p)

        S_conf_total *= -1.0 * self._GAS_CONST
        logger.debug(f"Total conformational entropy: {S_conf_total}")
        return float(S_conf_total)
