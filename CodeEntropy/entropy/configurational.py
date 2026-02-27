"""Conformational entropy utilities.

This module provides:
  * Assigning discrete conformational states for a single dihedral time series.
  * Computing conformational entropy from a sequence of state labels.

The public surface area is intentionally small to keep responsibilities clear.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformationConfig:
    """Configuration for assigning conformational states from a dihedral.

    Attributes:
        bin_width: Histogram bin width in degrees for peak detection.
        start: Inclusive start frame index for trajectory slicing.
        end: Exclusive end frame index for trajectory slicing.
        step: Stride for trajectory slicing (must be positive).
    """

    bin_width: int
    start: int
    end: int
    step: int


class ConformationalEntropy:
    """Assign dihedral conformational states and compute conformational entropy.

    This class contains two independent responsibilities:
      1) `assign_conformation`: Map a single dihedral angle time series to discrete
         state labels by detecting histogram peaks and assigning the nearest peak.
      2) `conformational_entropy_calculation`: Compute Shannon entropy of the
         state distribution (in J/mol/K).

    Notes:
        `number_frames` is accepted by `conformational_entropy_calculation` for
        compatibility with calling sites that track frame counts, but the entropy
        is computed from the observed state counts (i.e., `len(states)`), which is
        the correct normalization for the sampled distribution.
    """

    _GAS_CONST: float = 8.3144598484848

    def __init__(self) -> None:
        """Math-only engine.

        This class assigns conformational states and computes conformational entropy.
        It does not depend on the workflow runner, universe, grouping, or reporting.
        """
        pass

    def assign_conformation(
        self,
        data_container: Any,
        dihedral: Any,
        number_frames: int,
        bin_width: int,
        start: int,
        end: int,
        step: int,
    ) -> np.ndarray:
        """Assign discrete conformational states for a single dihedral.

        The dihedral angle time series is:
          1) Collected across the trajectory slice [start:end:step].
          2) Converted to [0, 360) degrees.
          3) Histogrammed using `bin_width`.
          4) Peaks are identified as bins with locally maximal population.
          5) Each frame is assigned the index of the nearest peak.

        Args:
            data_container: MDAnalysis Universe/AtomGroup with a trajectory.
            dihedral: Object providing `value()` for the current frame dihedral.
            number_frames: Provided for call-site compatibility; not used for sizing.
            bin_width: Histogram bin width in degrees.
            start: Inclusive start frame index.
            end: Exclusive end frame index.
            step: Stride for trajectory slicing.

        Returns:
            Array of integer state labels of length equal to the trajectory slice.
            Returns an empty array if the slice is empty.

        Raises:
            ValueError: If `bin_width` or `step` are invalid.
        """
        _ = number_frames

        config = ConformationConfig(
            bin_width=int(bin_width),
            start=int(start),
            end=int(end),
            step=int(step),
        )
        self._validate_assignment_config(config)

        traj_slice = data_container.trajectory[config.start : config.end : config.step]
        n_slice = len(traj_slice)
        if n_slice <= 0:
            return np.array([], dtype=int)

        phi = self._collect_dihedral_angles(traj_slice, dihedral)
        peak_values = self._find_histogram_peaks(phi, config.bin_width)

        if peak_values.size == 0:
            return np.zeros(n_slice, dtype=int)

        states = self._assign_nearest_peaks(phi, peak_values)
        logger.debug("Final conformations: %s", states)
        return states

    def conformational_entropy_calculation(
        self, states: Any, number_frames: int
    ) -> float:
        """Compute conformational entropy for a sequence of state labels.

        Entropy is computed as:
            S = -R * sum_i p_i * ln(p_i)
        where p_i is the observed probability of state i in `states`.

        Args:
            states: Sequence/array of discrete state labels. Empty/None yields 0.0.
            number_frames: Frame count metadata.

        Returns:
            Conformational entropy in J/mol/K.
        """
        _ = number_frames

        arr = self._to_1d_array(states)
        if arr is None or arr.size == 0:
            return 0.0

        values, counts = np.unique(arr, return_counts=True)
        total_count = int(np.sum(counts))
        if total_count <= 0 or values.size <= 1:
            return 0.0

        probs = counts.astype(float) / float(total_count)
        probs = probs[probs > 0.0]

        s_conf = -self._GAS_CONST * float(np.sum(probs * np.log(probs)))
        logger.debug("Total conformational entropy: %s", s_conf)
        return s_conf

    @staticmethod
    def _validate_assignment_config(config: ConformationConfig) -> None:
        """Validate conformation assignment configuration.

        Args:
            config: Assignment configuration.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if config.step <= 0:
            raise ValueError("step must be a positive integer")
        if config.bin_width <= 0 or config.bin_width > 360:
            raise ValueError("bin_width must be in the range (0, 360]")
        if 360 % config.bin_width != 0:
            logger.warning(
                "bin_width=%s does not evenly divide 360; histogram bins will be "
                "uneven.",
                config.bin_width,
            )

    @staticmethod
    def _collect_dihedral_angles(traj_slice: Any, dihedral: Any) -> np.ndarray:
        """Collect dihedral angles for each frame in the trajectory slice.

        Args:
            traj_slice: Slice of a trajectory iterable where iterating advances frames.
            dihedral: Object with `value()` returning the dihedral in degrees.

        Returns:
            Array of dihedral values mapped into [0, 360).
        """
        phi = np.zeros(len(traj_slice), dtype=float)
        for i, _ts in enumerate(traj_slice):
            value = float(dihedral.value())
            if value < 0.0:
                value += 360.0
            phi[i] = value
        return phi

    @staticmethod
    def _find_histogram_peaks(phi: np.ndarray, bin_width: int) -> np.ndarray:
        """Identify peak bin centers from a histogram of dihedral angles.

        A peak is defined as a bin whose population is greater than or equal to
        its immediate neighbors (with circular handling at the final bin).

        Args:
            phi: Dihedral angles in degrees, in [0, 360).
            bin_width: Histogram bin width in degrees.

        Returns:
            1D array of peak bin center values (degrees). Empty if no peaks found.
        """
        number_bins = int(360 / bin_width)
        popul, bin_edges = np.histogram(phi, bins=number_bins, range=(0.0, 360.0))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        peaks: list[float] = []
        for idx in range(number_bins):
            if popul[idx] == 0:
                continue

            left = popul[idx - 1] if idx > 0 else popul[number_bins - 1]
            right = popul[idx + 1] if idx < number_bins - 1 else popul[0]

            if popul[idx] >= left and popul[idx] > right:
                peaks.append(float(bin_centers[idx]))

        return np.asarray(peaks, dtype=float)

    @staticmethod
    def _assign_nearest_peaks(phi: np.ndarray, peak_values: np.ndarray) -> np.ndarray:
        """Assign each phi value to the index of its nearest peak.

        Args:
            phi: Dihedral angles in degrees.
            peak_values: Peak centers (degrees).

        Returns:
            Integer state labels aligned with `phi`.
        """
        distances = np.abs(phi[:, None] - peak_values[None, :])
        return np.argmin(distances, axis=1).astype(int)

    @staticmethod
    def _to_1d_array(states: Any) -> Optional[np.ndarray]:
        """Convert a state sequence into a 1D numpy array.

        Args:
            states: Input sequence/array.

        Returns:
            1D numpy array, or None if input is not usable.
        """
        if states is None:
            return None

        if isinstance(states, np.ndarray):
            arr = states.reshape(-1)
        else:
            try:
                arr = np.asarray(list(states)).reshape(-1)
            except TypeError:
                return None

        return arr
