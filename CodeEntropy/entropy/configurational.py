"""Conformational entropy utilities.

This module provides:

- Assignment of discrete conformational states for a single dihedral time series.
- Computation of conformational entropy from a sequence of state labels.

"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ConformationalEntropy:
    """Compute conformational entropy from states information."""

    _GAS_CONST: float = 8.3144598484848

    def __init__(self) -> None:
        """Math-only engine.

        This class assigns conformational states and computes conformational entropy.
        It does not depend on the workflow runner, universe, grouping, or reporting.
        """
        pass

    def conformational_entropy_calculation(self, states: Any) -> float:
        """Compute conformational entropy for a sequence of state labels.

        Entropy is computed as:

            S = -R * sum_i p_i * ln(p_i)

        where p_i is the observed probability of state i in ``states``.

        Args:
            states: Sequence/array of discrete state labels. Empty/None yields 0.0.
            number_frames: Frame count metadata.

        Returns:
            float: Conformational entropy in J/mol/K.
        """
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
        logger.debug(f"Total conformational entropy: {s_conf}")
        return s_conf

    @staticmethod
    def _to_1d_array(states: Any) -> np.ndarray | None:
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
