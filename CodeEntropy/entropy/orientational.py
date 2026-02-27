"""Orientational entropy calculations.

This module defines `OrientationalEntropy`, which computes orientational entropy
from a neighbour count and symmetry information.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_GAS_CONST_J_PER_MOL_K = 8.3144598484848


@dataclass(frozen=True)
class OrientationalEntropyResult:
    """Result of an orientational entropy calculation.

    Attributes:
        total: Total orientational entropy (J/mol/K).
    """

    total: float


class OrientationalEntropy:
    """Compute orientational entropy from neighbor counts.

    This class is intentionally small and focused: it provides a single public
    method that converts a mapping of neighbor species to neighbor counts into
    an orientational entropy value.

    Notes:
        The manager-like constructor signature is kept for compatibility with
        the rest of the codebase, but the calculation itself does not depend on
        those objects.
    """

    def __init__(
        self,
        gas_constant: float = _GAS_CONST_J_PER_MOL_K,
    ) -> None:
        """Initialize the orientational entropy calculator.

        Args:
            gas_constant: Gas constant in J/(mol*K).
        """
        self._gas_constant = float(gas_constant)

    def calculate_orientational(
        self,
        neighbour_count: float,
        symmetry_number: int,
        linear: bool,
    ) -> OrientationalEntropyResult:
        """Calculate orientational entropy from neighbour counts.

        The number of orientations is estimated as:
            Ω = sqrt(N_av^3 * π)/symmetry_number for non-linear molecules
            Ω = N_av / symmetry_number for linear molecules

        and the entropy contribution is:

            S = R * ln(Ω)

        where N_av is the average number of neighbours and R is the gas constant.

        Args:
            neighbours: average number of neighbours
            symmetry_number: symmetry number of molecule of interest
            linear: True if molecule of interest is linear

        Returns:
            OrientationalEntropyResult containing the total entropy in J/mol/K.

        Raises:
            ValueError if number of neighbours is negative.
        """
        if neighbour_count < 0:
            raise ValueError(f"neighbour_count must be >= 0, got {neighbour_count}")

        omega = self._omega(neighbour_count, symmetry_number, linear)

        total = self._gas_constant * math.log(omega)
        logger.debug("Orientational entropy total: %s", total)

        return OrientationalEntropyResult(total=float(total))

    @staticmethod
    def _omega(neighbour_count: int, symmetry: int, linear: bool) -> float:
        """Compute the number of orientations Ω.

        Args:
            neighbour_count: average number of neighbours.
            symmetry_number: The symmetry number of the molecule.
            linear: Is the molecule linear (True or False).

        Returns:
            Ω (unitless).
        """
        # symmetry number 0 = spherically symmetric = no orientational entropy
        if symmetry == 0:
            omega = 1
        else:
            if linear:
                omega = neighbour_count / symmetry
            else:
                omega = np.sqrt((neighbour_count**3) * math.pi) / symmetry

        # avoid negative orientational entropy
        if omega < 1:
            omega = 1

        return omega
