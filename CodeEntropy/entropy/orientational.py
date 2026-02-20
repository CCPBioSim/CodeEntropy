"""Orientational entropy calculations.

This module defines `OrientationalEntropy`, which computes orientational entropy
from a neighbor-count mapping.

The current implementation supports non-water neighbors. Water-specific behavior
can be implemented later behind an interface so the core calculation remains
stable and testable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

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
        run_manager: Any,
        args: Any,
        universe: Any,
        reporter: Any,
        group_molecules: Any,
        gas_constant: float = _GAS_CONST_J_PER_MOL_K,
    ) -> None:
        """Initialize the orientational entropy calculator.

        Args:
            run_manager: Run manager (currently unused by this class).
            args: User arguments (currently unused by this class).
            universe: MDAnalysis Universe (currently unused by this class).
            reporter: Data logger (currently unused by this class).
            group_molecules: Grouping helper (currently unused by this class).
            gas_constant: Gas constant in J/(mol*K).
        """
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._reporter = reporter
        self._group_molecules = group_molecules
        self._gas_constant = float(gas_constant)

    def calculate(self, neighbours: Mapping[str, int]) -> OrientationalEntropyResult:
        """Calculate orientational entropy from neighbor counts.

        For each neighbor species (except water), the number of orientations is
        estimated as:

            Ω = sqrt(Nc^3 * π)

        and the entropy contribution is:

            S = R * ln(Ω)

        where Nc is the neighbor count and R is the gas constant.

        Args:
            neighbours: Mapping of neighbor species name to count.

        Returns:
            OrientationalEntropyResult containing the total entropy in J/mol/K.
        """
        total = 0.0
        for species, count in neighbours.items():
            if self._is_water(species):
                # Water handling can be added later (e.g., via a strategy).
                logger.debug(
                    "Skipping water species %s in orientational entropy.", species
                )
                continue

            contribution = self._entropy_contribution(count)
            logger.debug(
                "Orientational entropy contribution for %s: %s", species, contribution
            )
            total += contribution

        logger.debug("Final orientational entropy total: %s", total)
        return OrientationalEntropyResult(total=float(total))

    @staticmethod
    def _is_water(species: str) -> bool:
        """Return True if the species should be treated as water.

        Args:
            species: Species identifier.

        Returns:
            True if the species is considered water.
        """
        return species in {"H2O", "WAT", "HOH"}

    def _entropy_contribution(self, neighbour_count: int) -> float:
        """Compute the entropy contribution for a single neighbor count.

        Args:
            neighbour_count: Number of neighbors (Nc).

        Returns:
            Entropy contribution in J/mol/K.

        Raises:
            ValueError: If neighbour_count is negative.
        """
        if neighbour_count < 0:
            raise ValueError(f"neighbour_count must be >= 0, got {neighbour_count}")

        if neighbour_count == 0:
            return 0.0

        omega = self._omega(neighbour_count)
        # omega should always be > 0 when neighbour_count > 0, but guard anyway.
        if omega <= 0.0:
            return 0.0

        return self._gas_constant * math.log(omega)

    @staticmethod
    def _omega(neighbour_count: int) -> float:
        """Compute the number of orientations Ω.

        Args:
            neighbour_count: Number of neighbors (Nc).

        Returns:
            Ω (unitless).
        """
        return float(np.sqrt((neighbour_count**3) * math.pi))
