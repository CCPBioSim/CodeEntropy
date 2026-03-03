"""Vibrational entropy calculations.

This module provides `VibrationalEntropy`, which computes vibrational entropy
from force, torque, or combined force-torque covariance matrices.

The implementation is intentionally split into small, single-purpose methods:
- Eigenvalue extraction + unit conversion
- Frequency calculation with robust filtering
- Entropy component computation
- Mode selection / summation rules based on matrix type
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import numpy as np
from numpy import linalg as la

logger = logging.getLogger(__name__)

MatrixType = Literal["force", "torque", "forcetorqueTRANS", "forcetorqueROT"]


@dataclass(frozen=True)
class VibrationalEntropyResult:
    """Result of a vibrational entropy computation.

    Attributes:
        total: Computed entropy value (J/mol/K) for the requested matrix type.
        n_modes: Number of vibrational modes used (after filtering eigenvalues).
    """

    total: float
    n_modes: int


class VibrationalEntropy:
    """Compute vibrational entropy from covariance matrices.

    This class focuses only on vibrational entropy math and relies on `run_manager`
    for unit conversions (eigenvalue unit conversion and kT conversion).
    """

    def __init__(
        self,
        run_manager: Any,
        planck_const: float = 6.62607004081818e-34,
        gas_const: float = 8.3144598484848,
    ) -> None:
        """Initialize the vibrational entropy calculator.

        Args:
            run_manager: Provides thermodynamic conversions (e.g., kT in Joules)
                and eigenvalue unit conversion.
            planck_const: Planck constant (J*s).
            gas_const: Gas constant (J/(mol*K)).
        """
        self._run_manager = run_manager
        self._planck_const = float(planck_const)
        self._gas_const = float(gas_const)

    def vibrational_entropy_calculation(
        self,
        matrix: np.ndarray,
        matrix_type: MatrixType,
        temp: float,
        highest_level: bool,
        flexible: int,
    ) -> float:
        """Compute vibrational entropy for the given covariance matrix.

        Supported matrix types:
            - "force": 3N x 3N force covariance.
            - "torque": 3N x 3N torque covariance.
            - "forcetorqueTRANS": 6N x 6N combined covariance (translational part).
            - "forcetorqueROT": 6N x 6N combined covariance (rotational part).

        Mode handling:
            - Frequencies are computed from eigenvalues, filtered to valid values,
              then sorted ascending.
            - For "force":
                - If highest_level, include all modes.
                - Otherwise, drop the lowest 6 modes.
            - For "torque": include all modes.
            - For combined "forcetorque*":
                - Split the sorted spectrum into two halves (first 3N, last 3N).
                - If not highest_level, drop the lowest 6 modes only within the
                  translational half.

        Args:
            matrix: Covariance matrix (shape depends on matrix_type).
            matrix_type: Type of covariance matrix.
            temp: Temperature in Kelvin.
            highest_level: Whether this is the highest level in the hierarchy.

        Returns:
            Vibrational entropy value in J/mol/K.

        Raises:
            ValueError: If matrix_type is unknown.
        """
        components = self._entropy_components(matrix, matrix_type, flexible, temp)
        total = self._sum_components(components, matrix_type, highest_level)
        return float(total)

    def _entropy_components(
        self,
        matrix: np.ndarray,
        matrix_type: str,
        flexible: int,
        temp: float,
    ) -> np.ndarray:
        """Compute per-mode entropy components from a covariance matrix.

        Args:
            matrix: Covariance matrix.
            temp: Temperature in Kelvin.

        Returns:
            Array of entropy components (J/mol/K) for each valid mode.
        """
        lambdas = self._matrix_eigenvalues(matrix)
        logger.debug("lambdas: %s", lambdas)
        lambdas = self._convert_lambda_units(lambdas)
        logger.debug("lambdas converted units: %s", lambdas)
        if matrix_type == "force" and flexible > 0:
            lambdas = self._flexible_dihedral(lambdas, flexible)
            logger.debug("lambdas flexible halved: %s", lambdas)

        freqs = self._frequencies_from_lambdas(lambdas, temp)
        if freqs.size == 0:
            return np.array([], dtype=float)

        freqs = np.sort(freqs)
        return self._entropy_components_from_frequencies(freqs, temp)

    @staticmethod
    def _matrix_eigenvalues(matrix: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of a matrix.

        Args:
            matrix: Input matrix.

        Returns:
            Eigenvalues as a NumPy array.
        """
        matrix = np.asarray(matrix, dtype=float)
        return la.eigvals(matrix)

    def _convert_lambda_units(self, lambdas: np.ndarray) -> np.ndarray:
        """Convert eigenvalues into SI units using run_manager.

        Args:
            lambdas: Eigenvalues.

        Returns:
            Converted eigenvalues.
        """
        return self._run_manager.change_lambda_units(lambdas)

    def _flexible_dihedral(self, lambdas: np.ndarray, flexible: int) -> np.ndarray:
        """Force halving for flexible dihedrals.

        If N flexible dihedrals, halve the forces for the N largest eigenvalues.
        The matrix has force^2 so use factor of 0.25 for eigenvalues.

        Args:
            lambdas: Eigenvalues
            flexible: the number of flexible dihedrals in the molecule

        Returns:
            reduced lambdas
        """
        halved = sorted(lambdas, reverse=True)
        for i in range(flexible):
            halved[i] = 0.25 * halved[i]
        lambdas = halved

        return lambdas

    def _frequencies_from_lambdas(self, lambdas: np.ndarray, temp: float) -> np.ndarray:
        """Convert eigenvalues to frequencies with robust filtering.

        Filters out eigenvalues that are complex, non-positive, or near-zero to
        avoid invalid frequencies and unstable entropies.

        Args:
            lambdas: Eigenvalues (post unit conversion).
            temp: Temperature in Kelvin.

        Returns:
            Frequencies in Hz.
        """
        lambdas = np.asarray(lambdas)
        lambdas = np.real_if_close(lambdas, tol=1000)

        valid_mask = (
            np.isreal(lambdas) & (lambdas > 0) & (~np.isclose(lambdas, 0, atol=1e-7))
        )

        removed = int(len(lambdas) - np.count_nonzero(valid_mask))
        if removed:
            logger.warning(
                "%d invalid eigenvalues excluded (complex, non-positive, "
                "or near-zero).",
                removed,
            )

        lambdas = np.asarray(lambdas[valid_mask].real, dtype=float)
        if lambdas.size == 0:
            return np.array([], dtype=float)

        kT = float(self._run_manager.get_KT2J(temp))
        pi = float(np.pi)
        return (1.0 / (2.0 * pi)) * np.sqrt(lambdas / kT)

    def _entropy_components_from_frequencies(
        self, frequencies: np.ndarray, temp: float
    ) -> np.ndarray:
        """Compute per-mode entropy components from frequencies.

        Args:
            frequencies: Frequencies (Hz), sorted ascending.
            temp: Temperature in Kelvin.

        Returns:
            Per-mode entropy components in J/mol/K.
        """
        kT = float(self._run_manager.get_KT2J(temp))
        exponent = (self._planck_const * frequencies) / kT

        exp_pos = np.exp(exponent)
        exp_neg = np.exp(-exponent)

        components = exponent / (exp_pos - 1.0) - np.log(1.0 - exp_neg)
        return components * self._gas_const

    @staticmethod
    def _split_halves(components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a component array into two equal halves.

        Args:
            components: Array with an even length.

        Returns:
            Tuple of (first_half, second_half). If odd-length, returns
            (components, empty).

        Notes:
            For combined force-torque matrices (6N x 6N), the valid number of modes
            should be 6N. After sorting, we split into two halves of size 3N.
        """
        n = int(components.size)
        if n % 2 != 0:
            return components, np.array([], dtype=float)
        half = n // 2
        return components[:half], components[half:]

    def _sum_components(
        self,
        components: np.ndarray,
        matrix_type: MatrixType,
        highest_level: bool,
    ) -> float:
        if components.size == 0:
            return 0.0

        if matrix_type == "force":
            return float(
                np.sum(components) if highest_level else np.sum(components[6:])
            )

        if matrix_type == "torque":
            return float(np.sum(components))

        if matrix_type in ("forcetorqueTRANS", "forcetorqueROT"):
            if matrix_type == "forcetorqueTRANS":
                return float(np.sum(components[:3]))
            return float(np.sum(components[3:]))

        raise ValueError(f"Unknown matrix_type: {matrix_type}")
