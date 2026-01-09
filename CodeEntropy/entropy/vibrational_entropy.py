import logging

import numpy as np
from numpy import linalg as la

logger = logging.getLogger(__name__)


class VibrationalEntropy:
    """
    Performs vibrational entropy calculations using molecular trajectory data.
    """

    def __init__(
        self, run_manager, args, universe, data_logger, level_manager, group_molecules
    ):
        """
        Initializes the VibrationalEntropy manager with all required components and
        defines physical constants used in vibrational entropy calculations.
        """
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
