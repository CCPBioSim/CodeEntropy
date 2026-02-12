import logging

import numpy as np
from numpy import linalg as la

logger = logging.getLogger(__name__)


class VibrationalEntropy:
    """
    Performs vibrational entropy calculations using molecular trajectory data.
    """

    def __init__(self, run_manager, args, universe, data_logger, group_molecules):
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._data_logger = data_logger
        self._group_molecules = group_molecules

        self._PLANCK_CONST = 6.62607004081818e-34
        self._GAS_CONST = 8.3144598484848

    def frequency_calculation(self, lambdas, temp):
        pi = np.pi
        kT = self._run_manager.get_KT2J(temp)

        lambdas = np.array(lambdas)
        lambdas = np.real_if_close(lambdas, tol=1000)

        valid_mask = (
            np.isreal(lambdas) & (lambdas > 0) & (~np.isclose(lambdas, 0, atol=1e-7))
        )
        if len(lambdas) > np.count_nonzero(valid_mask):
            logger.warning(
                f"{len(lambdas) - np.count_nonzero(valid_mask)} "
                f"invalid eigenvalues excluded (complex, non-positive, or near-zero)."
            )

        lambdas = lambdas[valid_mask].real
        frequencies = 1 / (2 * pi) * np.sqrt(lambdas / kT)
        return frequencies

    def vibrational_entropy_calculation(self, matrix, matrix_type, temp, highest_level):
        """
        Supports matrix_type:
          - "force"  (3N x 3N)
          - "torque" (3N x 3N)
          - "forcetorqueTRANS" (6N x 6N -> translational part)
          - "forcetorqueROT"   (6N x 6N -> rotational part)

        Procedural matching behavior for FTmat:
          - compute entropy components from the full 6N spectrum
          - split into first 3N and last 3N *after sorting frequencies*
          - so: FTmat-Trans + FTmat-Rot == total FT entropy
        """
        matrix = np.asarray(matrix)
        lambdas = la.eigvals(matrix)
        lambdas = self._run_manager.change_lambda_units(lambdas)

        freqs = self.frequency_calculation(lambdas, temp)
        freqs = np.sort(freqs)

        kT = self._run_manager.get_KT2J(temp)
        exponent = self._PLANCK_CONST * freqs / kT
        power_positive = np.exp(exponent)
        power_negative = np.exp(-exponent)

        S_components = exponent / (power_positive - 1.0) - np.log(1.0 - power_negative)
        S_components *= self._GAS_CONST

        n_modes = len(S_components)

        if matrix_type == "force":
            if highest_level:
                return float(np.sum(S_components))
            return float(np.sum(S_components[6:]))

        if matrix_type == "torque":
            return float(np.sum(S_components))

        if matrix_type in ("forcetorqueTRANS", "forcetorqueROT"):
            if n_modes % 2 != 0:
                logger.warning(
                    f"FTmat has odd number of modes ({n_modes}); cannot cleanly split."
                )
                return float(np.sum(S_components))

            half = n_modes // 2  # == 3N
            trans_part = float(np.sum(S_components[:half]))
            rot_part = float(np.sum(S_components[half:]))

            if not highest_level:
                trans_keep = max(0, half - 6)
                trans_part = (
                    float(np.sum(S_components[6:half])) if trans_keep > 0 else 0.0
                )

            return trans_part if matrix_type == "forcetorqueTRANS" else rot_part

        raise ValueError(f"Unknown matrix_type: {matrix_type}")
