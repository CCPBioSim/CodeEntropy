import logging

import numpy as np

from CodeEntropy.entropy.vibrational_entropy import VibrationalEntropy

logger = logging.getLogger(__name__)


class VibrationalEntropyNode:
    """
    DAG node responsible for computing vibrational entropy
    from precomputed force and torque covariance matrices.
    """

    def __init__(self, run_manager, data_logger):
        self._ve = VibrationalEntropy(run_manager)
        self._data_logger = data_logger

    def run(self, shared_data, **_):
        levels = shared_data["levels"]
        groups = shared_data["groups"]
        args = shared_data["args"]

        force_cov = shared_data["force_covariance"]
        torque_cov = shared_data["torque_covariance"]
        frame_counts = shared_data["frame_counts"]

        vibrational_results = {}

        for group_id, mol_ids in groups.items():
            mol_index = mol_ids[0]
            vibrational_results[group_id] = {}

            for level in levels[mol_index]:
                highest = level == levels[mol_index][-1]

                if level == "united_atom":
                    S_trans, S_rot = self._ua_entropy(
                        group_id,
                        force_cov["ua"],
                        torque_cov["ua"],
                        frame_counts["ua"],
                        args.temperature,
                        highest,
                    )

                else:
                    S_trans, S_rot = self._level_entropy(
                        group_id,
                        level,
                        force_cov[level][group_id],
                        torque_cov[level][group_id],
                        args.temperature,
                        highest,
                    )

                vibrational_results[group_id][level] = {
                    "trans": S_trans,
                    "rot": S_rot,
                }

                self._data_logger.add_results_data(
                    group_id, level, "Transvibrational", S_trans
                )
                self._data_logger.add_results_data(
                    group_id, level, "Rovibrational", S_rot
                )

        shared_data["vibrational_entropy"] = vibrational_results
        return {"vibrational_entropy": vibrational_results}

    def _ua_entropy(
        self,
        group_id,
        force_matrices,
        torque_matrices,
        frame_counts,
        temperature,
        highest,
    ):
        S_trans = 0.0
        S_rot = 0.0

        for key, fmat in force_matrices.items():
            fmat = self._filter_matrix(fmat)
            tmat = self._filter_matrix(torque_matrices[key])

            S_trans += self._ve.vibrational_entropy_calculation(
                fmat, "force", temperature, highest
            )
            S_rot += self._ve.vibrational_entropy_calculation(
                tmat, "torque", temperature, highest
            )

        return S_trans, S_rot

    def _level_entropy(
        self,
        group_id,
        level,
        force_matrix,
        torque_matrix,
        temperature,
        highest,
    ):
        fmat = self._filter_matrix(force_matrix)
        tmat = self._filter_matrix(torque_matrix)

        S_trans = self._ve.vibrational_entropy_calculation(
            fmat, "force", temperature, highest
        )
        S_rot = self._ve.vibrational_entropy_calculation(
            tmat, "torque", temperature, highest
        )

        return S_trans, S_rot

    @staticmethod
    def _filter_matrix(matrix):
        mask = ~(np.all(matrix == 0, axis=0))
        return matrix[np.ix_(mask, mask)]
