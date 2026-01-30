# CodeEntropy/levels/nodes/init_covariance_accumulators.py

import numpy as np


class InitCovarianceAccumulatorsNode:
    """
    Allocate containers for running averages, matching the original manager:

      force_covariances = {"ua": {}, "res": [None]*n_groups, "poly": [None]*n_groups}
      torque_covariances = same
      frame_counts = {"ua": {}, "res": np.zeros(n_groups), "poly": np.zeros(n_groups)}
    """

    def run(self, shared_data):
        groups = shared_data["groups"]
        n_groups = len(groups)

        force_avg = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}
        torque_avg = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}

        frame_counts = {
            "ua": {},
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        shared_data["force_covariances"] = force_avg
        shared_data["torque_covariances"] = torque_avg
        shared_data["frame_counts"] = frame_counts

        return {
            "force_covariances": force_avg,
            "torque_covariances": torque_avg,
            "frame_counts": frame_counts,
        }
