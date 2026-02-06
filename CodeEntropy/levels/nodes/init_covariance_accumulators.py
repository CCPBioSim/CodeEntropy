# CodeEntropy/levels/nodes/init_covariance_accumulators.py

import numpy as np


class InitCovarianceAccumulatorsNode:
    """
    Allocate containers for running averages, matching procedural semantics.

    force_covariances  = {"ua": {}, "res": [None]*n_groups, "poly": [None]*n_groups}
    torque_covariances = {"ua": {}, "res": [None]*n_groups, "poly": [None]*n_groups}

    frame_counts    = {"ua": {}, "res": np.zeros(n_groups), "poly": np.zeros(n_groups)}

    Also stores:
      shared_data["group_id_to_index"]  : dict[group_id -> 0..n_groups-1]
      shared_data["index_to_group_id"]  : list[index -> group_id]
    """

    def run(self, shared_data):
        groups = shared_data["groups"]
        group_ids = list(groups.keys())
        n_groups = len(group_ids)

        gid2i = {gid: i for i, gid in enumerate(group_ids)}
        i2gid = list(group_ids)

        force_avg = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}
        torque_avg = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}

        frame_counts = {
            "ua": {},
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        shared_data["group_id_to_index"] = gid2i
        shared_data["index_to_group_id"] = i2gid
        shared_data["force_covariances"] = force_avg
        shared_data["torque_covariances"] = torque_avg
        shared_data["frame_counts"] = frame_counts

        return {
            "group_id_to_index": gid2i,
            "index_to_group_id": i2gid,
            "force_covariances": force_avg,
            "torque_covariances": torque_avg,
            "frame_counts": frame_counts,
        }
