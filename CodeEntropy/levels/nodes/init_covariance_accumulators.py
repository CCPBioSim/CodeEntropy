import logging

import numpy as np

logger = logging.getLogger(__name__)


def _empty_stats():
    return {"n": 0, "mean": None, "M2": None}


class InitCovarianceAccumulatorsNode:
    """
    Allocate accumulators for per-frame reductions.

    Canonical mean accumulators:
      shared_data["force_covariances"]
      shared_data["torque_covariances"]

    Canonical combined (full 6N x 6N) mean accumulator:
      shared_data["forcetorque_covariances"]
      shared_data["forcetorque_counts"]

    Backwards-compatible aliases (point to the same objects):
      shared_data["force_torque_stats"]   -> shared_data["forcetorque_covariances"]
      shared_data["force_torque_counts"]  -> shared_data["forcetorque_counts"]
    """

    def run(self, shared_data):
        groups = shared_data["groups"]
        group_ids = list(groups.keys())
        n_groups = len(group_ids)

        gid2i = {gid: i for i, gid in enumerate(group_ids)}
        i2gid = list(group_ids)

        force_cov = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}
        torque_cov = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}

        force_stats = {
            "ua": {},
            "res": [_empty_stats() for _ in range(n_groups)],
            "poly": [_empty_stats() for _ in range(n_groups)],
        }
        torque_stats = {
            "ua": {},
            "res": [_empty_stats() for _ in range(n_groups)],
            "poly": [_empty_stats() for _ in range(n_groups)],
        }
        force_torque_stats = {
            "ua": {},
            "res": [None] * n_groups,
            "poly": [None] * n_groups,
        }

        frame_counts = {
            "ua": {},
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        force_torque_counts = {
            "ua": {},
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        shared_data["group_id_to_index"] = gid2i
        shared_data["index_to_group_id"] = i2gid

        shared_data["force_covariances"] = force_cov
        shared_data["torque_covariances"] = torque_cov
        shared_data["frame_counts"] = frame_counts

        shared_data["force_stats"] = force_stats
        shared_data["torque_stats"] = torque_stats

        shared_data["force_torque_stats"] = force_torque_stats
        shared_data["force_torque_counts"] = force_torque_counts

        shared_data["forcetorque_covariances"] = force_torque_stats
        shared_data["forcetorque_counts"] = force_torque_counts

        logger.warning(f"[InitCovAcc] group_ids={group_ids} gid2i={gid2i}")

        return {
            "group_id_to_index": gid2i,
            "index_to_group_id": i2gid,
            "force_covariances": force_cov,
            "torque_covariances": torque_cov,
            "frame_counts": frame_counts,
            "force_stats": force_stats,
            "torque_stats": torque_stats,
            "force_torque_stats": force_torque_stats,
            "force_torque_counts": force_torque_counts,
            "forcetorque_covariances": force_torque_stats,
            "forcetorque_counts": force_torque_counts,
        }
