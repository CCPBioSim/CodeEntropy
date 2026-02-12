import logging

import numpy as np

logger = logging.getLogger(__name__)


def _empty_stats():
    return {"n": 0, "mean": None, "M2": None}


class InitCovarianceAccumulatorsNode:
    """
    Allocate Welford online covariance accumulators (procedural semantics).

    After LevelDAG finishes iterating frames, it will "finalize" stats into:
      shared_data["force_covariances"]
      shared_data["torque_covariances"]

    Plus frame counts, and group_id_to_index mapping.
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

        frame_counts = {
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

        logger.warning(f"[InitCovAcc] group_ids={group_ids} gid2i={gid2i}")

        return {
            "group_id_to_index": gid2i,
            "index_to_group_id": i2gid,
            "force_covariances": force_cov,
            "torque_covariances": torque_cov,
            "frame_counts": frame_counts,
            "force_stats": force_stats,
            "torque_stats": torque_stats,
        }
