import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class InitCovarianceAccumulatorsNode:
    """
    Allocate accumulators for per-frame reductions.

    Canonical mean accumulators:
      shared_data["force_covariances"]
      shared_data["torque_covariances"]
      shared_data["forcetorque_covariances"]   # 6N x 6N mean (highest level only)

    Counters:
      shared_data["frame_counts"]
      shared_data["forcetorque_counts"]

    Backwards-compatible aliases:
      shared_data["force_torque_stats"]   -> shared_data["forcetorque_covariances"]
      shared_data["force_torque_counts"]  -> shared_data["forcetorque_counts"]
    """

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        groups = shared_data["groups"]
        group_ids = list(groups.keys())
        n_groups = len(group_ids)

        gid2i = {gid: i for i, gid in enumerate(group_ids)}
        i2gid = list(group_ids)

        force_cov = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}
        torque_cov = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}

        frame_counts = {
            "ua": {},
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        forcetorque_cov = {
            "res": [None] * n_groups,
            "poly": [None] * n_groups,
        }
        forcetorque_counts = {
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        shared_data["group_id_to_index"] = gid2i
        shared_data["index_to_group_id"] = i2gid

        shared_data["force_covariances"] = force_cov
        shared_data["torque_covariances"] = torque_cov
        shared_data["frame_counts"] = frame_counts

        shared_data["forcetorque_covariances"] = forcetorque_cov
        shared_data["forcetorque_counts"] = forcetorque_counts

        shared_data["force_torque_stats"] = forcetorque_cov
        shared_data["force_torque_counts"] = forcetorque_counts

        logger.info(f"[InitCovAcc] group_ids={group_ids} gid2i={gid2i}")

        return {
            "group_id_to_index": gid2i,
            "index_to_group_id": i2gid,
            "force_covariances": force_cov,
            "torque_covariances": torque_cov,
            "frame_counts": frame_counts,
            "forcetorque_covariances": forcetorque_cov,
            "forcetorque_counts": forcetorque_counts,
            "force_torque_stats": forcetorque_cov,
            "force_torque_counts": forcetorque_counts,
        }
