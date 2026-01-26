import logging
from typing import Any, Dict

from CodeEntropy.entropy.vibrational_entropy import VibrationalEntropy

logger = logging.getLogger(__name__)


class VibrationalEntropyNode:
    """
    Computes vibrational entropy from force/torque covariance matrices.
    Expects Level DAG to have filled:
      shared_data["force_covariances"], shared_data["torque_covariances"]
    """

    def run(self, shared_data: Dict[str, Any], **_kwargs):
        run_manager = shared_data["run_manager"]
        args = shared_data["args"]
        universe = shared_data["reduced_universe"]
        data_logger = shared_data.get("data_logger")

        level_manager = shared_data.get("level_manager")
        group_molecules = shared_data.get("group_molecules")

        ve = VibrationalEntropy(
            run_manager=run_manager,
            args=args,
            universe=universe,
            data_logger=data_logger,
            level_manager=level_manager,
            group_molecules=group_molecules,
        )

        temp = args.temperature
        # levels = shared_data["levels"]
        groups = shared_data["groups"]

        force_cov = shared_data["force_covariances"]
        # torque_cov = shared_data["torque_covariances"]

        vib_results = {}

        for group_id in groups.keys():
            vib_results[group_id] = {"ua": 0.0, "res": 0.0, "poly": 0.0}

            # UA is dict keyed by (group_id, res_id)
            for (gid, _res_id), mat in force_cov["ua"].items():
                if gid != group_id:
                    continue
                vib_results[group_id]["ua"] += ve.vibrational_entropy_calculation(
                    mat, "force", temp, highest_level=False
                )

            # residue / polymer are list indexed by group_id
            if force_cov["res"][group_id] is not None:
                vib_results[group_id]["res"] += ve.vibrational_entropy_calculation(
                    force_cov["res"][group_id], "force", temp, highest_level=False
                )

            if force_cov["poly"][group_id] is not None:
                vib_results[group_id]["poly"] += ve.vibrational_entropy_calculation(
                    force_cov["poly"][group_id], "force", temp, highest_level=True
                )

        logger.info("[VibrationalEntropyNode] Done")
        return {"vibrational_entropy": vib_results}
