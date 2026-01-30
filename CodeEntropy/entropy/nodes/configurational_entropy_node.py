# CodeEntropy/entropy/nodes/configurational_entropy_node.py

import logging
from typing import Any, Dict

from CodeEntropy.entropy.configurational_entropy import ConformationalEntropy

logger = logging.getLogger(__name__)


class ConfigurationalEntropyNode:
    """
    Computes conformational entropy from conformational states produced by LevelDAG.

    Expected shapes:
      shared_data["conformational_states"]["ua"]
      -> dict[(group_id, res_id)] = list[str]
      shared_data["conformational_states"]["res"]
      -> list indexed by group_id = list[str] or []
    """

    def run(self, shared_data: Dict[str, Any], **_kwargs):
        run_manager = shared_data["run_manager"]
        args = shared_data["args"]
        universe = shared_data["reduced_universe"]
        data_logger = shared_data.get("data_logger")

        ce = ConformationalEntropy(
            run_manager=run_manager,
            args=args,
            universe=universe,
            data_logger=data_logger,
            group_molecules=shared_data.get("group_molecules"),
        )

        conf_states = shared_data["conformational_states"]

        n_frames = shared_data.get("n_frames", shared_data.get("number_frames"))
        if n_frames is None:
            raise KeyError("shared_data must contain n_frames (or number_frames)")

        groups = shared_data["groups"]
        levels = shared_data["levels"]

        results: Dict[int, Dict[str, float]] = {}

        states_ua = conf_states.get("ua", {})
        states_res = conf_states.get("res", [])

        for group_id, mol_ids in groups.items():
            mol_id = mol_ids[0]
            level_list = levels[mol_id]

            results[group_id] = {"ua": 0.0, "res": 0.0, "poly": 0.0}

            # -------- united atom (sum over residues) --------
            if "united_atom" in level_list:
                total = 0.0
                for (gid, _res_id), states in states_ua.items():
                    if gid != group_id:
                        continue
                    if not states:
                        continue
                    total += ce.conformational_entropy_calculation(states, n_frames)

                results[group_id]["ua"] = total
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "united_atom", "Conformational", total
                    )

            # -------- residue (one per group) --------
            if "residue" in level_list:
                if group_id < len(states_res) and states_res[group_id]:
                    val = ce.conformational_entropy_calculation(
                        states_res[group_id], n_frames
                    )
                else:
                    val = 0.0

                results[group_id]["res"] = val
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "residue", "Conformational", val
                    )

        logger.info("[ConfigurationalEntropyNode] Done")
        return {"configurational_entropy": results}
