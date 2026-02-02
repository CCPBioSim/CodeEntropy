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
        -> dict[(group_id, res_id)] = list[int] or list[str] (length ~ n_frames)
      shared_data["conformational_states"]["res"]
        -> list indexed by group_id = list[int] or list[str] (length ~ n_frames) OR []
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

        if "conformational_states" not in shared_data:
            raise KeyError(
                "shared_data['conformational_states'] is missing. "
                "Did LevelDAG run ComputeConformationalStatesNode?"
            )

        conf_states = shared_data["conformational_states"]
        states_ua = conf_states.get("ua", {})  # dict[(group_id, res_id)] -> states
        states_res = conf_states.get("res", [])  # list[group_id] -> states

        n_frames = shared_data.get("n_frames", shared_data.get("number_frames"))
        if n_frames is None:
            raise KeyError("shared_data must contain n_frames (or number_frames)")

        groups = shared_data["groups"]
        levels = shared_data["levels"]

        fragments = universe.atoms.fragments

        results: Dict[int, Dict[str, float]] = {}

        for group_id, mol_ids in groups.items():
            mol_id = mol_ids[0]
            level_list = levels[mol_id]
            mol = fragments[mol_id]

            results[group_id] = {"ua": 0.0, "res": 0.0, "poly": 0.0}

            if "united_atom" in level_list:
                total_ua = 0.0

                for (gid, res_id), states in states_ua.items():
                    if gid != group_id or not states:
                        continue

                    s_res = ce.conformational_entropy_calculation(states, n_frames)
                    total_ua += s_res

                    if data_logger is not None:
                        if res_id < len(mol.residues):
                            resname = mol.residues[res_id].resname
                        else:
                            resname = f"RES{res_id}"

                        data_logger.add_residue_data(
                            group_id=group_id,
                            resname=resname,
                            level="united_atom",
                            entropy_type="Conformational",
                            frame_count=n_frames,
                            value=s_res,
                        )

                results[group_id]["ua"] = total_ua

                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "united_atom", "Conformational", total_ua
                    )

            if "residue" in level_list:
                if group_id < len(states_res) and states_res[group_id]:
                    s = states_res[group_id]
                    val = ce.conformational_entropy_calculation(s, n_frames)
                else:
                    val = 0.0

                results[group_id]["res"] = val

                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "residue", "Conformational", val
                    )

        logger.info("[ConfigurationalEntropyNode] Done")
        return {"configurational_entropy": results}
