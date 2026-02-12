import logging
from typing import Any, Dict

from CodeEntropy.entropy.configurational_entropy import ConformationalEntropy

logger = logging.getLogger(__name__)


class ConfigurationalEntropyNode:
    """
    Computes conformational entropy from conformational states produced by LevelDAG.

    Expected:
      shared_data["conformational_states"]["ua"]
        -> dict[(group_id, res_id)] = list/int states
      shared_data["conformational_states"]["res"]
        -> list indexed by group index OR dict[group_id]=states
    """

    def run(self, shared_data: Dict[str, Any], **_kwargs) -> Dict[str, Any]:
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
        gid2i = shared_data.get(
            "group_id_to_index", {gid: i for i, gid in enumerate(groups.keys())}
        )

        states_ua = conf_states.get("ua", {}) or {}
        states_res = conf_states.get("res", {})

        results: Dict[int, Dict[str, float]] = {}

        fragments = universe.atoms.fragments

        for group_id, mol_ids in groups.items():
            mol_id = mol_ids[0]
            mol = fragments[mol_id]
            level_list = levels[mol_id]

            results[group_id] = {"ua": 0.0, "res": 0.0, "poly": 0.0}

            if "united_atom" in level_list:
                total = 0.0
                for res_id, res in enumerate(mol.residues):
                    key = (group_id, res_id)
                    states = states_ua.get(key, [])
                    if not states:
                        val = 0.0
                    else:
                        val = ce.conformational_entropy_calculation(states, n_frames)

                    total += val

                    if data_logger is not None:
                        data_logger.add_residue_data(
                            group_id=group_id,
                            resname=res.resname,
                            level="united_atom",
                            entropy_type="Conformational",
                            frame_count=n_frames,
                            value=val,
                        )

                results[group_id]["ua"] = total
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "united_atom", "Conformational", total
                    )

            if "residue" in level_list:
                val = 0.0

                if isinstance(states_res, dict):
                    s = states_res.get(group_id, [])
                    if s:
                        val = ce.conformational_entropy_calculation(s, n_frames)
                else:
                    gi = gid2i[group_id]
                    if gi < len(states_res) and states_res[gi]:
                        val = ce.conformational_entropy_calculation(
                            states_res[gi], n_frames
                        )

                results[group_id]["res"] = val
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "residue", "Conformational", val
                    )

        logger.info("[ConfigurationalEntropyNode] Done")
        return {"configurational_entropy": results}
