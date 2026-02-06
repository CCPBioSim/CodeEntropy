import logging
from typing import Any, Dict

from CodeEntropy.entropy.configurational_entropy import ConformationalEntropy

logger = logging.getLogger(__name__)


def _build_gid2i(shared_data: Dict[str, Any]) -> Dict[int, int]:
    """
    Prefer LevelDAG-provided mapping. Otherwise create a stable mapping based on
    insertion order of groups.
    """
    gid2i = shared_data.get("group_id_to_index")
    if isinstance(gid2i, dict) and gid2i:
        return gid2i

    groups = shared_data["groups"]
    return {gid: i for i, gid in enumerate(groups.keys())}


class ConfigurationalEntropyNode:
    """
    Computes conformational entropy from conformational states produced by LevelDAG.

    Expected:
      shared_data["conformational_states"]["ua"]
        dict[(group_id, res_id)] -> list/array of state labels (len ~ n_frames)
      shared_data["conformational_states"]["res"]
        list indexed by group_index -> list/array of state labels
    """

    def run(self, shared_data: Dict[str, Any], **_kwargs) -> Dict[str, Any]:
        run_manager = shared_data["run_manager"]
        args = shared_data["args"]
        universe = shared_data["reduced_universe"]
        data_logger = shared_data.get("data_logger")
        group_molecules = shared_data.get("group_molecules")

        ce = ConformationalEntropy(
            run_manager=run_manager,
            args=args,
            universe=universe,
            data_logger=data_logger,
            group_molecules=group_molecules,
        )

        conf_states = shared_data["conformational_states"]
        states_ua = conf_states.get("ua", {}) or {}
        states_res = conf_states.get("res", []) or []

        n_frames = shared_data.get("n_frames", shared_data.get("number_frames"))
        if n_frames is None:
            raise KeyError("shared_data must contain n_frames (or number_frames)")

        groups = shared_data["groups"]
        levels = shared_data["levels"]
        frame_counts = shared_data.get("frame_counts", {})
        ua_counts = frame_counts.get("ua", {}) if isinstance(frame_counts, dict) else {}

        gid2i = _build_gid2i(shared_data)

        fragments = universe.atoms.fragments
        results: Dict[int, Dict[str, float]] = {}

        for group_id, mol_ids in groups.items():
            mol_id = mol_ids[0]
            mol = fragments[mol_id]
            level_list = levels[mol_id]

            results[group_id] = {"ua": 0.0, "res": 0.0}

            if "united_atom" in level_list:
                total_ua = 0.0

                for res_id, res in enumerate(mol.residues):
                    key = (group_id, res_id)
                    st = states_ua.get(key, None)
                    if not st:
                        val = 0.0
                    else:
                        val = ce.conformational_entropy_calculation(st, n_frames)

                    total_ua += val

                    if data_logger is not None:
                        fc = ua_counts.get(key, n_frames)
                        data_logger.add_residue_data(
                            group_id=group_id,
                            resname=res.resname,
                            level="united_atom",
                            entropy_type="Conformational",
                            frame_count=fc,
                            value=val,
                        )

                results[group_id]["ua"] = total_ua
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "united_atom", "Conformational", total_ua
                    )

            if "residue" in level_list:
                gi = gid2i[group_id]
                st = states_res[gi] if gi < len(states_res) else None
                val = ce.conformational_entropy_calculation(st, n_frames) if st else 0.0

                results[group_id]["res"] = val
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "residue", "Conformational", val
                    )

        logger.info("[ConfigurationalEntropyNode] Done")
        return {"configurational_entropy": results}
