import logging
from typing import Any, Dict

import numpy as np

from CodeEntropy.entropy.configurational_entropy import ConformationalEntropy

logger = logging.getLogger(__name__)


class ConfigurationalEntropyNode:
    """
    Procedural-parity conformational entropy.
    """

    @staticmethod
    def _has_state_data(states) -> bool:
        if states is None:
            return False
        if isinstance(states, np.ndarray):
            return bool(np.any(states))
        try:
            return any(states)
        except TypeError:
            return bool(states)

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

        conf_states = shared_data.get("conformational_states", {}) or {}
        states_ua = conf_states.get("ua", {}) or {}
        states_res = conf_states.get("res", {})

        n_frames = shared_data.get("n_frames", shared_data.get("number_frames"))
        if n_frames is None:
            raise KeyError("shared_data must contain n_frames (or number_frames)")
        n_frames = int(n_frames)

        groups = shared_data["groups"]
        levels = shared_data["levels"]
        fragments = universe.atoms.fragments

        results: Dict[int, Dict[str, float]] = {}

        for group_id, mol_ids in groups.items():
            results[group_id] = {"ua": 0.0, "res": 0.0, "poly": 0.0}
            if not mol_ids:
                continue

            rep_mol_id = mol_ids[0]
            rep_mol = fragments[rep_mol_id]
            level_list = levels[rep_mol_id]

            if "united_atom" in level_list:
                S_conf_ua = 0.0
                for res_id, res in enumerate(rep_mol.residues):
                    key = (group_id, res_id)
                    states = states_ua.get(key, [])

                    if self._has_state_data(states):
                        val = float(
                            ce.conformational_entropy_calculation(states, n_frames)
                        )
                    else:
                        val = 0.0

                    S_conf_ua += val

                    if data_logger is not None:
                        data_logger.add_residue_data(
                            group_id=group_id,
                            resname=getattr(res, "resname", "UNK"),
                            level="united_atom",
                            entropy_type="Conformational",
                            frame_count=n_frames,
                            value=val,
                        )

                results[group_id]["ua"] = S_conf_ua
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "united_atom", "Conformational", S_conf_ua
                    )

            if "residue" in level_list:
                if isinstance(states_res, dict):
                    group_states = states_res.get(group_id, None)
                else:
                    group_states = (
                        states_res[group_id] if group_id < len(states_res) else None
                    )

                if self._has_state_data(group_states):
                    S_conf_res = float(
                        ce.conformational_entropy_calculation(group_states, n_frames)
                    )
                else:
                    S_conf_res = 0.0

                results[group_id]["res"] = S_conf_res
                if data_logger is not None:
                    data_logger.add_results_data(
                        group_id, "residue", "Conformational", S_conf_res
                    )

        logger.info("[ConfigurationalEntropyNode] Done")
        return {"configurational_entropy": results}
