import logging
from typing import Any, Dict

from CodeEntropy.entropy.configurational_entropy import ConformationalEntropy

logger = logging.getLogger(__name__)


class ConfigurationalEntropyNode:
    """
    Computes conformational (configurational) entropy from conformational states.

    Requires:
      shared_data["conformational_states"] = {"ua": ..., "res": ...}
      shared_data["n_frames"]
      shared_data["levels"], shared_data["groups"]
    """

    def run(self, shared_data: Dict[str, Any], **_kwargs):
        run_manager = shared_data["run_manager"]
        args = shared_data["args"]
        universe = shared_data["reduced_universe"]
        data_logger = shared_data.get("data_logger")

        group_molecules = shared_data.get("group_molecules")

        levels = shared_data["levels"]
        groups = shared_data["groups"]
        number_frames = shared_data["n_frames"]

        if "conformational_states" in shared_data:
            states_ua = shared_data["conformational_states"]["ua"]
            states_res = shared_data["conformational_states"]["res"]
        else:
            states_ua = shared_data.get("states_united_atom", {})
            states_res = shared_data.get("states_residue", [])

        ce = ConformationalEntropy(
            run_manager=run_manager,
            args=args,
            universe=universe,
            data_logger=data_logger,
            group_molecules=group_molecules,
        )

        conf_results = {}

        for group_id, mol_indices in groups.items():
            group_total = 0.0

            mol_index = mol_indices[0]
            for level in levels[mol_index]:
                if level == "united_atom":
                    group_total += self._ua_entropy(
                        ce, group_id, states_ua, number_frames
                    )

                elif level == "residue":
                    group_total += self._residue_entropy(
                        ce, group_id, states_res, number_frames
                    )

            conf_results[group_id] = group_total

        logger.info("[ConfigurationalEntropyNode] Done")
        return {"configurational_entropy": conf_results}

    def _has_states(self, values):
        return values is not None and len(values) > 0

    def _ua_entropy(self, ce, group_id, states_ua, number_frames):
        total = 0.0
        for key, values in states_ua.items():
            if key[0] != group_id:
                continue
            if self._has_states(values):
                total += ce.conformational_entropy_calculation(values, number_frames)
        return total

    def _residue_entropy(self, ce, group_id, states_res, number_frames):
        if group_id >= len(states_res):
            return 0.0
        values = states_res[group_id]
        if not self._has_states(values):
            return 0.0
        return ce.conformational_entropy_calculation(values, number_frames)
