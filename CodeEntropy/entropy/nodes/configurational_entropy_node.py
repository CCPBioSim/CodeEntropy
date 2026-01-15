import logging

import numpy as np

from CodeEntropy.entropy.configurational_entropy import ConformationalEntropy

logger = logging.getLogger(__name__)


class ConfigurationalEntropyNode:
    """
    DAG node responsible for computing configurational entropy
    from precomputed dihedral conformational states.
    """

    def __init__(self, data_logger):
        self._ce = ConformationalEntropy()
        self._data_logger = data_logger

    def run(self, shared_data, **_):
        levels = shared_data["levels"]
        groups = shared_data["groups"]

        states_ua = shared_data["conformational_states"]["ua"]
        states_res = shared_data["conformational_states"]["res"]

        conf_results = {}

        for group_id, mol_ids in groups.items():
            mol_index = mol_ids[0]
            conf_results[group_id] = {}

            for level in levels[mol_index]:

                if level == "united_atom":
                    S_conf = self._ua_entropy(group_id, states_ua)

                elif level == "residue":
                    S_conf = self._residue_entropy(group_id, states_res)

                else:
                    continue  # polymer has no conformational entropy

                conf_results[group_id][level] = S_conf

                self._data_logger.add_results_data(
                    group_id, level, "Conformational", S_conf
                )

        shared_data["configurational_entropy"] = conf_results
        return {"configurational_entropy": conf_results}

    def _ua_entropy(self, group_id, states):
        S_total = 0.0

        for key, values in states.items():
            if key[0] != group_id:
                continue

            if self._has_states(values):
                S_total += self._ce.conformational_entropy_calculation(values)

        return S_total

    def _residue_entropy(self, group_id, states):
        if group_id >= len(states):
            return 0.0

        values = states[group_id]

        if self._has_states(values):
            return self._ce.conformational_entropy_calculation(values)

        return 0.0

    @staticmethod
    def _has_states(values):
        if values is None:
            return False
        if isinstance(values, np.ndarray):
            return np.any(values)
        return any(values)
