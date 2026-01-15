from CodeEntropy.levels.dihedral_tools import DihedralAnalysis


class ComputeConformationalStatesNode:
    def __init__(self, universe_operations):
        self._dih = DihedralAnalysis(universe_operations)

    def run(self, shared_data):
        u = shared_data["universe"]
        levels = shared_data["levels"]
        groups = shared_data["groups"]

        start = shared_data["start"]
        end = shared_data["end"]
        step = shared_data["step"]
        bin_width = shared_data["args"].bin_width

        states_ua, states_res = self._dih.build_conformational_states(
            data_container=u,
            levels=levels,
            groups=groups,
            start=start,
            end=end,
            step=step,
            bin_width=bin_width,
        )

        shared_data["states_united_atom"] = states_ua
        shared_data["states_residue"] = states_res

        return {
            "states_united_atom": states_ua,
            "states_residue": states_res,
        }
