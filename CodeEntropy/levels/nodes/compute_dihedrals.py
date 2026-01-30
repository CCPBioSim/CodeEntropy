from CodeEntropy.levels.dihedral_tools import DihedralAnalysis


class ComputeConformationalStatesNode:
    """
    Static node (runs once). Internally scans the trajectory to build conformational
    states.
    Produces:
      shared_data["conformational_states"] = {"ua": states_ua, "res": states_res}
    """

    def __init__(self, universe_operations):
        self._dih = DihedralAnalysis(universe_operations=universe_operations)

    def run(self, shared_data):
        u = shared_data["reduced_universe"]
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

        shared_data["conformational_states"] = {"ua": states_ua, "res": states_res}
        return {"conformational_states": shared_data["conformational_states"]}
