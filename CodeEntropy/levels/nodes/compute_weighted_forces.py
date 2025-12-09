from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class ComputeWeightedForcesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data, compute_axes, build_beads):
        u = shared_data["universe"]
        forces = {}

        for key, bead_list in build_beads["beads"].items():
            forces[key] = []
            for bead, ax in zip(bead_list, compute_axes["axes"][key]):
                forces[key].append(self._ft.get_weighted_forces(u, bead, ax, False))
        return {"forces": forces}
