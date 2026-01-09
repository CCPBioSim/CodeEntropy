from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class ComputeWeightedForcesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        u = shared_data["universe"]
        beads = shared_data["beads"]
        axes = shared_data["axes"]

        forces = {}

        for key, bead_list in beads.items():
            forces[key] = []
            for bead, ax in zip(bead_list, axes[key]):
                forces[key].append(self._ft.get_weighted_forces(u, bead, ax, False))

        shared_data["weighted_forces"] = forces
