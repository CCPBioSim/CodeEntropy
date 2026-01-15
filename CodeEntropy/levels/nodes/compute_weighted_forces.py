from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class ComputeWeightedForcesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        u = shared_data["universe"]
        beads = shared_data["beads"]
        trans_axes = shared_data["trans_axes"]

        forces = {}

        for key, bead_list in beads.items():
            forces[key] = [
                self._ft.get_weighted_forces(u, bead, t_ax, False)
                for bead, t_ax in zip(bead_list, trans_axes[key])
            ]

        shared_data["weighted_forces"] = forces
