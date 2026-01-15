from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class ComputeWeightedTorquesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        u = shared_data["universe"]
        beads = shared_data["beads"]
        rot_axes = shared_data["rot_axes"]

        torques = {}

        for key, bead_list in beads.items():
            torques[key] = [
                self._ft.get_weighted_torques(u, bead, r_ax)
                for bead, r_ax in zip(bead_list, rot_axes[key])
            ]

        shared_data["weighted_torques"] = torques
