from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class ComputeWeightedTorquesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        u = shared_data["universe"]
        beads = shared_data["beads"]
        axes = shared_data["axes"]

        torques = {}

        for key, bead_list in beads.items():
            torques[key] = []
            for bead, ax in zip(bead_list, axes[key]):
                torques[key].append(self._ft.get_weighted_torques(u, bead, ax))

        shared_data["weighted_torques"] = torques
