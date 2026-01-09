from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class ComputeWeightedTorquesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        """
        Compute weighted torques for each bead using precomputed axes.
        """

        u = shared_data["universe"]
        beads_by_key = shared_data["beads"]
        axes_by_key = shared_data["axes"]

        torques = {}

        for key, bead_list in beads_by_key.items():
            torques[key] = []
            for bead, ax in zip(bead_list, axes_by_key[key]):
                torques[key].append(self._ft.get_weighted_torques(u, bead, ax))

        shared_data["weighted_torques"] = torques
