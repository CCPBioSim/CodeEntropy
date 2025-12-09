from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class ComputeWeightedTorquesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data, compute_axes, build_beads):
        u = shared_data["universe"]
        torques = {}

        for key, bead_list in build_beads["beads"].items():
            torques[key] = []
            for bead, ax in zip(bead_list, compute_axes["axes"][key]):
                torques[key].append(self._ft.get_weighted_torques(u, bead, ax))
        return {"torques": torques}
