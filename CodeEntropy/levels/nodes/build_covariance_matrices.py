from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class BuildCovarianceMatricesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data, compute_weighted_forces, compute_weighted_torques):
        return self._ft.build_covariance_matrices(
            compute_weighted_forces["forces"], compute_weighted_torques["torques"]
        )
