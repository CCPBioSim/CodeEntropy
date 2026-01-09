from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class BuildCovarianceMatricesNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        """
        Build force and torque covariance matrices from weighted forces/torques.
        """

        weighted_forces = shared_data["weighted_forces"]
        weighted_torques = shared_data["weighted_torques"]

        force_cov, torque_cov, frame_counts = self._ft.build_covariance_matrices(
            weighted_forces,
            weighted_torques,
        )

        shared_data["force_covariance"] = force_cov
        shared_data["torque_covariance"] = torque_cov
        shared_data["frame_counts"] = frame_counts
