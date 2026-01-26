# CodeEntropy/levels/nodes/build_covariance_matrices.py

from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class BuildCovarianceMatricesNode:
    def __init__(self, universe_operations):
        self._ft = ForceTorqueManager(universe_operations)

    def run(self, shared_data):
        force_avg, torque_avg, frame_counts = self._ft.build_covariance_matrices(
            entropy_manager=shared_data.get("entropy_manager"),
            reduced_atom=shared_data["reduced_universe"],
            levels=shared_data["levels"],
            groups=shared_data["groups"],
            start=shared_data["start"],
            end=shared_data["end"],
            step=shared_data["step"],
            number_frames=shared_data["n_frames"],
            force_partitioning=shared_data["args"].force_partitioning,
        )

        shared_data["force_covariances"] = force_avg
        shared_data["torque_covariances"] = torque_avg
        shared_data["frame_counts"] = frame_counts
