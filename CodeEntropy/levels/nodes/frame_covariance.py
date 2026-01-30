# CodeEntropy/levels/nodes/frame_covariance.py

import logging

from CodeEntropy.levels.force_torque_manager import ForceTorqueManager

logger = logging.getLogger(__name__)


class FrameCovarianceNode:
    """
    Per-frame covariance computation.
    Assumes reduced_universe trajectory has already been advanced by the caller.
    Uses ForceTorqueManager for the physics and updates running means here.
    """

    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        u = shared_data["reduced_universe"]

        groups = shared_data["groups"]
        levels = shared_data["levels"]
        beads = shared_data["beads"]

        force_avg = shared_data["force_covariances"]
        torque_avg = shared_data["torque_covariances"]
        counts = shared_data["frame_counts"]

        fp = shared_data["args"].force_partitioning
        fragments = u.atoms.fragments

        for group_id, mol_ids in groups.items():
            for mol_id in mol_ids:
                mol = fragments[mol_id]
                trans_axes = mol.principal_axes()
                level_list = levels[mol_id]

                for level in level_list:
                    highest = level == level_list[-1]

                    if level == "united_atom":
                        for res_id in range(len(mol.residues)):
                            bead_key = (mol_id, "united_atom", res_id)
                            bead_idx_arrays = beads.get(bead_key, [])
                            if not bead_idx_arrays:
                                continue

                            bead_ags = [u.atoms[idx] for idx in bead_idx_arrays]
                            if any(len(ag) == 0 for ag in bead_ags):
                                continue

                            F, T = self._ft.compute_frame_covariance(
                                bead_ags, trans_axes, highest, fp
                            )

                            acc_key = (group_id, res_id)
                            n = counts["ua"].get(acc_key, 0) + 1
                            counts["ua"][acc_key] = n

                            if acc_key not in force_avg["ua"]:
                                force_avg["ua"][acc_key] = F
                                torque_avg["ua"][acc_key] = T
                            else:
                                force_avg["ua"][acc_key] += (
                                    F - force_avg["ua"][acc_key]
                                ) / n
                                torque_avg["ua"][acc_key] += (
                                    T - torque_avg["ua"][acc_key]
                                ) / n

                    elif level in ("residue", "polymer"):
                        bead_key = (mol_id, level)
                        bead_idx_arrays = beads.get(bead_key, [])
                        if not bead_idx_arrays:
                            continue

                        bead_ags = [u.atoms[idx] for idx in bead_idx_arrays]
                        if any(len(ag) == 0 for ag in bead_ags):
                            continue

                        F, T = self._ft.compute_frame_covariance(
                            bead_ags, trans_axes, highest, fp
                        )

                        k = "res" if level == "residue" else "poly"
                        counts[k][group_id] += 1
                        n = counts[k][group_id]

                        if force_avg[k][group_id] is None:
                            force_avg[k][group_id] = F
                            torque_avg[k][group_id] = T
                        else:
                            force_avg[k][group_id] += (F - force_avg[k][group_id]) / n
                            torque_avg[k][group_id] += (T - torque_avg[k][group_id]) / n

        return {
            "force_covariances": force_avg,
            "torque_covariances": torque_avg,
            "frame_counts": counts,
        }
