# CodeEntropy/levels/nodes/frame_covariance.py

import logging

from CodeEntropy.levels.force_torque_manager import ForceTorqueManager

logger = logging.getLogger(__name__)


class FrameCovarianceNode:
    """
    Per-frame covariance computation.
    Uses ForceTorqueManager exactly as designed.
    """

    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        u = shared_data["reduced_universe"]
        # t = shared_data["time_index"]

        groups = shared_data["groups"]
        levels = shared_data["levels"]
        beads = shared_data["beads"]

        force_avg = shared_data["force_covariances"]
        torque_avg = shared_data["torque_covariances"]
        counts = shared_data["frame_counts"]

        fp = shared_data["args"].force_partitioning

        # # advance trajectory safely
        # u.trajectory[t]

        fragments = u.atoms.fragments

        for group_id, mol_ids in groups.items():
            for mol_id in mol_ids:
                mol = fragments[mol_id]
                trans_axes = mol.principal_axes()

                level_list = levels[mol_id]

                for level in level_list:
                    highest = level == level_list[-1]

                    # --- UNITED ATOM (per residue) ---
                    if level == "united_atom":
                        for res_id in range(len(mol.residues)):
                            key = (mol_id, "united_atom", res_id)
                            bead_groups = beads.get(key, [])
                            if not bead_groups:
                                continue

                            F, T = self._ft.compute_frame_covariance(
                                beads=[u.atoms[idx] for idx in bead_groups],
                                trans_axes=trans_axes,
                                highest_level=highest,
                                force_partitioning=fp,
                            )

                            acc_key = (group_id, res_id)
                            n = counts["ua"].get(acc_key, 0) + 1
                            counts["ua"][acc_key] = n

                            force_avg["ua"][acc_key] = (
                                F
                                if acc_key not in force_avg["ua"]
                                else force_avg["ua"][acc_key]
                                + (F - force_avg["ua"][acc_key]) / n
                            )

                            torque_avg["ua"][acc_key] = (
                                T
                                if acc_key not in torque_avg["ua"]
                                else torque_avg["ua"][acc_key]
                                + (T - torque_avg["ua"][acc_key]) / n
                            )

                    # --- RESIDUE / POLYMER ---
                    elif level in ("residue", "polymer"):
                        key = (mol_id, level)
                        bead_groups = beads.get(key, [])
                        if not bead_groups:
                            continue

                        F, T = self._ft.compute_frame_covariance(
                            beads=[u.atoms[idx] for idx in bead_groups],
                            trans_axes=trans_axes,
                            highest_level=highest,
                            force_partitioning=fp,
                        )

                        k = "res" if level == "residue" else "poly"
                        counts[k][group_id] += 1
                        n = counts[k][group_id]

                        force_avg[k][group_id] = (
                            F
                            if force_avg[k][group_id] is None
                            else force_avg[k][group_id]
                            + (F - force_avg[k][group_id]) / n
                        )

                        torque_avg[k][group_id] = (
                            T
                            if torque_avg[k][group_id] is None
                            else torque_avg[k][group_id]
                            + (T - torque_avg[k][group_id]) / n
                        )
