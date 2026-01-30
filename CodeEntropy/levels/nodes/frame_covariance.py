# CodeEntropy/levels/nodes/frame_covariance.py

from CodeEntropy.levels.force_torque_manager import ForceTorqueManager


class FrameCovarianceNode:
    """
    MAP stage node: compute covariances for ONE frame.
    Returns per-frame covariance dicts; does not reduce/average.
    """

    def __init__(self):
        self._ft = ForceTorqueManager()

    def run(self, shared_data):
        u = shared_data["reduced_universe"]
        groups = shared_data["groups"]
        levels = shared_data["levels"]
        beads = shared_data["beads"]
        fp = shared_data["args"].force_partitioning

        force_frame = {"ua": {}, "res": {}, "poly": {}}
        torque_frame = {"ua": {}, "res": {}, "poly": {}}

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
                            bead_indices_list = beads.get(bead_key, [])
                            if not bead_indices_list:
                                continue

                            bead_ags = [u.atoms[idx] for idx in bead_indices_list]
                            if any(len(ag) == 0 for ag in bead_ags):
                                continue

                            F, T = self._ft.compute_frame_covariance(
                                beads=bead_ags,
                                trans_axes=trans_axes,
                                highest_level=highest,
                                force_partitioning=fp,
                            )

                            acc_key = (group_id, res_id)
                            force_frame["ua"][acc_key] = F
                            torque_frame["ua"][acc_key] = T

                    elif level in ("residue", "polymer"):
                        bead_key = (mol_id, level)
                        bead_indices_list = beads.get(bead_key, [])
                        if not bead_indices_list:
                            continue

                        bead_ags = [u.atoms[idx] for idx in bead_indices_list]
                        if any(len(ag) == 0 for ag in bead_ags):
                            continue

                        F, T = self._ft.compute_frame_covariance(
                            beads=bead_ags,
                            trans_axes=trans_axes,
                            highest_level=highest,
                            force_partitioning=fp,
                        )

                        k = "res" if level == "residue" else "poly"
                        force_frame[k][group_id] = F
                        torque_frame[k][group_id] = T

        return {"force": force_frame, "torque": torque_frame}
