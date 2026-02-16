import logging
from typing import Any, Dict, List

import numpy as np

from CodeEntropy.levels.force_torque_manager import ForceTorqueManager

logger = logging.getLogger(__name__)


class FrameCovarianceNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    @staticmethod
    def _full_ft_second_moment(
        force_vecs: List[np.ndarray], torque_vecs: List[np.ndarray]
    ) -> np.ndarray:
        """
        Procedural-equivalent FT construction:

        Build a FULL 6N x 6N second-moment matrix from concatenated bead vectors:
            [F1, F2, ... FN, T1, T2, ... TN]
        where each Fi, Ti is a 3-vector (already projected/weighted).

        This includes the F<->T cross blocks (off-diagonal blocks).
        """
        if len(force_vecs) != len(torque_vecs):
            raise ValueError(
                "force_vecs and torque_vecs must have the same number of beads"
            )

        if len(force_vecs) == 0:
            raise ValueError("force_vecs/torque_vecs are empty")

        f = [np.asarray(v, dtype=float).reshape(-1) for v in force_vecs]
        t = [np.asarray(v, dtype=float).reshape(-1) for v in torque_vecs]

        if any(v.shape[0] != 3 for v in f + t):
            raise ValueError(
                "Each force/torque vector must be length 3 after weighting"
            )

        flat = np.concatenate(f + t, axis=0)
        return np.outer(flat, flat)

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if "shared" not in ctx:
            raise KeyError("FrameCovarianceNode expects ctx['shared'].")

        shared = ctx["shared"]
        u = shared["reduced_universe"]

        groups = shared["groups"]
        levels = shared["levels"]
        beads = shared["beads"]
        args = shared["args"]

        fp = args.force_partitioning
        combined = bool(getattr(args, "combined_forcetorque", False))

        axes_manager = shared.get("axes_manager", None)
        box = None
        try:
            box = np.asarray(u.dimensions[:3], dtype=float)
        except Exception:
            box = None

        fragments = u.atoms.fragments

        out_force = {"ua": {}, "res": {}, "poly": {}}
        out_torque = {"ua": {}, "res": {}, "poly": {}}
        out_ft = {"ua": {}, "res": {}, "poly": {}} if combined else None

        for group_id, mol_ids in groups.items():
            for mol_id in mol_ids:
                mol = fragments[mol_id]
                level_list = levels[mol_id]

                if "united_atom" in level_list:
                    for local_res_i, res in enumerate(mol.residues):
                        bead_key = (mol_id, "united_atom", local_res_i)
                        bead_idx_list = beads.get(bead_key, [])
                        if not bead_idx_list:
                            continue

                        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
                        if any(len(bg) == 0 for bg in bead_groups):
                            continue

                        force_vecs = []
                        torque_vecs = []

                        for ua_i, bead in enumerate(bead_groups):
                            trans_axes, rot_axes, center, moi = (
                                axes_manager.get_UA_axes(res.atoms, ua_i)
                            )

                            force_vecs.append(
                                self._ft.get_weighted_forces(
                                    bead=bead,
                                    trans_axes=np.asarray(trans_axes),
                                    highest_level=False,
                                    force_partitioning=fp,
                                )
                            )
                            torque_vecs.append(
                                self._ft.get_weighted_torques(
                                    bead=bead,
                                    rot_axes=np.asarray(rot_axes),
                                    center=np.asarray(center),
                                    force_partitioning=fp,
                                    moment_of_inertia=np.asarray(moi),
                                    axes_manager=axes_manager,
                                    box=box,
                                )
                            )

                        F, T = self._ft.compute_frame_covariance(
                            force_vecs, torque_vecs
                        )

                        key = (group_id, local_res_i)
                        out_force["ua"][key] = F
                        out_torque["ua"][key] = T
                        if combined and out_ft is not None:
                            out_ft["ua"][key] = self._full_ft_second_moment(
                                force_vecs, torque_vecs
                            )

                if "residue" in level_list:
                    bead_key = (mol_id, "residue")
                    bead_idx_list = beads.get(bead_key, [])
                    if bead_idx_list:
                        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
                        if not any(len(bg) == 0 for bg in bead_groups):

                            force_vecs = []
                            torque_vecs = []

                            for local_res_i, bead in enumerate(bead_groups):
                                res = mol.residues[local_res_i]
                                trans_axes, rot_axes, center, moi = (
                                    axes_manager.get_residue_axes(
                                        mol, int(res.resindex)
                                    )
                                )

                                force_vecs.append(
                                    self._ft.get_weighted_forces(
                                        bead=bead,
                                        trans_axes=np.asarray(trans_axes),
                                        highest_level=False,
                                        force_partitioning=fp,
                                    )
                                )
                                torque_vecs.append(
                                    self._ft.get_weighted_torques(
                                        bead=bead,
                                        rot_axes=np.asarray(rot_axes),
                                        center=np.asarray(center),
                                        force_partitioning=fp,
                                        moment_of_inertia=np.asarray(moi),
                                        axes_manager=axes_manager,
                                        box=box,
                                    )
                                )

                            F, T = self._ft.compute_frame_covariance(
                                force_vecs, torque_vecs
                            )

                            out_force["res"][group_id] = F
                            out_torque["res"][group_id] = T

                            if combined and out_ft is not None:
                                out_ft["res"][group_id] = self._full_ft_second_moment(
                                    force_vecs, torque_vecs
                                )

                if "polymer" in level_list:
                    bead_key = (mol_id, "polymer")
                    bead_idx_list = beads.get(bead_key, [])
                    if bead_idx_list:
                        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
                        if not any(len(bg) == 0 for bg in bead_groups):
                            bead = bead_groups[0]

                            if axes_manager is not None:
                                rot_axes, moi = axes_manager.get_vanilla_axes(bead)
                                trans_axes = rot_axes
                                center = bead.center_of_mass(unwrap=True)
                            else:
                                trans_axes = bead.principal_axes()
                                rot_axes = trans_axes
                                center = bead.center_of_mass(unwrap=True)
                                moi = np.linalg.eigvals(bead.moment_of_inertia())

                            force_vecs = [
                                self._ft.get_weighted_forces(
                                    bead=bead,
                                    trans_axes=np.asarray(trans_axes),
                                    highest_level=True,
                                    force_partitioning=fp,
                                )
                            ]
                            torque_vecs = [
                                self._ft.get_weighted_torques(
                                    bead=bead,
                                    rot_axes=np.asarray(rot_axes),
                                    center=np.asarray(center),
                                    force_partitioning=fp,
                                    moment_of_inertia=np.asarray(moi),
                                    axes_manager=axes_manager,
                                    box=box,
                                )
                            ]

                            F, T = self._ft.compute_frame_covariance(
                                force_vecs, torque_vecs
                            )

                            out_force["poly"][group_id] = F
                            out_torque["poly"][group_id] = T

                            if combined and out_ft is not None:
                                out_ft["poly"][group_id] = self._full_ft_second_moment(
                                    force_vecs, torque_vecs
                                )
                                # M = out_ft["poly"][group_id]
                                # half = M.shape[0] // 2
                                # cross_norm = np.linalg.norm(M[:half, half:])
                                # logger.warning(f"[FT DEBUG] group={group_id} "
                                # f"cross_norm={cross_norm:.6e}")

        frame_cov = {"force": out_force, "torque": out_torque}
        if combined and out_ft is not None:
            frame_cov["forcetorque"] = out_ft

        ctx["frame_covariance"] = frame_cov
        return frame_cov
