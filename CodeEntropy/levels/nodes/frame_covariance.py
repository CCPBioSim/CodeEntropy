import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

from CodeEntropy.levels.force_torque_manager import ForceTorqueManager

logger = logging.getLogger(__name__)


class FrameCovarianceNode:
    def __init__(self):
        self._ft = ForceTorqueManager()

    @staticmethod
    def _inc_mean(old: np.ndarray | None, new: np.ndarray, n: int) -> np.ndarray:
        """Incremental mean over molecules within the same frame."""
        if old is None:
            return new.copy()
        return old + (new - old) / float(n)

    @staticmethod
    def _build_ft_block_procedural(force_vecs, torque_vecs) -> np.ndarray:
        """
        Match procedural get_combined_forcetorque_matrices:
        - per bead vector is [Fi, Ti]
        - subblock(i,j) = outer([Fi,Ti], [Fj,Tj])
        - assemble np.block over beads
        """
        if len(force_vecs) != len(torque_vecs):
            raise ValueError("force_vecs and torque_vecs must match length")

        n = len(force_vecs)
        if n == 0:
            raise ValueError("No beads provided for FT matrix build")

        bead_vecs: List[np.ndarray] = []
        for Fi, Ti in zip(force_vecs, torque_vecs):
            Fi = np.asarray(Fi, dtype=float).reshape(-1)
            Ti = np.asarray(Ti, dtype=float).reshape(-1)
            if Fi.shape[0] != 3 or Ti.shape[0] != 3:
                raise ValueError("Each force/torque must be length 3")
            bead_vecs.append(np.concatenate([Fi, Ti], axis=0))

        blocks = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                sub = np.outer(bead_vecs[i], bead_vecs[j])
                blocks[i][j] = sub
                blocks[j][i] = sub.T

        return np.block(blocks)

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
        customised_axes = bool(getattr(args, "customised_axes", False))

        axes_manager = shared.get("axes_manager", None)

        try:
            box = np.asarray(u.dimensions[:3], dtype=float)
        except Exception:
            box = None

        fragments = u.atoms.fragments

        out_force: Dict[str, Dict[Any, np.ndarray]] = {"ua": {}, "res": {}, "poly": {}}
        out_torque: Dict[str, Dict[Any, np.ndarray]] = {"ua": {}, "res": {}, "poly": {}}
        out_ft: Dict[str, Dict[Any, np.ndarray]] | None = (
            {"ua": {}, "res": {}, "poly": {}} if combined else None
        )

        ua_molcount: Dict[Tuple[int, int], int] = {}
        res_molcount: Dict[int, int] = {}
        poly_molcount: Dict[int, int] = {}

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

                        n = ua_molcount.get(key, 0) + 1
                        out_force["ua"][key] = self._inc_mean(
                            out_force["ua"].get(key), F, n
                        )
                        out_torque["ua"][key] = self._inc_mean(
                            out_torque["ua"].get(key), T, n
                        )
                        ua_molcount[key] = n

                if "residue" in level_list:
                    bead_key = (mol_id, "residue")
                    bead_idx_list = beads.get(bead_key, [])
                    if bead_idx_list:
                        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
                        if not any(len(bg) == 0 for bg in bead_groups):
                            force_vecs = []
                            torque_vecs = []

                            highest = "residue" == level_list[-1]

                            for local_res_i, bead in enumerate(bead_groups):
                                if customised_axes and axes_manager is not None:
                                    res = mol.residues[local_res_i]
                                    trans_axes, rot_axes, center, moi = (
                                        axes_manager.get_residue_axes(
                                            mol, local_res_i, residue=res.atoms
                                        )
                                    )
                                else:
                                    make_whole(mol.atoms)
                                    make_whole(bead)
                                    trans_axes = mol.atoms.principal_axes()
                                    if axes_manager is not None:
                                        rot_axes, moi = axes_manager.get_vanilla_axes(
                                            bead
                                        )
                                    else:
                                        rot_axes = np.real(bead.principal_axes())
                                        eigvals, _ = np.linalg.eig(
                                            bead.moment_of_inertia(unwrap=True)
                                        )
                                        moi = sorted(eigvals, reverse=True)
                                    center = bead.center_of_mass(unwrap=True)

                                force_vecs.append(
                                    self._ft.get_weighted_forces(
                                        bead=bead,
                                        trans_axes=np.asarray(trans_axes),
                                        highest_level=highest,
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

                            n = res_molcount.get(group_id, 0) + 1
                            out_force["res"][group_id] = self._inc_mean(
                                out_force["res"].get(group_id), F, n
                            )
                            out_torque["res"][group_id] = self._inc_mean(
                                out_torque["res"].get(group_id), T, n
                            )
                            res_molcount[group_id] = n

                            if combined and highest and out_ft is not None:
                                M = self._build_ft_block_procedural(
                                    force_vecs, torque_vecs
                                )
                                out_ft["res"][group_id] = self._inc_mean(
                                    out_ft["res"].get(group_id), M, n
                                )

                if "polymer" in level_list:
                    bead_key = (mol_id, "polymer")
                    bead_idx_list = beads.get(bead_key, [])
                    if bead_idx_list:
                        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
                        if not any(len(bg) == 0 for bg in bead_groups):
                            bead = bead_groups[0]

                            highest = "polymer" == level_list[-1]

                            if axes_manager is not None:
                                rot_axes, moi = axes_manager.get_vanilla_axes(bead)
                                trans_axes = mol.atoms.principal_axes()
                                center = bead.center_of_mass(unwrap=True)
                            else:
                                make_whole(mol.atoms)
                                make_whole(bead)
                                trans_axes = mol.atoms.principal_axes()
                                rot_axes = np.real(bead.principal_axes())
                                eigvals, _ = np.linalg.eig(
                                    bead.moment_of_inertia(unwrap=True)
                                )
                                moi = sorted(eigvals, reverse=True)
                                center = bead.center_of_mass(unwrap=True)

                            force_vecs = [
                                self._ft.get_weighted_forces(
                                    bead=bead,
                                    trans_axes=np.asarray(trans_axes),
                                    highest_level=highest,
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

                            n = poly_molcount.get(group_id, 0) + 1
                            out_force["poly"][group_id] = self._inc_mean(
                                out_force["poly"].get(group_id), F, n
                            )
                            out_torque["poly"][group_id] = self._inc_mean(
                                out_torque["poly"].get(group_id), T, n
                            )
                            poly_molcount[group_id] = n

                            if combined and highest and out_ft is not None:
                                M = self._build_ft_block_procedural(
                                    force_vecs, torque_vecs
                                )
                                out_ft["poly"][group_id] = self._inc_mean(
                                    out_ft["poly"].get(group_id), M, n
                                )

        frame_cov = {"force": out_force, "torque": out_torque}
        if combined and out_ft is not None:
            frame_cov["forcetorque"] = out_ft

        ctx["frame_covariance"] = frame_cov
        return frame_cov
