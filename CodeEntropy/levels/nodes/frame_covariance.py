import inspect
import logging
from typing import Any, Dict, Tuple

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

from CodeEntropy.axes import AxesManager
from CodeEntropy.levels.force_torque_manager import ForceTorqueManager

logger = logging.getLogger(__name__)


class FrameCovarianceNode:
    """
    Per-frame covariance computation (FRAME DAG node).

    Input (frame_ctx):
      frame_ctx["shared"]        -> shared_data dict
      frame_ctx["frame_index"]   -> absolute trajectory frame index
      frame_ctx["frame_axes"]    -> optional output from FrameAxesNode:
                                   {"trans": {mol_id: 3x3}, "custom": bool}

    Output (frame_ctx["frame_covariance"]):
      {
        "force":  {"ua": {(gid,res_id): F}, "res": {gid: F}, "poly": {gid: F}},
        "torque": {"ua": {(gid,res_id): T}, "res": {gid: T}, "poly": {gid: T}},
      }

    IMPORTANT:
      - produces pure per-frame output only
      - reduction/averaging happens in LevelDAG
    """

    def __init__(self):
        self._ft = ForceTorqueManager()
        self._ft_sig = inspect.signature(self._ft.compute_frame_covariance)
        self._axes_manager = AxesManager()

    def _call_ft(
        self,
        *,
        data_container,
        beads,
        trans_axes,
        highest_level: bool,
        force_partitioning: float,
        extra: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call ForceTorqueManager.compute_frame_covariance in a signature-adaptive way.
        """
        kwargs = {}

        if "data_container" in self._ft_sig.parameters:
            kwargs["data_container"] = data_container
        if "beads" in self._ft_sig.parameters:
            kwargs["beads"] = beads
        if "trans_axes" in self._ft_sig.parameters:
            kwargs["trans_axes"] = trans_axes
        if "highest_level" in self._ft_sig.parameters:
            kwargs["highest_level"] = highest_level
        if "force_partitioning" in self._ft_sig.parameters:
            kwargs["force_partitioning"] = force_partitioning

        for k, v in extra.items():
            if k in self._ft_sig.parameters:
                kwargs[k] = v

        return self._ft.compute_frame_covariance(**kwargs)

    @staticmethod
    def _prepare_vanilla_geometry(beads):
        """
        Vanilla geometry path matching procedural fallback:
          make_whole(bead), principal_axes, COM unwrap=True, MOI unwrap=True
        Returns: rot_axes_list, centers, mois
        """
        rot_axes_list = []
        centers = []
        mois = []

        for bead_ag in beads:
            make_whole(bead_ag)
            rot_axes = np.real(bead_ag.principal_axes())
            center = bead_ag.center_of_mass(unwrap=True)
            eigvals, _ = np.linalg.eig(bead_ag.moment_of_inertia(unwrap=True))
            moi = sorted(np.real(eigvals), reverse=True)

            rot_axes_list.append(rot_axes)
            centers.append(np.asarray(center, dtype=float))
            mois.append(np.asarray(moi, dtype=float))

        return rot_axes_list, centers, mois

    def _prepare_custom_geometry(self, *, level: str, data_container, n_beads: int):
        """
        Customised axes path (procedural parity) if AxesManager exposes:
          - get_UA_axes(data_container, bead_index)
          - get_residue_axes(data_container, bead_index)

        Returns: trans_axes, rot_axes_list, centers, mois
        """
        if level == "united_atom" and hasattr(self._axes_manager, "get_UA_axes"):
            trans0, _, _, _ = self._axes_manager.get_UA_axes(data_container, 0)
            rot_axes_list, centers, mois = [], [], []
            for bead_index in range(n_beads):
                _t, rot_axes, center, moi = self._axes_manager.get_UA_axes(
                    data_container, bead_index
                )
                rot_axes_list.append(np.asarray(rot_axes))
                centers.append(np.asarray(center, dtype=float))
                mois.append(np.asarray(moi, dtype=float))
            return np.asarray(trans0), rot_axes_list, centers, mois

        if level == "residue" and hasattr(self._axes_manager, "get_residue_axes"):
            trans0, _, _, _ = self._axes_manager.get_residue_axes(data_container, 0)
            rot_axes_list, centers, mois = [], [], []
            for bead_index in range(n_beads):
                _t, rot_axes, center, moi = self._axes_manager.get_residue_axes(
                    data_container, bead_index
                )
                rot_axes_list.append(np.asarray(rot_axes))
                centers.append(np.asarray(center, dtype=float))
                mois.append(np.asarray(moi, dtype=float))
            return np.asarray(trans0), rot_axes_list, centers, mois

        raise RuntimeError(
            "Custom geometry requested but AxesManager lacks required methods."
        )

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        shared = ctx.get("shared", ctx)

        u = shared.get("reduced_universe", shared.get("universe"))
        if u is None:
            raise KeyError("shared_data must contain 'reduced_universe' or 'universe'")

        args = shared["args"]
        fp = args.force_partitioning

        groups = shared["groups"]
        levels = shared["levels"]
        beads_map = shared["beads"]

        frame_index = ctx.get("frame_index", shared.get("frame_index", 0))
        u.trajectory[frame_index]

        frame_axes = ctx.get("frame_axes") or {}
        trans_axes_by_mol = frame_axes.get("trans", {})
        customised_axes = bool(
            frame_axes.get("custom", getattr(args, "customised_axes", False))
        )

        fragments = u.atoms.fragments

        dims = None
        try:
            dims = np.asarray(u.dimensions[:3], dtype=float)
        except Exception:
            dims = None

        out_force = {"ua": {}, "res": {}, "poly": {}}
        out_torque = {"ua": {}, "res": {}, "poly": {}}

        for group_id, mol_ids in groups.items():
            for mol_id in mol_ids:
                mol = fragments[mol_id]
                level_list = levels[mol_id]

                base_trans_axes = trans_axes_by_mol.get(
                    mol_id, np.asarray(mol.principal_axes())
                )

                for level in level_list:
                    highest = level == level_list[-1]

                    if level == "united_atom":
                        for res_id, residue in enumerate(mol.residues):
                            bead_key = (mol_id, "united_atom", res_id)
                            bead_idx_list = beads_map.get(bead_key, [])
                            if not bead_idx_list:
                                continue

                            bead_groups = [u.atoms[idx] for idx in bead_idx_list]
                            if any(len(bg) == 0 for bg in bead_groups):
                                continue

                            data_container = residue.atoms

                            if customised_axes and hasattr(
                                self._axes_manager, "get_UA_axes"
                            ):
                                trans_axes, rot_axes_list, centers, mois = (
                                    self._prepare_custom_geometry(
                                        level="united_atom",
                                        data_container=data_container,
                                        n_beads=len(bead_groups),
                                    )
                                )
                            else:
                                make_whole(data_container.atoms)
                                trans_axes = np.asarray(
                                    data_container.atoms.principal_axes()
                                )
                                rot_axes_list, centers, mois = (
                                    self._prepare_vanilla_geometry(bead_groups)
                                )

                            extra = {
                                "rot_axes_list": rot_axes_list,
                                "centers": centers,
                                "mois": mois,
                                "axes_manager": self._axes_manager,
                                "dimensions": dims,
                            }

                            F, T = self._call_ft(
                                data_container=data_container,
                                beads=bead_groups,
                                trans_axes=trans_axes,
                                highest_level=highest,
                                force_partitioning=fp,
                                extra=extra,
                            )

                            out_force["ua"][(group_id, res_id)] = F
                            out_torque["ua"][(group_id, res_id)] = T

                    elif level in ("residue", "polymer"):
                        bead_key = (mol_id, level)
                        bead_idx_list = beads_map.get(bead_key, [])
                        if not bead_idx_list:
                            continue

                        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
                        if any(len(bg) == 0 for bg in bead_groups):
                            continue

                        data_container = mol

                        if (
                            level == "residue"
                            and customised_axes
                            and hasattr(self._axes_manager, "get_residue_axes")
                        ):
                            trans_axes, rot_axes_list, centers, mois = (
                                self._prepare_custom_geometry(
                                    level="residue",
                                    data_container=data_container,
                                    n_beads=len(bead_groups),
                                )
                            )
                        else:
                            make_whole(data_container.atoms)
                            trans_axes = np.asarray(base_trans_axes)
                            rot_axes_list, centers, mois = (
                                self._prepare_vanilla_geometry(bead_groups)
                            )

                        extra = {
                            "rot_axes_list": rot_axes_list,
                            "centers": centers,
                            "mois": mois,
                            "axes_manager": self._axes_manager,
                            "dimensions": dims,
                        }

                        F, T = self._call_ft(
                            data_container=data_container,
                            beads=bead_groups,
                            trans_axes=trans_axes,
                            highest_level=highest,
                            force_partitioning=fp,
                            extra=extra,
                        )

                        bucket = "res" if level == "residue" else "poly"
                        out_force[bucket][group_id] = F
                        out_torque[bucket][group_id] = T

                    else:
                        raise ValueError(f"Unknown level: {level}")

        ctx["frame_covariance"] = {"force": out_force, "torque": out_torque}
        return ctx["frame_covariance"]
