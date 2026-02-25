"""Frame-level covariance (second-moment) construction.

This module computes per-frame second-moment matrices for force and torque
vectors at each hierarchy level (united_atom, residue, polymer). Results are
incrementally averaged across molecules within a group for the current frame.

Responsibilities:
- Build bead-level force/torque vectors using ForceTorqueCalculator.
- Construct per-frame force/torque second moments (outer products).
- Optionally construct combined force-torque block matrices.
- Average per-frame matrices across molecules in the same group.

Not responsible for:
- Defining groups/levels/beads mapping (provided via shared context).
- Axis construction policy (delegated to axes_manager).
- Accumulating across frames (handled by the higher-level reducer).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

from CodeEntropy.levels.forces import ForceTorqueCalculator

logger = logging.getLogger(__name__)

FrameCtx = Dict[str, Any]
Matrix = np.ndarray


class FrameCovarianceNode:
    """Build per-frame covariance-like (second-moment) matrices for each group."""

    def __init__(self) -> None:
        self._ft = ForceTorqueCalculator()

    def run(self, ctx: FrameCtx) -> Dict[str, Any]:
        """Compute and store per-frame force/torque (and optional FT) matrices.

        Args:
            ctx: Frame context dict expected to include:
                - "shared": dict containing reduced_universe, groups, levels, beads,
                args
                - MUST include shared["axes_manager"] (created in static stage)

        Returns:
            The frame covariance payload also stored at ctx["frame_covariance"].

        Raises:
            KeyError: If ctx is missing required fields.
        """
        shared = self._get_shared(ctx)

        u = shared["reduced_universe"]
        groups = shared["groups"]
        levels = shared["levels"]
        beads = shared["beads"]
        args = shared["args"]
        axes_manager = shared.get("axes_manager")

        fp = float(args.force_partitioning)
        combined = bool(getattr(args, "combined_forcetorque", False))
        customised_axes = bool(getattr(args, "customised_axes", False))

        box = self._try_get_box(u)
        fragments = u.atoms.fragments

        out_force: Dict[str, Dict[Any, Matrix]] = {"ua": {}, "res": {}, "poly": {}}
        out_torque: Dict[str, Dict[Any, Matrix]] = {"ua": {}, "res": {}, "poly": {}}
        out_ft: Optional[Dict[str, Dict[Any, Matrix]]] = (
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
                    self._process_united_atom(
                        u=u,
                        mol=mol,
                        mol_id=mol_id,
                        group_id=group_id,
                        beads=beads,
                        axes_manager=axes_manager,
                        box=box,
                        force_partitioning=fp,
                        customised_axes=customised_axes,
                        is_highest=("united_atom" == level_list[-1]),
                        out_force=out_force,
                        out_torque=out_torque,
                        molcount=ua_molcount,
                    )

                if "residue" in level_list:
                    self._process_residue(
                        u=u,
                        mol=mol,
                        mol_id=mol_id,
                        group_id=group_id,
                        beads=beads,
                        axes_manager=axes_manager,
                        box=box,
                        customised_axes=customised_axes,
                        force_partitioning=fp,
                        is_highest=("residue" == level_list[-1]),
                        out_force=out_force,
                        out_torque=out_torque,
                        out_ft=out_ft,
                        molcount=res_molcount,
                        combined=combined,
                    )

                if "polymer" in level_list:
                    self._process_polymer(
                        u=u,
                        mol=mol,
                        mol_id=mol_id,
                        group_id=group_id,
                        beads=beads,
                        axes_manager=axes_manager,
                        box=box,
                        force_partitioning=fp,
                        is_highest=("polymer" == level_list[-1]),
                        out_force=out_force,
                        out_torque=out_torque,
                        out_ft=out_ft,
                        molcount=poly_molcount,
                        combined=combined,
                    )

        frame_cov: Dict[str, Any] = {"force": out_force, "torque": out_torque}
        if combined and out_ft is not None:
            frame_cov["forcetorque"] = out_ft

        ctx["frame_covariance"] = frame_cov
        return frame_cov

    def _process_united_atom(
        self,
        *,
        u: Any,
        mol: Any,
        mol_id: int,
        group_id: int,
        beads: Dict[Any, List[Any]],
        axes_manager: Any,
        box: Optional[np.ndarray],
        force_partitioning: float,
        customised_axes: bool,
        is_highest: bool,
        out_force: Dict[str, Dict[Any, Matrix]],
        out_torque: Dict[str, Dict[Any, Matrix]],
        molcount: Dict[Tuple[int, int], int],
    ) -> None:
        for local_res_i, res in enumerate(mol.residues):
            bead_key = (mol_id, "united_atom", local_res_i)
            bead_idx_list = beads.get(bead_key, [])
            if not bead_idx_list:
                continue

            bead_groups = [u.atoms[idx] for idx in bead_idx_list]
            if any(len(bg) == 0 for bg in bead_groups):
                continue

            force_vecs, torque_vecs = self._build_ua_vectors(
                residue_atoms=res.atoms,
                bead_groups=bead_groups,
                axes_manager=axes_manager,
                box=box,
                force_partitioning=force_partitioning,
                customised_axes=customised_axes,
                is_highest=is_highest,
            )

            F, T = self._ft.compute_frame_covariance(force_vecs, torque_vecs)

            key = (group_id, local_res_i)
            n = molcount.get(key, 0) + 1
            out_force["ua"][key] = self._inc_mean(out_force["ua"].get(key), F, n)
            out_torque["ua"][key] = self._inc_mean(out_torque["ua"].get(key), T, n)
            molcount[key] = n

    def _process_residue(
        self,
        *,
        u: Any,
        mol: Any,
        mol_id: int,
        group_id: int,
        beads: Dict[Any, List[Any]],
        axes_manager: Any,
        box: Optional[np.ndarray],
        customised_axes: bool,
        force_partitioning: float,
        is_highest: bool,
        out_force: Dict[str, Dict[Any, Matrix]],
        out_torque: Dict[str, Dict[Any, Matrix]],
        out_ft: Optional[Dict[str, Dict[Any, Matrix]]],
        molcount: Dict[int, int],
        combined: bool,
    ) -> None:
        """
        Compute residue-level force/torque (and optional FT) moments for one molecule.
        """
        bead_key = (mol_id, "residue")
        bead_idx_list = beads.get(bead_key, [])
        if not bead_idx_list:
            return

        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
        if any(len(bg) == 0 for bg in bead_groups):
            return

        force_vecs, torque_vecs = self._build_residue_vectors(
            mol=mol,
            bead_groups=bead_groups,
            axes_manager=axes_manager,
            box=box,
            customised_axes=customised_axes,
            force_partitioning=force_partitioning,
            is_highest=is_highest,
        )

        F, T = self._ft.compute_frame_covariance(force_vecs, torque_vecs)

        n = molcount.get(group_id, 0) + 1
        out_force["res"][group_id] = self._inc_mean(
            out_force["res"].get(group_id), F, n
        )
        out_torque["res"][group_id] = self._inc_mean(
            out_torque["res"].get(group_id), T, n
        )
        molcount[group_id] = n

        if combined and is_highest and out_ft is not None:
            M = self._build_ft_block(force_vecs, torque_vecs)
            out_ft["res"][group_id] = self._inc_mean(out_ft["res"].get(group_id), M, n)

    def _process_polymer(
        self,
        *,
        u: Any,
        mol: Any,
        mol_id: int,
        group_id: int,
        beads: Dict[Any, List[Any]],
        axes_manager: Any,
        box: Optional[np.ndarray],
        force_partitioning: float,
        is_highest: bool,
        out_force: Dict[str, Dict[Any, Matrix]],
        out_torque: Dict[str, Dict[Any, Matrix]],
        out_ft: Optional[Dict[str, Dict[Any, Matrix]]],
        molcount: Dict[int, int],
        combined: bool,
    ) -> None:
        """
        Compute polymer-level force/torque (and optional FT) moments for one molecule.
        """
        bead_key = (mol_id, "polymer")
        bead_idx_list = beads.get(bead_key, [])
        if not bead_idx_list:
            return

        bead_groups = [u.atoms[idx] for idx in bead_idx_list]
        if any(len(bg) == 0 for bg in bead_groups):
            return

        bead = bead_groups[0]

        trans_axes, rot_axes, center, moi = self._get_polymer_axes(
            mol=mol, bead=bead, axes_manager=axes_manager
        )

        force_vecs = [
            self._ft.get_weighted_forces(
                bead=bead,
                trans_axes=np.asarray(trans_axes),
                highest_level=is_highest,
                force_partitioning=force_partitioning,
            )
        ]
        torque_vecs = [
            self._ft.get_weighted_torques(
                bead=bead,
                rot_axes=np.asarray(rot_axes),
                center=np.asarray(center),
                force_partitioning=force_partitioning,
                moment_of_inertia=np.asarray(moi),
                axes_manager=axes_manager,
                box=box,
            )
        ]

        F, T = self._ft.compute_frame_covariance(force_vecs, torque_vecs)

        n = molcount.get(group_id, 0) + 1
        out_force["poly"][group_id] = self._inc_mean(
            out_force["poly"].get(group_id), F, n
        )
        out_torque["poly"][group_id] = self._inc_mean(
            out_torque["poly"].get(group_id), T, n
        )
        molcount[group_id] = n

        if combined and is_highest and out_ft is not None:
            M = self._build_ft_block(force_vecs, torque_vecs)
            out_ft["poly"][group_id] = self._inc_mean(
                out_ft["poly"].get(group_id), M, n
            )

    def _build_ua_vectors(
        self,
        *,
        bead_groups: List[Any],
        residue_atoms: Any,
        axes_manager: Any,
        box: Optional[np.ndarray],
        force_partitioning: float,
        customised_axes: bool,
        is_highest: bool,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Build force/torque vectors for UA-level beads of one residue."""
        force_vecs: List[np.ndarray] = []
        torque_vecs: List[np.ndarray] = []

        for ua_i, bead in enumerate(bead_groups):
            if customised_axes:
                trans_axes, rot_axes, center, moi = axes_manager.get_UA_axes(
                    residue_atoms, ua_i
                )
            else:
                make_whole(residue_atoms)
                make_whole(bead)

                trans_axes = residue_atoms.principal_axes()
                rot_axes, moi = axes_manager.get_vanilla_axes(bead)
                center = bead.center_of_mass(unwrap=True)

            force_vecs.append(
                self._ft.get_weighted_forces(
                    bead=bead,
                    trans_axes=np.asarray(trans_axes),
                    highest_level=is_highest,
                    force_partitioning=force_partitioning,
                )
            )
            torque_vecs.append(
                self._ft.get_weighted_torques(
                    bead=bead,
                    rot_axes=np.asarray(rot_axes),
                    center=np.asarray(center),
                    force_partitioning=force_partitioning,
                    moment_of_inertia=np.asarray(moi),
                    axes_manager=axes_manager,
                    box=box,
                )
            )

        return force_vecs, torque_vecs

    def _build_residue_vectors(
        self,
        *,
        mol: Any,
        bead_groups: List[Any],
        axes_manager: Any,
        box: Optional[np.ndarray],
        customised_axes: bool,
        force_partitioning: float,
        is_highest: bool,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Build force/torque vectors for residue-level beads of one molecule."""
        force_vecs: List[np.ndarray] = []
        torque_vecs: List[np.ndarray] = []

        for local_res_i, bead in enumerate(bead_groups):
            trans_axes, rot_axes, center, moi = self._get_residue_axes(
                mol=mol,
                bead=bead,
                local_res_i=local_res_i,
                axes_manager=axes_manager,
                customised_axes=customised_axes,
            )

            force_vecs.append(
                self._ft.get_weighted_forces(
                    bead=bead,
                    trans_axes=np.asarray(trans_axes),
                    highest_level=is_highest,
                    force_partitioning=force_partitioning,
                )
            )
            torque_vecs.append(
                self._ft.get_weighted_torques(
                    bead=bead,
                    rot_axes=np.asarray(rot_axes),
                    center=np.asarray(center),
                    force_partitioning=force_partitioning,
                    moment_of_inertia=np.asarray(moi),
                    axes_manager=axes_manager,
                    box=box,
                )
            )

        return force_vecs, torque_vecs

    def _get_residue_axes(
        self,
        *,
        mol: Any,
        bead: Any,
        local_res_i: int,
        axes_manager: Any,
        customised_axes: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get translation/rotation axes, center and MOI for a residue bead."""
        if customised_axes:
            res = mol.residues[local_res_i]
            return axes_manager.get_residue_axes(mol, local_res_i, residue=res.atoms)

        make_whole(mol.atoms)
        make_whole(bead)

        trans_axes = mol.atoms.principal_axes()
        rot_axes, moi = axes_manager.get_vanilla_axes(bead)
        center = bead.center_of_mass(unwrap=True)
        return (
            np.asarray(trans_axes),
            np.asarray(rot_axes),
            np.asarray(center),
            np.asarray(moi),
        )

    def _get_polymer_axes(
        self,
        *,
        mol: Any,
        bead: Any,
        axes_manager: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get translation/rotation axes, center and MOI for a polymer bead."""
        make_whole(mol.atoms)
        make_whole(bead)

        trans_axes = mol.atoms.principal_axes()
        rot_axes, moi = axes_manager.get_vanilla_axes(bead)
        center = bead.center_of_mass(unwrap=True)

        return (
            np.asarray(trans_axes),
            np.asarray(rot_axes),
            np.asarray(center),
            np.asarray(moi),
        )

    @staticmethod
    def _get_shared(ctx: FrameCtx) -> Dict[str, Any]:
        """Fetch shared context from a frame context dict."""
        if "shared" not in ctx:
            raise KeyError("FrameCovarianceNode expects ctx['shared'].")
        return ctx["shared"]

    @staticmethod
    def _try_get_box(u: Any) -> Optional[np.ndarray]:
        """Extract a (3,) box vector from an MDAnalysis universe when available."""
        try:
            return np.asarray(u.dimensions[:3], dtype=float)
        except Exception:
            return None

    @staticmethod
    def _inc_mean(old: Optional[np.ndarray], new: np.ndarray, n: int) -> np.ndarray:
        """Compute an incremental mean (streaming average)."""
        if old is None:
            return new.copy()
        return old + (new - old) / float(n)

    @staticmethod
    def _build_ft_block(
        force_vecs: List[np.ndarray], torque_vecs: List[np.ndarray]
    ) -> np.ndarray:
        """Build a combined force-torque block matrix for a frame.

        For each bead i, create a 6-vector [Fi, Ti]. The block matrix is built
        from outer products of these 6-vectors.
        """
        if len(force_vecs) != len(torque_vecs):
            raise ValueError("force_vecs and torque_vecs must have the same length.")

        n = len(force_vecs)
        if n == 0:
            raise ValueError("No bead vectors available to build an FT matrix.")

        bead_vecs: List[np.ndarray] = []
        for Fi, Ti in zip(force_vecs, torque_vecs, strict=True):
            Fi = np.asarray(Fi, dtype=float).reshape(-1)
            Ti = np.asarray(Ti, dtype=float).reshape(-1)
            if Fi.size != 3 or Ti.size != 3:
                raise ValueError("Each force/torque vector must be length 3.")
            bead_vecs.append(np.concatenate([Fi, Ti], axis=0))

        blocks: List[List[np.ndarray]] = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                sub = np.outer(bead_vecs[i], bead_vecs[j])
                blocks[i][j] = sub
                blocks[j][i] = sub.T

        return np.block(blocks)
