"""Frame-level covariance (second-moment) construction.

This module computes per-frame second-moment matrices for force and torque
vectors at each hierarchy level (united_atom, residue, polymer). Results are
accumulated as deterministic sums and counts across molecules within a group
for the current frame.

Responsibilities:
- Build bead-level force/torque vectors using ForceTorqueCalculator.
- Construct per-frame force/torque second moments (outer products).
- Optionally construct combined force-torque block matrices.
- Accumulate per-frame matrices and counts across molecules in the same group.

Not responsible for:
- Defining groups/levels/beads mapping (provided via shared context).
- Axis construction policy (delegated to axes_manager).
- Accumulating across frames (handled by the higher-level reducer).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

from CodeEntropy.levels.forces import ForceTorqueCalculator

logger = logging.getLogger(__name__)

FrameCtx = dict[str, Any]
Matrix = np.ndarray


class FrameCovarianceNode:
    """Build per-frame covariance-like (second-moment) matrices for each group.

    This node computes per-frame second-moment matrices (outer products) for
    force and torque generalized vectors at hierarchy levels:

    - united_atom
    - residue
    - polymer

    Within a single frame, outputs are accumulated as sums together with sample
    counts across molecules that belong to the same group. Frame-to-frame
    accumulation is handled elsewhere (by a higher-level reducer).

    """

    def __init__(self) -> None:
        """Initialise the frame covariance node."""
        self._ft = ForceTorqueCalculator()

    def run(self, ctx: FrameCtx) -> dict[str, Any]:
        """Compute and store per-frame force/torque (and optional FT) matrices.

        Args:
            ctx: Frame context dict expected to include:
                - "shared": dict containing reduced_universe, groups, levels, beads,
                  and args
                - shared["axes_manager"] (created in static stage)

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

        out_force: dict[str, dict[Any, Matrix]] = {"ua": {}, "res": {}, "poly": {}}
        out_torque: dict[str, dict[Any, Matrix]] = {"ua": {}, "res": {}, "poly": {}}
        out_counts: dict[str, dict[Any, int]] = {"ua": {}, "res": {}, "poly": {}}

        out_ft: dict[str, dict[Any, Matrix]] | None = (
            {"ua": {}, "res": {}, "poly": {}} if combined else None
        )
        out_ft_counts: dict[str, dict[Any, int]] | None = (
            {"ua": {}, "res": {}, "poly": {}} if combined else None
        )

        for group_id, mol_ids in sorted(groups.items()):
            for mol_id in sorted(mol_ids):
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
                        out_counts=out_counts,
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
                        out_counts=out_counts,
                        out_ft=out_ft,
                        out_ft_counts=out_ft_counts,
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
                        out_counts=out_counts,
                        out_ft=out_ft,
                        out_ft_counts=out_ft_counts,
                        combined=combined,
                    )

        frame_cov: dict[str, Any] = {
            "force": out_force,
            "torque": out_torque,
            "force_counts": out_counts,
            "torque_counts": {
                "ua": dict(out_counts["ua"]),
                "res": dict(out_counts["res"]),
                "poly": dict(out_counts["poly"]),
            },
        }
        if combined and out_ft is not None and out_ft_counts is not None:
            frame_cov["forcetorque"] = out_ft
            frame_cov["forcetorque_counts"] = out_ft_counts

        ctx["frame_covariance"] = frame_cov
        return frame_cov

    def _process_united_atom(
        self,
        *,
        u: Any,
        mol: Any,
        mol_id: int,
        group_id: int,
        beads: dict[Any, list[Any]],
        axes_manager: Any,
        box: np.ndarray | None,
        force_partitioning: float,
        customised_axes: bool,
        is_highest: bool,
        out_force: dict[str, dict[Any, Matrix]],
        out_torque: dict[str, dict[Any, Matrix]],
        out_counts: dict[str, dict[Any, int]],
    ) -> None:
        """Compute UA-level force/torque second moments for one molecule.

        For each residue in the molecule, bead vectors are computed for all UA
        beads in that residue. The resulting second-moment matrices are then
        incrementally averaged across molecules in the same group for this frame.

        Args:
            u: MDAnalysis Universe (or compatible) providing atom access.
            mol: Molecule/fragment object providing residues/atoms.
            mol_id: Molecule id used for bead keying.
            group_id: Group identifier used for within-frame averaging.
            beads: Mapping from bead keys to lists of atom indices.
            axes_manager: Axes manager used to determine axes/centers/MOI.
            box: Optional box vector used for PBC-aware displacements.
            force_partitioning: Force scaling factor applied at highest level.
            customised_axes: Whether to use customised axes methods when available.
            is_highest: Whether the UA level is the highest level for the molecule.
            out_force: Output accumulator for UA force second moments.
            out_torque: Output accumulator for UA torque second moments.
            out_counts: Output accumulator for UA molecule counts.

        Returns:
            None. Mutates out_force/out_torque and out_counts in-place.
        """
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
            out_force["ua"][key] = self._accumulate_sum(out_force["ua"].get(key), F)
            out_torque["ua"][key] = self._accumulate_sum(out_torque["ua"].get(key), T)
            out_counts["ua"][key] = out_counts["ua"].get(key, 0) + 1

    def _process_residue(
        self,
        *,
        u: Any,
        mol: Any,
        mol_id: int,
        group_id: int,
        beads: dict[Any, list[Any]],
        axes_manager: Any,
        box: np.ndarray | None,
        customised_axes: bool,
        force_partitioning: float,
        is_highest: bool,
        out_force: dict[str, dict[Any, Matrix]],
        out_torque: dict[str, dict[Any, Matrix]],
        out_counts: dict[str, dict[Any, int]],
        out_ft: dict[str, dict[Any, Matrix]] | None,
        out_ft_counts: dict[str, dict[Any, int]] | None,
        combined: bool,
    ) -> None:
        """Compute residue-level force/torque (and optional FT) moments for one
        molecule.

        Residue bead vectors are constructed for the molecule and used to compute
        per-frame force and torque second-moment matrices. Outputs are then
        accumulated as sums and counts across molecules in the same group for this
        frame. If combined FT matrices are enabled and this is the highest level,
        a force-torque block matrix is also constructed and averaged.

        Args:
            u: MDAnalysis Universe (or compatible) providing atom access.
            mol: Molecule/fragment object providing atoms/residues.
            mol_id: Molecule id used for bead keying.
            group_id: Group identifier used for within-frame averaging.
            beads: Mapping from bead keys to lists of atom indices.
            axes_manager: Axes manager used to determine axes/centers/MOI.
            box: Optional box vector used for PBC-aware displacements.
            customised_axes: Whether to use customised axes methods when available.
            force_partitioning: Force scaling factor applied at highest level.
            is_highest: Whether residue level is the highest level for the molecule.
            out_force: Output accumulator for residue force second moments.
            out_torque: Output accumulator for residue torque second moments.
            out_counts: Output accumulator for residue molecule counts.
            out_ft: Optional output accumulator for residue combined FT matrices.
            out_ft_counts: Optional output accumulator for residue FT counts.
            combined: Whether combined force-torque matrices are enabled.

        Returns:
            None. Mutates output dictionaries and count accumulators in-place.
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

        out_force["res"][group_id] = self._accumulate_sum(
            out_force["res"].get(group_id), F
        )
        out_torque["res"][group_id] = self._accumulate_sum(
            out_torque["res"].get(group_id), T
        )
        out_counts["res"][group_id] = out_counts["res"].get(group_id, 0) + 1

        if combined and is_highest and out_ft is not None and out_ft_counts is not None:
            M = self._build_ft_block(force_vecs, torque_vecs)
            out_ft["res"][group_id] = self._accumulate_sum(
                out_ft["res"].get(group_id), M
            )
            out_ft_counts["res"][group_id] = out_ft_counts["res"].get(group_id, 0) + 1

    def _process_polymer(
        self,
        *,
        u: Any,
        mol: Any,
        mol_id: int,
        group_id: int,
        beads: dict[Any, list[Any]],
        axes_manager: Any,
        box: np.ndarray | None,
        force_partitioning: float,
        is_highest: bool,
        out_force: dict[str, dict[Any, Matrix]],
        out_torque: dict[str, dict[Any, Matrix]],
        out_counts: dict[str, dict[Any, int]],
        out_ft: dict[str, dict[Any, Matrix]] | None,
        out_ft_counts: dict[str, dict[Any, int]] | None,
        combined: bool,
    ) -> None:
        """Compute polymer-level force/torque (and optional FT) moments for one
        molecule.

        Polymer level uses a single bead. Translation/rotation axes, center, and
        principal moments of inertia are computed, then used to build the
        generalized force and torque vectors. Outputs are accumulated
        as sums and counts across molecules in the same group for this frame.
        If combined FT matrices are enabled and this is the highest level,
        a force-torque block matrix is also constructed and averaged.

        Args:
            u: MDAnalysis Universe (or compatible) providing atom access.
            mol: Molecule/fragment object providing atoms.
            mol_id: Molecule id used for bead keying.
            group_id: Group identifier used for within-frame averaging.
            beads: Mapping from bead keys to lists of atom indices.
            axes_manager: Axes manager used to determine axes/centers/MOI.
            box: Optional box vector used for PBC-aware displacements.
            force_partitioning: Force scaling factor applied at highest level.
            is_highest: Whether polymer level is the highest level for the molecule.
            out_force: Output accumulator for polymer force second moments.
            out_torque: Output accumulator for polymer torque second moments.
            out_counts: Output accumulator for polymer molecule counts.
            out_ft: Optional output accumulator for polymer combined FT matrices.
            out_ft_counts: Optional output accumulator for polymer FT counts.
            combined: Whether combined force-torque matrices are enabled.

        Returns:
            None. Mutates output dictionaries and count accumulators in-place.
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

        out_force["poly"][group_id] = self._accumulate_sum(
            out_force["poly"].get(group_id), F
        )
        out_torque["poly"][group_id] = self._accumulate_sum(
            out_torque["poly"].get(group_id), T
        )
        out_counts["poly"][group_id] = out_counts["poly"].get(group_id, 0) + 1

        if combined and is_highest and out_ft is not None and out_ft_counts is not None:
            M = self._build_ft_block(force_vecs, torque_vecs)
            out_ft["poly"][group_id] = self._accumulate_sum(
                out_ft["poly"].get(group_id), M
            )
            out_ft_counts["poly"][group_id] = out_ft_counts["poly"].get(group_id, 0) + 1

    def _build_ua_vectors(
        self,
        *,
        bead_groups: list[Any],
        residue_atoms: Any,
        axes_manager: Any,
        box: np.ndarray | None,
        force_partitioning: float,
        customised_axes: bool,
        is_highest: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build force/torque vectors for UA-level beads of one residue.

        Args:
            bead_groups: List of UA bead AtomGroups for the residue.
            residue_atoms: AtomGroup for the residue atoms (used for axes when vanilla).
            axes_manager: Axes manager used to determine axes/centers/MOI.
            box: Optional box vector used for PBC-aware displacements.
            force_partitioning: Force scaling factor applied at highest level.
            customised_axes: Whether to use customised axes methods when available.
            is_highest: Whether UA level is the highest level for the molecule.

        Returns:
            A tuple (force_vecs, torque_vecs), each a list of (3,) vectors ordered
            by UA bead index within the residue.
        """
        force_vecs: list[np.ndarray] = []
        torque_vecs: list[np.ndarray] = []

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
        bead_groups: list[Any],
        axes_manager: Any,
        box: np.ndarray | None,
        customised_axes: bool,
        force_partitioning: float,
        is_highest: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build force/torque vectors for residue-level beads of one molecule.

        Args:
            mol: Molecule/fragment object providing residues/atoms.
            bead_groups: List of residue bead AtomGroups for the molecule.
            axes_manager: Axes manager used to determine axes/centers/MOI.
            box: Optional box vector used for PBC-aware displacements.
            customised_axes: Whether to use customised axes methods when available.
            force_partitioning: Force scaling factor applied at highest level.
            is_highest: Whether residue level is the highest level for the molecule.

        Returns:
            A tuple (force_vecs, torque_vecs), each a list of (3,) vectors ordered
            by residue index within the molecule.
        """
        force_vecs: list[np.ndarray] = []
        torque_vecs: list[np.ndarray] = []

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get translation/rotation axes, center and MOI for a residue bead.

        Args:
            mol: Molecule/fragment object providing residues/atoms.
            bead: Residue bead AtomGroup.
            local_res_i: Residue index within the molecule.
            axes_manager: Axes manager used to determine axes/centers/MOI.
            customised_axes: Whether to use customised axes methods when available.

        Returns:
            Tuple (trans_axes, rot_axes, center, moi) where:
              - trans_axes: (3, 3) translation axes
              - rot_axes: (3, 3) rotation axes
              - center: (3,) center of mass
              - moi: (3,) principal moments of inertia
        """
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get translation/rotation axes, center and MOI for a polymer bead.

        Args:
            mol: Molecule/fragment object providing atoms.
            bead: Polymer bead AtomGroup.
            axes_manager: Axes manager used to determine axes/centers/MOI.

        Returns:
            Tuple (trans_axes, rot_axes, center, moi) with shapes (3,3), (3,3), (3,),
            and (3,) respectively.
        """
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
    def _get_shared(ctx: FrameCtx) -> dict[str, Any]:
        """Fetch shared context from a frame context dict.

        Args:
            ctx: Frame context dictionary expected to contain a "shared" key.

        Returns:
            The shared context dict stored at ctx["shared"].

        Raises:
            KeyError: If "shared" is not present in ctx.
        """
        if "shared" not in ctx:
            raise KeyError("FrameCovarianceNode expects ctx['shared'].")
        return ctx["shared"]

    @staticmethod
    def _try_get_box(u: Any) -> np.ndarray | None:
        """Extract a (3,) box vector from an MDAnalysis universe when available.

        Args:
            u: MDAnalysis Universe (or compatible) that may expose dimensions.

        Returns:
            A numpy array of shape (3,) containing box lengths, or None if not
            available.
        """
        try:
            return np.asarray(u.dimensions[:3], dtype=float)
        except Exception:
            return None

    @staticmethod
    def _accumulate_sum(old: np.ndarray | None, new: np.ndarray) -> np.ndarray:
        """Accumulate a deterministic sum of matrix contributions.

        Args:
            old: Previous running sum value, or None for the first sample.
            new: New sample to add into the sum.

        Returns:
            Updated running sum.
        """
        if old is None:
            return new.copy()
        return old + new

    @staticmethod
    def _build_ft_block(
        force_vecs: list[np.ndarray], torque_vecs: list[np.ndarray]
    ) -> np.ndarray:
        """Build a combined force-torque block matrix for a frame.

        For each bead i, create a 6-vector [Fi, Ti]. The block matrix is built
        from outer products of these 6-vectors.

        Args:
            force_vecs: List of per-bead force vectors, each of shape (3,).
            torque_vecs: List of per-bead torque vectors, each of shape (3,).

        Returns:
            A block matrix of shape (6N, 6N) where N is the number of beads.

        Raises:
            ValueError: If force_vecs and torque_vecs have different lengths, if no
                bead vectors are provided, or if any input vector is not length 3.
        """
        if len(force_vecs) != len(torque_vecs):
            raise ValueError("force_vecs and torque_vecs must have the same length.")

        n = len(force_vecs)
        if n == 0:
            raise ValueError("No bead vectors available to build an FT matrix.")

        bead_vecs: list[np.ndarray] = []
        for Fi, Ti in zip(force_vecs, torque_vecs, strict=True):
            Fi = np.asarray(Fi, dtype=float).reshape(-1)
            Ti = np.asarray(Ti, dtype=float).reshape(-1)
            if Fi.size != 3 or Ti.size != 3:
                raise ValueError("Each force/torque vector must be length 3.")
            bead_vecs.append(np.concatenate([Fi, Ti], axis=0))

        blocks: list[list[np.ndarray]] = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                sub = np.outer(bead_vecs[i], bead_vecs[j])
                blocks[i][j] = sub
                blocks[j][i] = sub.T

        return np.block(blocks)
