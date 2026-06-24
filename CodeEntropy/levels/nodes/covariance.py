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

from typing import Any

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

from CodeEntropy.levels.forces import ForceTorqueCalculator

FrameCtx = dict[str, Any]
Matrix = np.ndarray


class FrameCovarianceNode:
    """Build per-frame covariance-like (second-moment) matrices for each group.

    This node computes per-frame second-moment matrices (outer products) for
    force and torque generalized vectors at hierarchy levels:

    - united_atom
    - residue
    - polymer

    Within a single frame, outputs are incrementally averaged across molecules
    that belong to the same group. Frame-to-frame accumulation is handled
    elsewhere (by a higher-level reducer).

    """

    def __init__(self) -> None:
        """Initialise the frame covariance node.

        Creates the force/torque calculator used by all frame-local covariance helper
        methods.
        """
        self._ft = ForceTorqueCalculator()

    def run(self, ctx: FrameCtx) -> dict[str, Any]:
        """Compute frame-local force, torque, and optional force-torque matrices.

        Args:
            ctx: Frame context containing ``shared`` workflow data. The shared data must
                provide ``reduced_universe``, ``groups``, ``levels``, ``beads``, and
                ``args``.

        Returns:
            The frame covariance payload written to ``ctx["frame_covariance"]``.

        Raises:
            KeyError: If ``ctx`` or the shared workflow data is missing required keys.
        """
        shared = self._get_shared(ctx)

        frame_source = shared.get("frame_source")
        if frame_source is None:
            u = shared["reduced_universe"]
        else:
            u = frame_source.universe
        groups = shared["groups"]
        levels = shared["levels"]
        beads = shared["beads"]
        args = shared["args"]
        axes_manager = shared.get("axes_manager")
        axes_topology = shared.get("axes_topology")

        fp = float(args.force_partitioning)
        combined = bool(getattr(args, "combined_forcetorque", False))
        customised_axes = bool(getattr(args, "customised_axes", False))

        box = self._try_get_box(u)
        fragments = u.atoms.fragments

        out_force: dict[str, dict[Any, Matrix]] = {"ua": {}, "res": {}, "poly": {}}
        out_torque: dict[str, dict[Any, Matrix]] = {"ua": {}, "res": {}, "poly": {}}
        out_ft: dict[str, dict[Any, Matrix]] | None = (
            {"ua": {}, "res": {}, "poly": {}} if combined else None
        )

        ua_molcount: dict[tuple[int, int], int] = {}
        res_molcount: dict[int, int] = {}
        poly_molcount: dict[int, int] = {}

        for group_id, mol_ids in sorted(groups.items()):
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
                        axes_topology=axes_topology,
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

        frame_cov: dict[str, Any] = {"force": out_force, "torque": out_torque}
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
        beads: dict[Any, list[Any]],
        axes_manager: Any,
        axes_topology: Any | None,
        box: np.ndarray | None,
        force_partitioning: float,
        customised_axes: bool,
        is_highest: bool,
        out_force: dict[str, dict[Any, Matrix]],
        out_torque: dict[str, dict[Any, Matrix]],
        molcount: dict[tuple[int, int], int],
    ) -> None:
        """Compute united-atom second moments for one molecule.

        Args:
            u: Universe-like object used to resolve bead atom indices.
            mol: Molecule fragment containing residues and atoms.
            mol_id: Molecule index used in bead lookup keys.
            group_id: Molecule-group identifier used for within-frame averaging.
            beads: Mapping of bead keys to reduced-universe atom-index arrays.
            axes_manager: Axes helper used to build translation and rotation axes.
            axes_topology: Optional cached axes topology generated during static setup.
            box: Optional periodic box vector.
            force_partitioning: Force partitioning factor for highest-level vectors.
            customised_axes: Whether customised UA axes should be used.
            is_highest: Whether united atom is the highest active level.
            out_force: Frame-local force second-moment accumulator, mutated in place.
            out_torque: Frame-local torque second-moment accumulator, mutated in place.
            molcount: Per-residue group sample counters, mutated in place.
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
                u=u,
                mol_id=mol_id,
                local_res_i=local_res_i,
                residue_atoms=res.atoms,
                bead_groups=bead_groups,
                axes_manager=axes_manager,
                axes_topology=axes_topology,
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
        beads: dict[Any, list[Any]],
        axes_manager: Any,
        box: np.ndarray | None,
        customised_axes: bool,
        force_partitioning: float,
        is_highest: bool,
        out_force: dict[str, dict[Any, Matrix]],
        out_torque: dict[str, dict[Any, Matrix]],
        out_ft: dict[str, dict[Any, Matrix]] | None,
        molcount: dict[int, int],
        combined: bool,
    ) -> None:
        """Compute residue-level second moments for one molecule.

        Args:
            u: Universe-like object used to resolve bead atom indices.
            mol: Molecule fragment containing residues and atoms.
            mol_id: Molecule index used in bead lookup keys.
            group_id: Molecule-group identifier used for within-frame averaging.
            beads: Mapping of bead keys to reduced-universe atom-index arrays.
            axes_manager: Axes helper used to build translation and rotation axes.
            box: Optional periodic box vector.
            customised_axes: Whether customised residue axes should be used.
            force_partitioning: Force partitioning factor for highest-level vectors.
            is_highest: Whether residue is the highest active level.
            out_force: Frame-local force second-moment accumulator, mutated in place.
            out_torque: Frame-local torque second-moment accumulator, mutated in place.
            out_ft: Optional combined force-torque accumulator, mutated in place.
            molcount: Per-group sample counters, mutated in place.
            combined: Whether combined force-torque matrices should be produced.
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
        beads: dict[Any, list[Any]],
        axes_manager: Any,
        box: np.ndarray | None,
        force_partitioning: float,
        is_highest: bool,
        out_force: dict[str, dict[Any, Matrix]],
        out_torque: dict[str, dict[Any, Matrix]],
        out_ft: dict[str, dict[Any, Matrix]] | None,
        molcount: dict[int, int],
        combined: bool,
    ) -> None:
        """Compute polymer-level second moments for one molecule.

        Args:
            u: Universe-like object used to resolve bead atom indices.
            mol: Molecule fragment containing atoms.
            mol_id: Molecule index used in bead lookup keys.
            group_id: Molecule-group identifier used for within-frame averaging.
            beads: Mapping of bead keys to reduced-universe atom-index arrays.
            axes_manager: Axes helper used to build translation and rotation axes.
            box: Optional periodic box vector.
            force_partitioning: Force partitioning factor for highest-level vectors.
            is_highest: Whether polymer is the highest active level.
            out_force: Frame-local force second-moment accumulator, mutated in place.
            out_torque: Frame-local torque second-moment accumulator, mutated in place.
            out_ft: Optional combined force-torque accumulator, mutated in place.
            molcount: Per-group sample counters, mutated in place.
            combined: Whether combined force-torque matrices should be produced.
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
        u: Any,
        mol_id: int,
        local_res_i: int,
        bead_groups: list[Any],
        residue_atoms: Any,
        axes_manager: Any,
        axes_topology: Any | None,
        box: np.ndarray | None,
        force_partitioning: float,
        customised_axes: bool,
        is_highest: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build force and torque vectors for united-atom beads.

        Args:
            u: Universe-like object used to resolve cached atom indices.
            mol_id: Molecule index used in axes-topology lookup keys.
            local_res_i: Local residue index used in axes-topology lookup keys.
            bead_groups: Atom groups representing UA beads in a residue.
            residue_atoms: Atom group for the parent residue.
            axes_manager: Axes helper used to select axes, centres, and moments.
            axes_topology: Optional cached axes topology generated during static setup.
            box: Optional periodic box vector.
            force_partitioning: Force partitioning factor for highest-level vectors.
            customised_axes: Whether customised UA axes should be used.
            is_highest: Whether UA is the highest active level.

        Returns:
            A tuple containing lists of force vectors and torque vectors.
        """
        force_vecs: list[np.ndarray] = []
        torque_vecs: list[np.ndarray] = []

        for ua_i, bead in enumerate(bead_groups):
            if customised_axes:
                ua_topology = None
                if axes_topology is not None:
                    ua_topology = axes_topology.ua.get((mol_id, local_res_i, ua_i))

                if ua_topology is not None:
                    trans_axes, rot_axes, center, moi = (
                        axes_manager.get_UA_axes_from_topology(
                            u=u,
                            residue_atoms=residue_atoms,
                            topology=ua_topology,
                            box=box,
                        )
                    )
                else:
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
        """Build force and torque vectors for residue beads.

        Args:
            mol: Molecule fragment containing residues and atoms.
            bead_groups: Atom groups representing residue beads.
            axes_manager: Axes helper used to select axes, centres, and moments.
            box: Optional periodic box vector.
            customised_axes: Whether customised residue axes should be used.
            force_partitioning: Force partitioning factor for highest-level vectors.
            is_highest: Whether residue is the highest active level.

        Returns:
            A tuple containing lists of force vectors and torque vectors.
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
        """Return axes, centre, and inertia data for a residue bead.

        Args:
            mol: Molecule fragment containing residues and atoms.
            bead: Atom group representing the residue bead.
            local_res_i: Residue index local to ``mol``.
            axes_manager: Axes helper used to select axes, centres, and moments.
            customised_axes: Whether customised residue axes should be used.

        Returns:
            A tuple of translation axes, rotation axes, centre, and moments of inertia.
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
        """Return axes, centre, and inertia data for a polymer bead.

        Args:
            mol: Molecule fragment containing atoms.
            bead: Atom group representing the polymer bead.
            axes_manager: Axes helper used to select axes, centres, and moments.

        Returns:
            A tuple of translation axes, rotation axes, centre, and moments of inertia.
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
        """Return shared workflow data from a frame context.

        Args:
            ctx: Frame-local context dictionary.

        Returns:
            The shared workflow data stored at ``ctx["shared"]``.

        Raises:
            KeyError: If ``ctx`` does not contain a ``shared`` entry.
        """
        if "shared" not in ctx:
            raise KeyError("FrameCovarianceNode expects ctx['shared'].")
        return ctx["shared"]

    @staticmethod
    def _try_get_box(u: Any) -> np.ndarray | None:
        """Extract periodic box lengths from a universe-like object.

        Args:
            u: Universe-like object that may expose ``dimensions``.

        Returns:
            A three-element NumPy array of box lengths, or ``None`` if unavailable.
        """
        try:
            return np.asarray(u.dimensions[:3], dtype=float)
        except Exception:
            return None

    @staticmethod
    def _inc_mean(old: np.ndarray | None, new: np.ndarray, n: int) -> np.ndarray:
        """Update a running mean with one new sample.

        Args:
            old: Existing running mean, or ``None`` for the first sample.
            new: New sample to incorporate.
            n: One-based sample count after adding ``new``.

        Returns:
            The updated running mean.
        """
        if old is None:
            return new.copy()
        return old + (new - old) / float(n)

    @staticmethod
    def _build_ft_block(
        force_vecs: list[np.ndarray], torque_vecs: list[np.ndarray]
    ) -> np.ndarray:
        """Build a combined force-torque block matrix.

        Args:
            force_vecs: Per-bead force vectors with length three.
            torque_vecs: Per-bead torque vectors with length three.

        Returns:
            A block matrix with shape ``(6N, 6N)`` for ``N`` bead vectors.

        Raises:
            ValueError: If the vector lists differ in length, are empty, or contain
                vectors that are not length three.
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
