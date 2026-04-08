"""Node for computing vibrational entropy from covariance matrices."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from CodeEntropy.entropy.vibrational import VibrationalEntropy
from CodeEntropy.levels.linalg import MatrixUtils

logger = logging.getLogger(__name__)


GroupId = int
ResidueId = int
CovKey = tuple[GroupId, ResidueId]


@dataclass(frozen=True)
class EntropyPair:
    """Container for paired translational and rotational entropy values.

    Attributes:
        trans: Translational vibrational entropy value.
        rot: Rotational vibrational entropy value.
    """

    trans: float
    rot: float


class VibrationalEntropyNode:
    """Compute vibrational entropy from force/torque (and optional FT) covariances.

    This node reads covariance matrices from a shared data mapping, computes
    translational and rotational vibrational entropy at requested hierarchy levels,
    and stores results back into the shared data structure.

    The node supports:
      - Force and torque covariance matrices ("force" / "torque") at residue/polymer
        levels.
      - United-atom per-residue covariances keyed by (group_id, residue_id).
      - Optional combined force-torque covariance matrices ("forcetorque") for the
        highest level when enabled via args.combined_forcetorque.
    """

    def __init__(self) -> None:
        """Initialize the node with matrix utilities and numerical tolerances."""
        self._mat_ops = MatrixUtils()
        self._zero_atol = 1e-8

    def run(self, shared_data: MutableMapping[str, Any], **_: Any) -> dict[str, Any]:
        """Run vibrational entropy calculations and update the shared data mapping.

        Args:
            shared_data: Mutable mapping containing inputs (covariances, groups,
                levels, args, etc.) and where outputs will be written.
            **_: Unused keyword arguments, accepted for framework compatibility.

        Returns:
            A dict containing the computed vibrational entropy results under the
            key "vibrational_entropy".

        Raises:
            ValueError: If an unknown level is encountered in the level list for a
                representative molecule.
        """
        ve = self._build_entropy_engine(shared_data)
        temp = shared_data["args"].temperature

        groups = shared_data["groups"]
        levels = shared_data["levels"]
        fragments = shared_data["reduced_universe"].atoms.fragments
        flexible = shared_data["flexible_dihedrals"]

        gid2i = self._get_group_id_to_index(shared_data)

        force_cov = shared_data["force_covariances"]
        torque_cov = shared_data["torque_covariances"]

        combined = bool(getattr(shared_data["args"], "combined_forcetorque", False))
        ft_cov = shared_data.get("forcetorque_covariances") if combined else None

        ua_frame_counts = self._get_ua_frame_counts(shared_data)
        reporter = shared_data.get("reporter")

        results: dict[int, dict[str, dict[str, float]]] = {}

        for group_id, mol_ids in sorted(groups.items()):
            results[group_id] = {}
            if not mol_ids:
                continue

            rep_mol_id = mol_ids[0]
            rep_mol = fragments[rep_mol_id]
            level_list = levels[rep_mol_id]

            for level in level_list:
                highest = level == level_list[-1]

                if level == "united_atom":
                    pair = self._compute_united_atom_entropy(
                        ve=ve,
                        temp=temp,
                        group_id=group_id,
                        residues=rep_mol.residues,
                        force_ua=force_cov["ua"],
                        torque_ua=torque_cov["ua"],
                        flexible_ua=flexible["ua"],
                        ua_frame_counts=ua_frame_counts,
                        reporter=reporter,
                        n_frames_default=shared_data.get("n_frames", 0),
                        highest=highest,
                    )
                    self._store_results(results, group_id, level, pair)
                    self._log_molecule_level_results(
                        reporter, group_id, level, pair, use_ft_labels=False
                    )
                    continue

                if level in ("residue", "polymer"):
                    gi = gid2i[group_id]

                    if level == "residue":
                        flexible_res = flexible["res"][group_id]
                    else:
                        flexible_res = 0  # No polymer level flexible dihedrals

                    if combined and highest and ft_cov is not None:
                        ft_key = "res" if level == "residue" else "poly"
                        ftmat = self._get_indexed_matrix(ft_cov.get(ft_key, []), gi)

                        pair = self._compute_ft_entropy(
                            ve=ve, temp=temp, ftmat=ftmat, flexible=flexible_res
                        )
                        self._store_results(results, group_id, level, pair)
                        self._log_molecule_level_results(
                            reporter, group_id, level, pair, use_ft_labels=True
                        )
                        continue

                    cov_key = "res" if level == "residue" else "poly"
                    fmat = self._get_indexed_matrix(force_cov.get(cov_key, []), gi)
                    tmat = self._get_indexed_matrix(torque_cov.get(cov_key, []), gi)

                    pair = self._compute_force_torque_entropy(
                        ve=ve,
                        temp=temp,
                        fmat=fmat,
                        tmat=tmat,
                        flexible=flexible_res,
                        highest=highest,
                    )
                    self._store_results(results, group_id, level, pair)
                    self._log_molecule_level_results(
                        reporter, group_id, level, pair, use_ft_labels=False
                    )
                    continue

                raise ValueError(f"Unknown level: {level}")

        shared_data["vibrational_entropy"] = results
        return {"vibrational_entropy": results}

    def _build_entropy_engine(
        self, shared_data: Mapping[str, Any]
    ) -> VibrationalEntropy:
        """Construct the vibrational entropy engine used for calculations.

        Args:
            shared_data: Read-only mapping containing a "run_manager" entry.

        Returns:
            A configured VibrationalEntropy instance.
        """
        return VibrationalEntropy(
            run_manager=shared_data["run_manager"],
        )

    def _get_group_id_to_index(self, shared_data: Mapping[str, Any]) -> dict[int, int]:
        """Return a mapping from group_id to contiguous index used by covariance lists.

        If a precomputed mapping is provided under "group_id_to_index", it is used.
        Otherwise, the mapping is derived from the insertion order of "groups".

        Args:
            shared_data: Read-only mapping containing "groups" and optionally
                "group_id_to_index".

        Returns:
            Dictionary mapping each group_id to an integer index.
        """
        gid2i = shared_data.get("group_id_to_index")
        if isinstance(gid2i, dict) and gid2i:
            return gid2i
        groups = shared_data["groups"]
        return {gid: i for i, gid in enumerate(sorted(groups.keys()))}

    def _get_ua_frame_counts(self, shared_data: Mapping[str, Any]) -> dict[CovKey, int]:
        """Extract per-(group,residue) frame counts for united-atom covariances.

        Args:
            shared_data: Read-only mapping which may contain nested frame count data
                under shared_data["frame_counts"]["ua"].

        Returns:
            A dict keyed by (group_id, residue_id) containing frame counts. Returns
            an empty dict if not present or not well-formed.
        """
        counts = shared_data.get("frame_counts", {})
        if isinstance(counts, dict):
            ua_counts = counts.get("ua", {})
            if isinstance(ua_counts, dict):
                return ua_counts
        return {}

    def _compute_united_atom_entropy(
        self,
        *,
        ve: VibrationalEntropy,
        temp: float,
        group_id: int,
        residues: Any,
        force_ua: Mapping[CovKey, Any],
        torque_ua: Mapping[CovKey, Any],
        flexible_ua: Any,
        ua_frame_counts: Mapping[CovKey, int],
        reporter: Any | None,
        n_frames_default: int,
        highest: bool,
    ) -> EntropyPair:
        """Compute total united-atom vibrational entropy for a group's residues.

        Iterates over residues, looks up per-residue force and torque covariance
        matrices keyed by (group_id, residue_index), computes entropy contributions,
        accumulates totals, and optionally reports per-residue values.

        Args:
            ve: VibrationalEntropy calculation engine.
            temp: Temperature (K) for entropy calculation.
            group_id: Identifier for the group being processed.
            residues: Residue container/sequence for the representative molecule.
            force_ua: Mapping from (group_id, residue_id) to force covariance matrix.
            torque_ua: Mapping from (group_id, residue_id) to torque covariance matrix.
            flexible: Data about number of flexible dihedrals
            ua_frame_counts: Mapping from (group_id, residue_id) to frame counts.
            reporter: Optional reporter object supporting add_residue_data calls.
            n_frames_default: Fallback frame count if per-residue count missing.
            highest: Whether this computation is at the highest requested level.

        Returns:
            EntropyPair with summed translational and rotational entropy across residues
        """
        s_trans_total = 0.0
        s_rot_total = 0.0

        for res_id, res in enumerate(residues):
            key = (group_id, res_id)
            fmat = force_ua.get(key)
            tmat = torque_ua.get(key)
            flexible = flexible_ua.get(key)

            pair = self._compute_force_torque_entropy(
                ve=ve,
                temp=temp,
                fmat=fmat,
                tmat=tmat,
                flexible=flexible,
                highest=highest,
            )

            s_trans_total += pair.trans
            s_rot_total += pair.rot

            if reporter is not None:
                frame_count = ua_frame_counts.get(key, int(n_frames_default or 0))
                reporter.add_residue_data(
                    group_id=group_id,
                    resname=getattr(res, "resname", "UNK"),
                    level="united_atom",
                    entropy_type="Transvibrational",
                    frame_count=frame_count,
                    value=pair.trans,
                )
                reporter.add_residue_data(
                    group_id=group_id,
                    resname=getattr(res, "resname", "UNK"),
                    level="united_atom",
                    entropy_type="Rovibrational",
                    frame_count=frame_count,
                    value=pair.rot,
                )

        return EntropyPair(trans=float(s_trans_total), rot=float(s_rot_total))

    def _compute_force_torque_entropy(
        self,
        *,
        ve: VibrationalEntropy,
        temp: float,
        fmat: Any,
        tmat: Any,
        flexible: int,
        highest: bool,
    ) -> EntropyPair:
        """Compute vibrational entropy from separate force and torque covariances.

        Matrices are filtered to remove (near-)zero rows/columns before computation.
        If either matrix is missing or becomes empty after filtering, returns zeros.

        Args:
            ve: VibrationalEntropy calculation engine.
            temp: Temperature (K) for entropy calculation.
            fmat: Force covariance matrix (array-like) or None.
            tmat: Torque covariance matrix (array-like) or None.
            highest: Whether this computation is at the highest requested level.

        Returns:
            EntropyPair containing translational entropy (from force covariance) and
            rotational entropy (from torque covariance).
        """
        if fmat is None or tmat is None:
            return EntropyPair(trans=0.0, rot=0.0)

        f = self._mat_ops.filter_zero_rows_columns(
            np.asarray(fmat), atol=self._zero_atol
        )
        t = self._mat_ops.filter_zero_rows_columns(
            np.asarray(tmat), atol=self._zero_atol
        )

        if f.size == 0 or t.size == 0:
            return EntropyPair(trans=0.0, rot=0.0)

        s_trans = ve.vibrational_entropy_calculation(
            f, "force", temp, highest_level=highest, flexible=flexible
        )
        s_rot = ve.vibrational_entropy_calculation(
            t, "torque", temp, highest_level=highest, flexible=flexible
        )
        return EntropyPair(trans=float(s_trans), rot=float(s_rot))

    def _compute_ft_entropy(
        self,
        *,
        ve: VibrationalEntropy,
        temp: float,
        ftmat: Any,
        flexible: int,
    ) -> EntropyPair:
        """Compute vibrational entropy from a combined force-torque covariance matrix.

        The combined covariance matrix is filtered to remove (near-)zero rows/columns
        before computation. If missing or empty after filtering, returns zeros.

        Args:
            ve: VibrationalEntropy calculation engine.
            temp: Temperature (K) for entropy calculation.
            ftmat: Combined force-torque covariance matrix (array-like) or None.

        Returns:
            EntropyPair containing translational and rotational entropy values derived
            from the combined covariance matrix.
        """
        if ftmat is None:
            return EntropyPair(trans=0.0, rot=0.0)

        ft = self._mat_ops.filter_zero_rows_columns(
            np.asarray(ftmat), atol=self._zero_atol
        )
        if ft.size == 0:
            return EntropyPair(trans=0.0, rot=0.0)

        s_trans = ve.vibrational_entropy_calculation(
            ft, "forcetorqueTRANS", temp, highest_level=True, flexible=flexible
        )
        s_rot = ve.vibrational_entropy_calculation(
            ft, "forcetorqueROT", temp, highest_level=True, flexible=flexible
        )
        return EntropyPair(trans=float(s_trans), rot=float(s_rot))

    @staticmethod
    def _store_results(
        results: dict[int, dict[str, dict[str, float]]],
        group_id: int,
        level: str,
        pair: EntropyPair,
    ) -> None:
        """Store entropy results for a group/level into the results structure.

        Args:
            results: Nested results dict indexed by group_id then level.
            group_id: Group identifier to store under.
            level: Hierarchy level name (e.g., "united_atom", "residue", "polymer").
            pair: EntropyPair containing translational and rotational values.
        """
        results[group_id][level] = {"trans": pair.trans, "rot": pair.rot}

    @staticmethod
    def _log_molecule_level_results(
        reporter: Any | None,
        group_id: int,
        level: str,
        pair: EntropyPair,
        *,
        use_ft_labels: bool,
    ) -> None:
        """Log molecule-level entropy results to the reporter, if available.

        Args:
            reporter: Optional reporter object supporting add_results_data calls.
            group_id: Group identifier being reported.
            level: Hierarchy level name being reported.
            pair: EntropyPair containing translational and rotational values.
            use_ft_labels: Whether to use FT-specific labels for the entropy types.
        """
        if reporter is None:
            return

        if use_ft_labels:
            reporter.add_results_data(
                group_id, level, "FTmat-Transvibrational", pair.trans
            )
            reporter.add_results_data(group_id, level, "FTmat-Rovibrational", pair.rot)
            return

        reporter.add_results_data(group_id, level, "Transvibrational", pair.trans)
        reporter.add_results_data(group_id, level, "Rovibrational", pair.rot)

    @staticmethod
    def _get_indexed_matrix(mats: Any, index: int) -> Any:
        """Safely retrieve mats[index] if mats is indexable and index is in range.

        Args:
            mats: Indexable container of matrices (e.g., list/tuple) or other object.
            index: Desired index.

        Returns:
            The matrix at the given index if available; otherwise None.
        """
        try:
            return mats[index] if index < len(mats) else None
        except TypeError:
            return None
