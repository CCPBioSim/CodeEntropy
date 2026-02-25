"""Node for computing vibrational entropy from covariance matrices."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from CodeEntropy.entropy.vibrational import VibrationalEntropy
from CodeEntropy.levels.linalg import MatrixUtils

logger = logging.getLogger(__name__)


GroupId = int
ResidueId = int
CovKey = Tuple[GroupId, ResidueId]


@dataclass(frozen=True)
class EntropyPair:
    """Container for paired translational and rotational entropy values."""

    trans: float
    rot: float


class VibrationalEntropyNode:
    """Compute vibrational entropy from force/torque (and optional FT) covariances."""

    def __init__(self) -> None:
        self._mat_ops = MatrixUtils()
        self._zero_atol = 1e-8

    def run(self, shared_data: MutableMapping[str, Any], **_: Any) -> Dict[str, Any]:
        ve = self._build_entropy_engine(shared_data)
        temp = shared_data["args"].temperature

        groups = shared_data["groups"]
        levels = shared_data["levels"]
        fragments = shared_data["reduced_universe"].atoms.fragments

        gid2i = self._get_group_id_to_index(shared_data)

        force_cov = shared_data["force_covariances"]
        torque_cov = shared_data["torque_covariances"]

        combined = bool(getattr(shared_data["args"], "combined_forcetorque", False))
        ft_cov = shared_data.get("forcetorque_covariances") if combined else None

        ua_frame_counts = self._get_ua_frame_counts(shared_data)
        reporter = shared_data.get("reporter")

        results: Dict[int, Dict[str, Dict[str, float]]] = {}

        for group_id, mol_ids in groups.items():
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

                    if combined and highest and ft_cov is not None:
                        ft_key = "res" if level == "residue" else "poly"
                        ftmat = self._get_indexed_matrix(ft_cov.get(ft_key, []), gi)

                        pair = self._compute_ft_entropy(ve=ve, temp=temp, ftmat=ftmat)
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
        return VibrationalEntropy(
            run_manager=shared_data["run_manager"],
        )

    def _get_group_id_to_index(self, shared_data: Mapping[str, Any]) -> Dict[int, int]:
        gid2i = shared_data.get("group_id_to_index")
        if isinstance(gid2i, dict) and gid2i:
            return gid2i
        groups = shared_data["groups"]
        return {gid: i for i, gid in enumerate(groups.keys())}

    def _get_ua_frame_counts(self, shared_data: Mapping[str, Any]) -> Dict[CovKey, int]:
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
        ua_frame_counts: Mapping[CovKey, int],
        reporter: Optional[Any],
        n_frames_default: int,
        highest: bool,
    ) -> EntropyPair:
        s_trans_total = 0.0
        s_rot_total = 0.0

        for res_id, res in enumerate(residues):
            key = (group_id, res_id)
            fmat = force_ua.get(key)
            tmat = torque_ua.get(key)

            pair = self._compute_force_torque_entropy(
                ve=ve,
                temp=temp,
                fmat=fmat,
                tmat=tmat,
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
        highest: bool,
    ) -> EntropyPair:
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
            f, "force", temp, highest_level=highest
        )
        s_rot = ve.vibrational_entropy_calculation(
            t, "torque", temp, highest_level=highest
        )
        return EntropyPair(trans=float(s_trans), rot=float(s_rot))

    def _compute_ft_entropy(
        self,
        *,
        ve: VibrationalEntropy,
        temp: float,
        ftmat: Any,
    ) -> EntropyPair:
        if ftmat is None:
            return EntropyPair(trans=0.0, rot=0.0)

        ft = self._mat_ops.filter_zero_rows_columns(
            np.asarray(ftmat), atol=self._zero_atol
        )
        if ft.size == 0:
            return EntropyPair(trans=0.0, rot=0.0)

        s_trans = ve.vibrational_entropy_calculation(
            ft, "forcetorqueTRANS", temp, highest_level=True
        )
        s_rot = ve.vibrational_entropy_calculation(
            ft, "forcetorqueROT", temp, highest_level=True
        )
        return EntropyPair(trans=float(s_trans), rot=float(s_rot))

    @staticmethod
    def _store_results(
        results: Dict[int, Dict[str, Dict[str, float]]],
        group_id: int,
        level: str,
        pair: EntropyPair,
    ) -> None:
        results[group_id][level] = {"trans": pair.trans, "rot": pair.rot}

    @staticmethod
    def _log_molecule_level_results(
        reporter: Optional[Any],
        group_id: int,
        level: str,
        pair: EntropyPair,
        *,
        use_ft_labels: bool,
    ) -> None:
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
        try:
            return mats[index] if index < len(mats) else None
        except TypeError:
            return None
