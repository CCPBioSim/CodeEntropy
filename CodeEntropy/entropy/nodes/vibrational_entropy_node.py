import logging
from typing import Any, Dict

import numpy as np

from CodeEntropy.entropy.vibrational_entropy import VibrationalEntropy
from CodeEntropy.levels.matrix_operations import MatrixOperations

logger = logging.getLogger(__name__)


def _build_gid2i(shared_data: Dict[str, Any]) -> Dict[int, int]:
    gid2i = shared_data.get("group_id_to_index")
    if isinstance(gid2i, dict) and gid2i:
        return gid2i
    groups = shared_data["groups"]
    return {gid: i for i, gid in enumerate(groups.keys())}


class VibrationalEntropyNode:
    def __init__(self):
        self._mat_ops = MatrixOperations()

    def run(self, shared_data: Dict[str, Any], **_kwargs) -> Dict[str, Any]:
        run_manager = shared_data["run_manager"]
        args = shared_data["args"]
        universe = shared_data["reduced_universe"]
        data_logger = shared_data.get("data_logger")
        group_molecules = shared_data.get("group_molecules")

        ve = VibrationalEntropy(
            run_manager=run_manager,
            args=args,
            universe=universe,
            data_logger=data_logger,
            group_molecules=group_molecules,
        )

        temp = args.temperature
        groups = shared_data["groups"]
        levels = shared_data["levels"]

        force_cov = shared_data["force_covariances"]
        torque_cov = shared_data["torque_covariances"]

        combined = bool(getattr(args, "combined_forcetorque", False))
        ft_cov = shared_data.get("forcetorque_covariances") if combined else None

        counts = shared_data.get("frame_counts", {})
        ua_counts = counts.get("ua", {}) if isinstance(counts, dict) else {}

        gid2i = _build_gid2i(shared_data)
        fragments = universe.atoms.fragments

        vib_results: Dict[int, Dict[str, Dict[str, float]]] = {}

        for group_id, mol_ids in groups.items():
            mol_id = mol_ids[0]
            mol = fragments[mol_id]
            level_list = levels[mol_id]

            vib_results[group_id] = {}

            for level in level_list:
                # highest = level == level_list[-1]

                if level == "united_atom":
                    S_trans = 0.0
                    S_rot = 0.0

                    for res_id, res in enumerate(mol.residues):
                        key = (group_id, res_id)

                        fmat = force_cov["ua"].get(key)
                        tmat = torque_cov["ua"].get(key)

                        if fmat is None or tmat is None:
                            val_trans, val_rot = 0.0, 0.0
                        else:
                            fmat = self._mat_ops.filter_zero_rows_columns(
                                np.asarray(fmat)
                            )
                            tmat = self._mat_ops.filter_zero_rows_columns(
                                np.asarray(tmat)
                            )

                            val_trans = ve.vibrational_entropy_calculation(
                                fmat, "force", temp, highest_level=False
                            )
                            val_rot = ve.vibrational_entropy_calculation(
                                tmat, "torque", temp, highest_level=False
                            )

                        S_trans += val_trans
                        S_rot += val_rot

                        if data_logger is not None:
                            fc = ua_counts.get(key, shared_data.get("n_frames", 0))
                            data_logger.add_residue_data(
                                group_id=group_id,
                                resname=res.resname,
                                level="united_atom",
                                entropy_type="Transvibrational",
                                frame_count=fc,
                                value=val_trans,
                            )
                            data_logger.add_residue_data(
                                group_id=group_id,
                                resname=res.resname,
                                level="united_atom",
                                entropy_type="Rovibrational",
                                frame_count=fc,
                                value=val_rot,
                            )

                elif level == "residue":
                    gi = gid2i[group_id]
                    fmat = force_cov["res"][gi] if gi < len(force_cov["res"]) else None
                    tmat = (
                        torque_cov["res"][gi] if gi < len(torque_cov["res"]) else None
                    )

                    if fmat is None or tmat is None:
                        S_trans, S_rot = 0.0, 0.0
                    else:
                        fmat = self._mat_ops.filter_zero_rows_columns(np.asarray(fmat))
                        tmat = self._mat_ops.filter_zero_rows_columns(np.asarray(tmat))

                        S_trans = ve.vibrational_entropy_calculation(
                            fmat, "force", temp, highest_level=False
                        )
                        S_rot = ve.vibrational_entropy_calculation(
                            tmat, "torque", temp, highest_level=False
                        )

                elif level == "polymer":
                    gi = gid2i[group_id]

                    if combined and ft_cov is not None:
                        ftmat = ft_cov["poly"][gi] if gi < len(ft_cov["poly"]) else None
                        if ftmat is None:
                            S_trans, S_rot = 0.0, 0.0
                        else:
                            ftmat = np.asarray(ftmat)
                            ftmat = self._mat_ops.filter_zero_rows_columns(ftmat)

                            S_trans = ve.vibrational_entropy_calculation(
                                ftmat, "forcetorqueTRANS", temp, highest_level=True
                            )
                            S_rot = ve.vibrational_entropy_calculation(
                                ftmat, "forcetorqueROT", temp, highest_level=True
                            )
                    else:
                        fmat = (
                            force_cov["poly"][gi]
                            if gi < len(force_cov["poly"])
                            else None
                        )
                        tmat = (
                            torque_cov["poly"][gi]
                            if gi < len(torque_cov["poly"])
                            else None
                        )

                        if fmat is None or tmat is None:
                            S_trans, S_rot = 0.0, 0.0
                        else:
                            fmat = self._mat_ops.filter_zero_rows_columns(
                                np.asarray(fmat)
                            )
                            tmat = self._mat_ops.filter_zero_rows_columns(
                                np.asarray(tmat)
                            )

                            S_trans = ve.vibrational_entropy_calculation(
                                fmat, "force", temp, highest_level=True
                            )
                            S_rot = ve.vibrational_entropy_calculation(
                                tmat, "torque", temp, highest_level=True
                            )

                else:
                    raise ValueError(f"Unknown level: {level}")

                vib_results[group_id][level] = {
                    "trans": float(S_trans),
                    "rot": float(S_rot),
                }

                if data_logger is not None:
                    if level == "polymer" and combined:
                        data_logger.add_results_data(
                            group_id, level, "FTmat-Transvibrational", S_trans
                        )
                        data_logger.add_results_data(
                            group_id, level, "FTmat-Rovibrational", S_rot
                        )
                    else:
                        data_logger.add_results_data(
                            group_id, level, "Transvibrational", S_trans
                        )
                        data_logger.add_results_data(
                            group_id, level, "Rovibrational", S_rot
                        )

        logger.info("[VibrationalEntropyNode] Done")
        return {"vibrational_entropy": vib_results}
