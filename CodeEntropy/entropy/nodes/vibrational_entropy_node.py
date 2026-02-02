# CodeEntropy/entropy/nodes/vibrational_entropy_node.py

import logging
from typing import Any, Dict

import numpy as np

from CodeEntropy.entropy.vibrational_entropy import VibrationalEntropy
from CodeEntropy.levels.matrix_operations import MatrixOperations

logger = logging.getLogger(__name__)


class VibrationalEntropyNode:
    """
    Computes vibrational entropy from force and torque covariance matrices.

    Expects LevelDAG to have produced:
      shared_data["force_covariances"]  : {"ua": dict, "res": list, "poly": list}
      shared_data["torque_covariances"] : {"ua": dict, "res": list, "poly": list}
      shared_data["levels"], shared_data["groups"]

    Also optionally uses:
      shared_data["frame_counts"] for residue table counts
    """

    def __init__(self):
        self._mat_ops = MatrixOperations()

    def run(self, shared_data: Dict[str, Any], **_kwargs):
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
        levels = shared_data["levels"]
        groups = shared_data["groups"]

        force_cov = shared_data["force_covariances"]
        torque_cov = shared_data["torque_covariances"]
        frame_counts = shared_data.get("frame_counts", None)

        fragments = universe.atoms.fragments

        vib_results: Dict[int, Dict[str, Dict[str, float]]] = {}

        for group_id, mol_ids in groups.items():
            mol_id = mol_ids[0]
            level_list = levels[mol_id]
            mol = fragments[mol_id]

            vib_results[group_id] = {}

            for level in level_list:
                if level == "united_atom":
                    S_trans_total = 0.0
                    S_rot_total = 0.0

                    # UA is stored per (group_id, res_id)
                    for (gid, res_id), fmat in force_cov["ua"].items():
                        if gid != group_id:
                            continue

                        tmat = torque_cov["ua"].get((gid, res_id))
                        if tmat is None:
                            continue

                        fmat_arr = self._mat_ops.filter_zero_rows_columns(
                            np.array(fmat)
                        )
                        tmat_arr = self._mat_ops.filter_zero_rows_columns(
                            np.array(tmat)
                        )

                        S_trans = ve.vibrational_entropy_calculation(
                            fmat_arr, "force", temp, highest_level=False
                        )
                        S_rot = ve.vibrational_entropy_calculation(
                            tmat_arr, "torque", temp, highest_level=False
                        )

                        S_trans_total += S_trans
                        S_rot_total += S_rot

                        if data_logger is not None:
                            if res_id < len(mol.residues):
                                resname = mol.residues[res_id].resname
                            else:
                                resname = f"RES{res_id}"

                            count_val = None
                            if frame_counts is not None:
                                count_val = frame_counts["ua"].get(
                                    (group_id, res_id), None
                                )

                            data_logger.add_residue_data(
                                group_id=group_id,
                                resname=resname,
                                level="united_atom",
                                entropy_type="Transvibrational",
                                frame_count=count_val,
                                value=S_trans,
                            )
                            data_logger.add_residue_data(
                                group_id=group_id,
                                resname=resname,
                                level="united_atom",
                                entropy_type="Rovibrational",
                                frame_count=count_val,
                                value=S_rot,
                            )

                    vib_results[group_id][level] = {
                        "trans": S_trans_total,
                        "rot": S_rot_total,
                    }

                    if data_logger is not None:
                        data_logger.add_results_data(
                            group_id, level, "Transvibrational", S_trans_total
                        )
                        data_logger.add_results_data(
                            group_id, level, "Rovibrational", S_rot_total
                        )

                elif level == "residue":
                    fmat = force_cov["res"][group_id]
                    tmat = torque_cov["res"][group_id]

                    if fmat is None or tmat is None:
                        S_trans, S_rot = 0.0, 0.0
                    else:
                        fmat_arr = self._mat_ops.filter_zero_rows_columns(
                            np.array(fmat)
                        )
                        tmat_arr = self._mat_ops.filter_zero_rows_columns(
                            np.array(tmat)
                        )

                        S_trans = ve.vibrational_entropy_calculation(
                            fmat_arr, "force", temp, highest_level=False
                        )
                        S_rot = ve.vibrational_entropy_calculation(
                            tmat_arr, "torque", temp, highest_level=False
                        )

                    vib_results[group_id][level] = {"trans": S_trans, "rot": S_rot}

                    if data_logger is not None:
                        data_logger.add_results_data(
                            group_id, level, "Transvibrational", S_trans
                        )
                        data_logger.add_results_data(
                            group_id, level, "Rovibrational", S_rot
                        )

                elif level == "polymer":
                    fmat = force_cov["poly"][group_id]
                    tmat = torque_cov["poly"][group_id]

                    if fmat is None or tmat is None:
                        S_trans, S_rot = 0.0, 0.0
                    else:
                        fmat_arr = self._mat_ops.filter_zero_rows_columns(
                            np.array(fmat)
                        )
                        tmat_arr = self._mat_ops.filter_zero_rows_columns(
                            np.array(tmat)
                        )

                        S_trans = ve.vibrational_entropy_calculation(
                            fmat_arr, "force", temp, highest_level=True
                        )
                        S_rot = ve.vibrational_entropy_calculation(
                            tmat_arr, "torque", temp, highest_level=True
                        )

                    vib_results[group_id][level] = {"trans": S_trans, "rot": S_rot}

                    if data_logger is not None:
                        data_logger.add_results_data(
                            group_id, level, "Transvibrational", S_trans
                        )
                        data_logger.add_results_data(
                            group_id, level, "Rovibrational", S_rot
                        )

                else:
                    raise ValueError(f"Unknown level: {level}")

        logger.info("[VibrationalEntropyNode] Done")
        return {"vibrational_entropy": vib_results}
