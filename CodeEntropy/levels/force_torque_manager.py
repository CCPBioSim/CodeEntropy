# CodeEntropy/levels/force_torque_manager.py

import logging
from typing import List, Optional, Tuple

import numpy as np

from CodeEntropy.levels.matrix_operations import MatrixOperations

logger = logging.getLogger(__name__)


class ForceTorqueManager:
    """
    Frame-local force/torque -> covariance builder.

    This class is intentionally "physics/math heavy" (same as the procedural code):
    DAG nodes orchestrate; this computes the actual per-frame matrices.
    """

    def __init__(self):
        self._mops = MatrixOperations()

    def get_weighted_forces(
        self,
        data_container,
        bead,
        trans_axes: np.ndarray,
        highest_level: bool,
        force_partitioning: float,
    ) -> np.ndarray:
        """
        Match procedural semantics:
          - rotate each atom force into trans_axes frame
          - sum over bead
          - apply force_partitioning ONLY at highest_level
          - mass-weight by sqrt(total_mass)
        """
        forces_trans = np.zeros((3,), dtype=float)

        for atom in bead.atoms:
            forces_local = np.matmul(trans_axes, atom.force)
            forces_trans += forces_local

        if highest_level:
            forces_trans = force_partitioning * forces_trans

        mass = bead.total_mass()
        if mass <= 0:
            raise ValueError(f"Invalid bead mass {mass}; cannot weight force.")

        return forces_trans / np.sqrt(mass)

    def get_weighted_torques(
        self,
        bead,
        rot_axes: np.ndarray,
        center: np.ndarray,
        force_partitioning: float,
        moment_of_inertia: np.ndarray,
        axes_manager,
        dimensions: np.ndarray,
    ) -> np.ndarray:
        """
        Match procedural semantics:
          - compute r vectors with PBC wrapping using axes_manager.get_vector
          - rotate r and f into rot_axes frame
          - apply force_partitioning to forces (procedural does this for torques)
          - torque = sum cross(r,f)
          - MOI-weight component-wise by sqrt(moi_k)
        """
        torques = np.zeros((3,), dtype=float)

        for atom in bead.atoms:
            # PBC-safe vector from center -> atom.position
            r = axes_manager.get_vector(center, atom.position, dimensions)
            r = np.matmul(rot_axes, r)

            f = np.matmul(rot_axes, atom.force)
            f = force_partitioning * f

            torques += np.cross(r, f)

        moi = np.array(moment_of_inertia, dtype=float)

        weighted = np.zeros((3,), dtype=float)
        for k in range(3):
            if np.isclose(torques[k], 0.0) or np.isclose(moi[k], 0.0):
                weighted[k] = 0.0
                continue
            if moi[k] < 0:
                raise ValueError(
                    f"Negative principal moment of inertia encountered: {moi[k]}"
                )
            weighted[k] = torques[k] / np.sqrt(moi[k])

        return weighted

    def compute_frame_covariance(
        self,
        data_container,
        beads: List,
        trans_axes: np.ndarray,
        highest_level: bool,
        force_partitioning: float,
        rot_axes_list: Optional[List[np.ndarray]] = None,
        centers: Optional[List[np.ndarray]] = None,
        mois: Optional[List[np.ndarray]] = None,
        axes_manager=None,
        dimensions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-frame block covariance matrices:
          - force covariance (3N x 3N)
          - torque covariance (3N x 3N)

        If rot_axes_list/centers/mois are supplied, they are used (this is how we
        match the procedural AxesManager paths exactly).
        Otherwise, caller should provide vanilla axes/mois/centers already or we error.
        """
        n_beads = len(beads)
        if n_beads == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))

        weighted_forces = [None] * n_beads
        weighted_torques = [None] * n_beads

        for i, bead in enumerate(beads):
            if len(bead.atoms) == 0:
                raise ValueError("AtomGroup is empty (bead has 0 atoms).")

            rot_axes = rot_axes_list[i] if rot_axes_list is not None else None
            center = centers[i] if centers is not None else None
            moi = mois[i] if mois is not None else None

            if rot_axes is None or center is None or moi is None:
                raise ValueError(
                    "rot_axes/center/moment_of_inertia must be provided per bead."
                )

            weighted_forces[i] = self.get_weighted_forces(
                data_container=data_container,
                bead=bead,
                trans_axes=trans_axes,
                highest_level=highest_level,
                force_partitioning=force_partitioning,
            )

            weighted_torques[i] = self.get_weighted_torques(
                bead=bead,
                rot_axes=rot_axes,
                center=center,
                force_partitioning=force_partitioning,
                moment_of_inertia=moi,
                axes_manager=axes_manager,
                dimensions=dimensions,
            )

        f_blocks = [[None] * n_beads for _ in range(n_beads)]
        t_blocks = [[None] * n_beads for _ in range(n_beads)]

        for i in range(n_beads):
            for j in range(i, n_beads):
                f_sub = self._mops.create_submatrix(
                    weighted_forces[i], weighted_forces[j]
                )
                t_sub = self._mops.create_submatrix(
                    weighted_torques[i], weighted_torques[j]
                )

                f_blocks[i][j] = f_sub
                f_blocks[j][i] = f_sub.T
                t_blocks[i][j] = t_sub
                t_blocks[j][i] = t_sub.T

        F = np.block([[f_blocks[i][j] for j in range(n_beads)] for i in range(n_beads)])
        T = np.block([[t_blocks[i][j] for j in range(n_beads)] for i in range(n_beads)])

        return F, T
