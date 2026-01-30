# CodeEntropy/levels/force_torque_manager.py

import logging
from typing import List, Tuple

import numpy as np

from CodeEntropy.levels.matrix_operations import MatrixOperations

logger = logging.getLogger(__name__)


class ForceTorqueManager:
    """
    Frame-local force/torque -> covariance builder.

    """

    def __init__(self):
        self._mops = MatrixOperations()

    def get_weighted_forces(
        self,
        bead,
        trans_axes: np.ndarray,
        highest_level: bool,
        force_partitioning: float,
    ) -> np.ndarray:
        forces_trans = np.zeros((3,), dtype=float)

        # atom.force is in the current Universe timestep already
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
        force_partitioning: float,
    ) -> np.ndarray:
        torques = np.zeros((3,), dtype=float)

        com = bead.center_of_mass()
        for atom in bead.atoms:
            r = atom.position - com
            r = np.matmul(rot_axes, r)

            f = np.matmul(rot_axes, atom.force)
            f = force_partitioning * f

            torques += np.cross(r, f)

        # MOI weighting
        eigvals, _ = np.linalg.eig(bead.moment_of_inertia())
        moi = sorted(np.real(eigvals), reverse=True)

        weighted = np.zeros((3,), dtype=float)
        for k in range(3):
            if np.isclose(torques[k], 0.0):
                weighted[k] = 0.0
                continue
            if np.isclose(moi[k], 0.0):
                weighted[k] = 0.0
                logger.warning(
                    "Zero principal moment of inertia; setting torque component to 0."
                )
                continue
            if moi[k] < 0:
                raise ValueError(
                    f"Negative principal moment of inertia encountered: {moi[k]}"
                )
            weighted[k] = torques[k] / np.sqrt(moi[k])

        return weighted

    def compute_frame_covariance(
        self,
        beads: List,
        trans_axes: np.ndarray,
        highest_level: bool,
        force_partitioning: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute full block covariance matrices for this frame:
        - force covariance (3N x 3N)
        - torque covariance (3N x 3N)
        """
        n_beads = len(beads)
        if n_beads == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))

        weighted_forces = [None] * n_beads
        weighted_torques = [None] * n_beads

        for i, bead in enumerate(beads):
            if len(bead.atoms) == 0:
                raise ValueError("AtomGroup is empty (bead has 0 atoms).")

            # rotation axes per bead (principal axes)
            rot_axes = np.real(bead.principal_axes())

            weighted_forces[i] = self.get_weighted_forces(
                bead=bead,
                trans_axes=trans_axes,
                highest_level=highest_level,
                force_partitioning=force_partitioning,
            )
            weighted_torques[i] = self.get_weighted_torques(
                bead=bead,
                rot_axes=rot_axes,
                force_partitioning=force_partitioning,
            )

        # build block matrices
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
