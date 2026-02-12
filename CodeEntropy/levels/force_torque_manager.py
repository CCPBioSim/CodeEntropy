import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ForceTorqueManager:
    def __init__(self):
        pass

    def get_weighted_forces(
        self,
        bead,
        trans_axes: np.ndarray,
        highest_level: bool,
        force_partitioning: float,
    ) -> np.ndarray:
        """
        Procedural-equivalent translational force:
          sum( trans_axes @ atom.force ) over bead atoms
          optionally scale by force_partitioning if highest_level
          divide by sqrt(bead mass)
        """
        forces_trans = np.zeros((3,), dtype=float)

        for atom in bead.atoms:
            forces_local = np.matmul(trans_axes, atom.force)
            forces_trans += forces_local

        if highest_level:
            forces_trans = force_partitioning * forces_trans

        mass = bead.total_mass()
        if mass <= 0:
            raise ValueError(f"Invalid mass value: {mass}")

        return forces_trans / np.sqrt(mass)

    def get_weighted_torques(
        self,
        bead,
        rot_axes: np.ndarray,
        center: np.ndarray,
        force_partitioning: float,
        moment_of_inertia: np.ndarray,
        axes_manager: Optional[Any],
        box: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Procedural-equivalent rotational torque:
          coords = axes_manager.get_vector(center, bead.positions, box)  (PBC)
          rotate coords/forces into rot_axes
          scale forces by force_partitioning
          torque = sum( cross(r, f) )
          divide componentwise by sqrt(principal moments)
        """
        if (
            axes_manager is not None
            and hasattr(axes_manager, "get_vector")
            and box is not None
        ):
            translated = axes_manager.get_vector(center, bead.positions, box)
        else:
            translated = bead.positions - center

        rotated_coords = np.tensordot(translated, rot_axes.T, axes=1)
        rotated_forces = np.tensordot(bead.forces, rot_axes.T, axes=1)

        rotated_forces *= force_partitioning

        torques = np.cross(rotated_coords, rotated_forces)
        torques = np.sum(torques, axis=0)

        moi = np.asarray(moment_of_inertia)
        moi = np.real_if_close(moi, tol=1000)
        moi = np.asarray(moi, dtype=float).reshape(-1)
        if moi.size != 3:
            raise ValueError(f"moment_of_inertia must be (3,), got {moi.shape}")

        weighted = np.zeros((3,), dtype=float)
        for d in range(3):
            if np.isclose(torques[d], 0.0):
                continue
            if moi[d] <= 0.0:
                continue
            weighted[d] = torques[d] / np.sqrt(moi[d])

        return weighted

    @staticmethod
    def _outer_second_moment(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Procedural-style per-frame "covariance" (actually second moment):
          If x is concatenated (3N,) vector of bead forces/torques,
          return outer(x, x) -> (3N,3N)
        """
        if not vectors:
            return np.zeros((0, 0), dtype=float)

        flat = np.concatenate(
            [
                np.asarray(v, dtype=float).reshape(
                    3,
                )
                for v in vectors
            ],
            axis=0,
        )
        return np.outer(flat, flat)

    def compute_frame_covariance(
        self,
        force_vecs: List[np.ndarray],
        torque_vecs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        F = self._outer_second_moment(force_vecs)
        T = self._outer_second_moment(torque_vecs)
        return F, T
