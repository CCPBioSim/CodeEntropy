"""Force/torque weighting and per-frame second-moment construction.

This module provides utilities for transforming atomic forces into bead-level
generalized forces (translation) and torques (rotation), and for assembling
per-frame second-moment matrices used downstream in entropy calculations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Vector3 = np.ndarray
Matrix = np.ndarray


@dataclass(frozen=True)
class TorqueInputs:
    """Container for torque computation inputs.

    Attributes:
        rot_axes: Rotation matrix mapping lab-frame vectors into the bead frame,
            shape (3, 3).
        center: Reference center for torque arm vectors, shape (3,).
        force_partitioning: Scaling factor applied to forces before torque
            accumulation.
        moment_of_inertia: Principal moments (aligned with rot_axes), shape (3,).
        axes_manager: Optional object that provides:
            get_vector(center, positions, box) -> displacement vectors (PBC-aware).
        box: Optional periodic box passed to axes_manager.get_vector.
    """

    rot_axes: Matrix
    center: Vector3
    force_partitioning: float
    moment_of_inertia: Vector3
    axes_manager: Optional[Any] = None
    box: Optional[np.ndarray] = None


class ForceTorqueManager:
    """Computes weighted generalized forces/torques and per-frame second moments."""

    def get_weighted_forces(
        self,
        bead: Any,
        trans_axes: Matrix,
        highest_level: bool,
        force_partitioning: float,
    ) -> Vector3:
        """Compute a mass-weighted translational generalized force.

        Args:
            bead: MDAnalysis AtomGroup-like bead with .atoms and .total_mass().
                Each atom must provide .force (shape (3,)).
            trans_axes: Transform matrix for translational forces, shape (3, 3).
            highest_level: If True, apply force_partitioning scaling.
            force_partitioning: Scaling factor applied when highest_level is True.

        Returns:
            Mass-weighted generalized force vector, shape (3,).

        Raises:
            ValueError: If mass is non-positive or trans_axes shape is invalid.
        """
        return self._compute_weighted_force(
            bead=bead,
            trans_axes=trans_axes,
            apply_partitioning=highest_level,
            force_partitioning=force_partitioning,
        )

    def get_weighted_torques(
        self,
        bead: Any,
        rot_axes: Matrix,
        center: Vector3,
        force_partitioning: float,
        moment_of_inertia: Vector3,
        axes_manager: Optional[Any],
        box: Optional[np.ndarray],
    ) -> Vector3:
        """Compute a moment-weighted generalized torque.

        Args:
            bead: MDAnalysis AtomGroup-like bead with .positions and .forces (N,3).
            rot_axes: Rotation matrix into bead frame, shape (3,3).
            center: Reference center for displacement vectors, shape (3,).
            force_partitioning: Scaling factor applied to forces before torque sum.
            moment_of_inertia: Principal moments aligned with rot_axes, shape (3,).
            axes_manager: Optional PBC displacement provider.
            box: Periodic box passed to axes_manager when used.

        Returns:
            Weighted torque vector, shape (3,).

        Raises:
            ValueError: If shapes are invalid.
        """
        inputs = TorqueInputs(
            rot_axes=np.asarray(rot_axes, dtype=float),
            center=np.asarray(center, dtype=float).reshape(3),
            force_partitioning=float(force_partitioning),
            moment_of_inertia=np.asarray(moment_of_inertia),
            axes_manager=axes_manager,
            box=box,
        )
        return self._compute_weighted_torque(bead=bead, inputs=inputs)

    def compute_frame_covariance(
        self,
        force_vecs: Sequence[Vector3],
        torque_vecs: Sequence[Vector3],
    ) -> Tuple[Matrix, Matrix]:
        """Compute per-frame second-moment matrices for force/torque vectors.

        Note:
            This returns outer(x, x) where x is the concatenation of all bead
            vectors in the frame.

        Args:
            force_vecs: Sequence of per-bead force vectors (3,).
            torque_vecs: Sequence of per-bead torque vectors (3,).

        Returns:
            Tuple (F, T) where each is a (3N, 3N) second-moment matrix.
        """
        return self._compute_frame_second_moments(force_vecs, torque_vecs)

    def _compute_weighted_force(
        self,
        bead: Any,
        trans_axes: Matrix,
        *,
        apply_partitioning: bool,
        force_partitioning: float,
    ) -> Vector3:
        """Implementation of translational generalized force computation."""
        trans_axes = np.asarray(trans_axes, dtype=float)
        if trans_axes.shape != (3, 3):
            raise ValueError(f"trans_axes must be (3,3), got {trans_axes.shape}")

        forces_trans = np.zeros((3,), dtype=float)
        for atom in bead.atoms:
            forces_trans += trans_axes @ np.asarray(atom.force, dtype=float)

        if apply_partitioning:
            forces_trans *= float(force_partitioning)

        mass = float(bead.total_mass())
        if mass <= 0.0:
            raise ValueError(f"Invalid bead mass: {mass}. Mass must be positive.")

        return forces_trans / np.sqrt(mass)

    def _compute_weighted_torque(self, bead: Any, inputs: TorqueInputs) -> Vector3:
        """Implementation of rotational generalized torque computation."""
        rot_axes = np.asarray(inputs.rot_axes, dtype=float)
        if rot_axes.shape != (3, 3):
            raise ValueError(f"rot_axes must be (3,3), got {rot_axes.shape}")

        moi = np.asarray(inputs.moment_of_inertia)
        moi = np.real_if_close(moi, tol=1000)
        moi = np.asarray(moi, dtype=float).reshape(-1)
        if moi.size != 3:
            raise ValueError(f"moment_of_inertia must be (3,), got {moi.shape}")

        translated = self._displacements_relative_to_center(
            center=np.asarray(inputs.center, dtype=float).reshape(3),
            positions=np.asarray(bead.positions, dtype=float),
            axes_manager=inputs.axes_manager,
            box=inputs.box,
        )

        rotated_coords = np.tensordot(translated, rot_axes.T, axes=1)
        rotated_forces = np.tensordot(
            np.asarray(bead.forces, dtype=float), rot_axes.T, axes=1
        )
        rotated_forces *= float(inputs.force_partitioning)

        torques = np.sum(np.cross(rotated_coords, rotated_forces), axis=0)

        weighted = np.zeros((3,), dtype=float)
        for d in range(3):
            if np.isclose(torques[d], 0.0):
                continue
            if moi[d] <= 0.0:
                continue
            weighted[d] = torques[d] / np.sqrt(moi[d])

        return weighted

    def _compute_frame_second_moments(
        self,
        force_vectors: Sequence[Vector3],
        torque_vectors: Sequence[Vector3],
    ) -> Tuple[Matrix, Matrix]:
        """Build outer products for concatenated force/torque vectors."""
        f = self._outer_second_moment(force_vectors)
        t = self._outer_second_moment(torque_vectors)
        return f, t

    @staticmethod
    def _displacements_relative_to_center(
        *,
        center: Vector3,
        positions: np.ndarray,
        axes_manager: Optional[Any],
        box: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute displacement vectors from center to positions (optionally PBC-aware).
        """
        if (
            axes_manager is not None
            and hasattr(axes_manager, "get_vector")
            and box is not None
        ):
            return axes_manager.get_vector(center, positions, box)
        return positions - center

    @staticmethod
    def _outer_second_moment(vectors: Sequence[Vector3]) -> Matrix:
        """Compute outer(flat, flat) for concatenated 3-vectors.

        Args:
            vectors: Sequence of vectors of shape (3,).

        Returns:
            Second-moment matrix with shape (3N, 3N). Returns (0,0) if empty.

        Raises:
            ValueError: If any vector is not length 3.
        """
        if not vectors:
            return np.zeros((0, 0), dtype=float)

        parts = []
        for v in vectors:
            arr = np.asarray(v, dtype=float).reshape(-1)
            if arr.size != 3:
                raise ValueError(
                    f"Expected vector of length 3, got shape {np.asarray(v).shape}"
                )
            parts.append(arr)

        flat = np.concatenate(parts, axis=0)
        return np.outer(flat, flat)
