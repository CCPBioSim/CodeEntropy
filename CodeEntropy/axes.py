"""Axes utilities for entropy calculations.

This module contains the :class:`AxesManager`, a geometry-focused helper used by
the entropy pipeline to compute translational and rotational axes, centres, and
moments of inertia at different hierarchy levels (residue / united-atom).
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

logger = logging.getLogger(__name__)


class AxesManager:
    """Compute translation/rotation axes and inertia utilities used by entropy.

    Manages the structural and dynamic levels involved in entropy calculations.
    This includes selecting relevant levels, computing axes for translation and
    rotation, and handling bead-based representations of molecular systems.

    Provides utility methods to:
      - extract averaged positions,
      - convert coordinates to spherical systems (future/legacy scope),
      - compute axes used to rotate forces around,
      - compute custom moments of inertia,
      - manipulate vectors under periodic boundary conditions (PBC),
      - construct custom moment-of-inertia tensors and principal axes.

    Notes:
        This class deliberately does **not**:
          - compute weighted forces/torques (that belongs in ForceTorqueManager),
          - build covariances,
          - compute entropies.
    """

    def __init__(self) -> None:
        """Initialize the AxesManager.

        The original implementation stored a few placeholders for level-related
        data (axes, bead counts, etc.). In the current design, AxesManager is a
        stateless helper, but we keep the attributes for compatibility and
        debugging/extension.

        Attributes:
            data_container: Optional container used by legacy workflows.
            _levels: Optional levels list (legacy/placeholder).
            _trans_axes: Optional cached translation axes (legacy/placeholder).
            _rot_axes: Optional cached rotation axes (legacy/placeholder).
            _number_of_beads: Optional bead count (legacy/placeholder).
        """
        self.data_container = None
        self._levels = None
        self._trans_axes = None
        self._rot_axes = None
        self._number_of_beads = None

    def get_residue_axes(self, data_container, index: int, residue=None):
        """Compute residue-level translational and rotational axes.

        The translational and rotational axes at the residue level.

        - Identify the residue (either provided or selected by `resindex index`).
        - Determine whether the residue is bonded to neighbouring residues
          (previous/next in sequence) using MDAnalysis bonded selections.
        - If there are *no* bonds to other residues:
            * Use a custom principal axes, from a moment-of-inertia (MOI) tensor
              that uses positions of heavy atoms only, but including masses of
              heavy atom + bonded hydrogens.
            * Set translational axes equal to rotational axes (as per the original
              code convention).
        - If bonded to other residues:
            * Use default axes and MOI (MDAnalysis principal axes / inertia).

        Args:
            data_container (MDAnalysis.Universe or AtomGroup):
                Molecule and trajectory data (the fragment/molecule container).
            index (int):
                Residue index (resindex) within `data_container`.
            residue (MDAnalysis.AtomGroup, optional):
                If provided, this residue selection will be used rather than
                selecting again.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - trans_axes: Translational axes array of shape (3, 3).
                - rot_axes: Rotational axes array of shape (3, 3).
                - center: Center of mass array of shape (3,).
                - moment_of_inertia: Principal moments array of shape (3,).

        Raises:
            ValueError:
                If the residue selection is empty.
        """
        # TODO refine selection so that it will work for branched polymers
        index_prev = index - 1
        index_next = index + 1

        if residue is None:
            residue = data_container.select_atoms(f"resindex {index}")
        if len(residue) == 0:
            raise ValueError(f"Empty residue selection for resindex={index}")

        center = residue.atoms.center_of_mass(unwrap=True)
        atom_set = data_container.select_atoms(
            f"(resindex {index_prev} or resindex {index_next}) "
            f"and bonded resid {index}"
        )

        if len(atom_set) == 0:
            # No bonds to other residues.
            # Use a custom principal axes, from a MOI tensor that uses positions of
            # heavy atoms only, but including masses of heavy atom + bonded H.
            uas = residue.select_atoms("mass 2 to 999")
            ua_masses = self.get_UA_masses(residue)
            moi_tensor = self.get_moment_of_inertia_tensor(
                center_of_mass=center,
                positions=uas.positions,
                masses=ua_masses,
                dimensions=data_container.dimensions[:3],
            )
            rot_axes, moment_of_inertia = self.get_custom_principal_axes(moi_tensor)
            trans_axes = rot_axes  # per original convention
        else:
            # If bonded to other residues, use default axes and MOI.
            make_whole(data_container.atoms)
            trans_axes = data_container.atoms.principal_axes()
            rot_axes, moment_of_inertia = self.get_vanilla_axes(residue)
            center = residue.center_of_mass(unwrap=True)

        return trans_axes, rot_axes, center, moment_of_inertia

    def get_UA_axes(self, data_container, index: int):
        """Compute united-atom-level translational and rotational axes.

        The translational and rotational axes at the united-atom level.

        This preserves the original behaviour and its rationale:

        - Translational axes:
            Use the same custom principal-axes approach as residue level:
            compute a custom MOI tensor using heavy-atom coordinates but UA masses
            (heavy + bonded H masses), then compute the principal axes from it.

        - Rotational axes:
            Identify heavy atoms in the residue/molecule of interest and choose
            the `index`-th heavy atom (where index corresponds to the bead index).
            Use bonded topology around that heavy atom to determine UA rotational
            axes (see :meth:`get_bonded_axes`).

        Args:
            data_container (MDAnalysis.Universe or AtomGroup):
                Molecule and trajectory data.
            index (int):
                Bead index (ordinal among heavy atoms).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - trans_axes: Translational axes (3, 3).
                - rot_axes: Rotational axes (3, 3).
                - center: Rotation centre (3,) (heavy atom position).
                - moment_of_inertia: (3,) moments for the UA around rot_axes.

        Raises:
            IndexError:
                If `index` does not correspond to an existing heavy atom.
            ValueError:
                If bonded-axis construction fails.
        """

        index = int(index)  # bead index

        # use the same customPI trans axes as the residue level
        heavy_atoms = data_container.select_atoms("prop mass > 1.1")
        if len(heavy_atoms) > 1:
            UA_masses = self.get_UA_masses(data_container.atoms)
            center = data_container.atoms.center_of_mass(unwrap=True)
            moment_of_inertia_tensor = self.get_moment_of_inertia_tensor(
                center, heavy_atoms.positions, UA_masses, data_container.dimensions[:3]
            )
            trans_axes, _moment_of_inertia = self.get_custom_principal_axes(
                moment_of_inertia_tensor
            )
        else:
            # use standard PA for UA not bonded to anything else
            make_whole(data_container.atoms)
            trans_axes = data_container.atoms.principal_axes()

        # look for heavy atoms in residue of interest
        heavy_atom_indices = []
        for atom in heavy_atoms:
            heavy_atom_indices.append(atom.index)
        # we find the nth heavy atom
        # where n is the bead index
        heavy_atom_index = heavy_atom_indices[index]
        heavy_atom = data_container.select_atoms(f"index {heavy_atom_index}")

        center = heavy_atom.positions[0]
        rot_axes, moment_of_inertia = self.get_bonded_axes(
            system=data_container,
            atom=heavy_atom[0],
            dimensions=data_container.dimensions[:3],
        )
        if rot_axes is None or moment_of_inertia is None:
            raise ValueError("Unable to compute bonded axes for UA bead.")

        logger.debug("Translational Axes: %s", trans_axes)
        logger.debug("Rotational Axes: %s", rot_axes)
        logger.debug("Center: %s", center)
        logger.debug("Moment of Inertia: %s", moment_of_inertia)

        return trans_axes, rot_axes, center, moment_of_inertia

    def get_bonded_axes(self, system, atom, dimensions: np.ndarray):
        r"""Compute UA rotational axes from bonded topology around a heavy atom.

        For a given heavy atom, use its bonded atoms to get the axes for rotating
        forces around. Few cases for choosing united atom axes, which are dependent
        on the bonds to the atom:

        ::

            X -- H = bonded to zero or more light atom/s (case1)

            X -- R = bonded to one heavy atom (case2)

            R -- X -- H = bonded to one heavy and at least one light atom (case3)

            R1 -- X -- R2 = bonded to two heavy atoms (case4)

            R1 -- X -- R2 = bonded to more than two heavy atoms (case5)
                  |
                  R3

        Note that axis2 is calculated by taking the cross product between axis1 and
        the vector chosen for each case, dependent on bonding:

        - case1: if all the bonded atoms are hydrogens, use the principal axes.

        - case2: use XR vector as axis1, arbitrary axis2.

        - case3: use XR vector as axis1, vector XH to calculate axis2

        - case4: use vector XR1 as axis1, and XR2 to calculate axis2

        - case5: get the sum of all XR normalised vectors as axis1, then use vector
          R1R2 to calculate axis2

        axis3 is always the cross product of axis1 and axis2.

        Args:
            system:
                MDAnalysis selection containing all atoms in current frame.
            atom:
                MDAnalysis Atom for the heavy atom.
            dimensions:
                Simulation box dimensions (3,).

        Returns:
            Tuple[np.ndarray | None, np.ndarray | None]:
                - custom_axes: Custom axes (3, 3), or None if atom is not heavy.
                - custom_moment_of_inertia: (3,) moment of inertia around axes.

        Notes:
            If custom_moment_of_inertia is not produced by the chosen method, it is
            computed using :meth:`get_custom_moment_of_inertia` with the heavy atom
            as COM (matching original behaviour).
        """
        # check atom is a heavy atom
        if not atom.mass > 1.1:
            return None, None

        custom_moment_of_inertia = None
        custom_axes = None

        heavy_bonded, light_bonded = self.find_bonded_atoms(atom.index, system)
        ua = atom + light_bonded
        ua_all = atom + heavy_bonded + light_bonded

        # case1
        if len(heavy_bonded) == 0:
            custom_axes, custom_moment_of_inertia = self.get_vanilla_axes(ua_all)

        # case2
        if len(heavy_bonded) == 1 and len(light_bonded) == 0:
            custom_axes = self.get_custom_axes(
                a=atom.position,
                b_list=[heavy_bonded[0].position],
                c=np.zeros(3),
                dimensions=dimensions,
            )

        # case3
        if len(heavy_bonded) == 1 and len(light_bonded) >= 1:
            custom_axes = self.get_custom_axes(
                a=atom.position,
                b_list=[heavy_bonded[0].position],
                c=light_bonded[0].position,
                dimensions=dimensions,
            )

        # case4 (not used in original 2019 code; case5 used instead)
        # case5
        if len(heavy_bonded) >= 2:
            custom_axes = self.get_custom_axes(
                a=atom.position,
                b_list=heavy_bonded.positions,
                c=heavy_bonded[1].position,
                dimensions=dimensions,
            )

        if custom_axes is None:
            return None, None

        if custom_moment_of_inertia is None:
            custom_moment_of_inertia = self.get_custom_moment_of_inertia(
                UA=ua,
                custom_rotation_axes=custom_axes,
                center_of_mass=atom.position,
                dimensions=dimensions,
            )

        # flip axes to face correct way wrt COM
        custom_axes = self.get_flipped_axes(ua, custom_axes, atom.position, dimensions)

        return custom_axes, custom_moment_of_inertia

    def find_bonded_atoms(self, atom_idx: int, system):
        """Find bonded heavy and hydrogen atoms for a given atom.

        Args:
            atom_idx: Atom index to find bonded atoms for.
            system: MDAnalysis selection containing all atoms in current frame.

        Returns:
            Tuple[AtomGroup, AtomGroup]:
                - bonded_heavy_atoms: bonded heavy atoms (mass 2 to 999)
                - bonded_H_atoms: bonded hydrogen atoms (mass 1 to 1.1)
        """
        bonded_atoms = system.select_atoms(f"bonded index {atom_idx}")
        bonded_heavy_atoms = bonded_atoms.select_atoms("mass 2 to 999")
        bonded_H_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
        return bonded_heavy_atoms, bonded_H_atoms

    def get_vanilla_axes(self, molecule):
        """Get principal axes and sorted principal moments (vanilla method).

        Compute the principal axes and moments of inertia for a molecule using
        MDAnalysis built-in functionality.

        The original description is preserved:
        - The molecule is made whole to ensure correct handling of PBC.
        - The moments are obtained by diagonalising the moment of inertia tensor.
        - Eigenvalues are returned sorted from largest to smallest magnitude.

        Args:
            molecule (MDAnalysis.core.groups.AtomGroup):
                AtomGroup representing the molecule/bead.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - principal_axes: (3, 3) axes.
                - moment_of_inertia: (3,) moments sorted descending by |value|.
        """
        moment_of_inertia_tensor = molecule.moment_of_inertia(unwrap=True)
        make_whole(molecule.atoms)
        principal_axes = molecule.principal_axes()

        eigenvalues, _ = np.linalg.eig(moment_of_inertia_tensor)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        moment_of_inertia = eigenvalues[order]

        return principal_axes, moment_of_inertia

    def get_custom_axes(
        self,
        a: np.ndarray,
        b_list: Sequence[np.ndarray],
        c: np.ndarray,
        dimensions: np.ndarray,
    ) -> np.ndarray:
        r"""Compute custom rotation axes from bonded vectors (PBC-aware).

        For atoms a, b_list and c, calculate the axis to rotate forces around:

        - axis1: use the normalised vector ab as axis1. If there is more than one
          bonded heavy atom (HA), average over all the normalised vectors
          calculated from b_list and use this as axis1). b_list contains all the
          bonded heavy atom coordinates.

        - axis2: use the cross product of normalised vector ac and axis1 as axis2.
          If there are more than two bonded heavy atoms, then use normalised vector
          b[0]c to cross product with axis1, this gives the axis perpendicular
          (represented by |_ symbol below) to axis1.

        - axis3: the cross product of axis1 and axis2, which is perpendicular to
          axis1 and axis2.

        Args:
            a: Central united-atom coordinates (3,).
            b_list: Positions of heavy bonded atoms.
            c: Coordinates of a second heavy atom or a hydrogen atom.
            dimensions: Simulation box dimensions (3,).

        ::

            a          1 = norm_ab
           / \         2 = |_ norm_ab and norm_ac (use bc if more than 2 HAs)
          /   \        3 = |_ 1 and 2
        b       c

        Returns:
            np.ndarray: (3, 3) array of the axes used to rotate forces.

        Raises:
            ValueError: If axes cannot be normalized due to degeneracy.
        """
        unscaled_axis1 = np.zeros(3, dtype=float)
        for b in b_list:
            ab_vector = self.get_vector(a, b, dimensions)
            unscaled_axis1 += ab_vector

        if np.allclose(unscaled_axis1, 0.0):
            raise ValueError("Degenerate axis1: summed bonded vectors are zero.")

        if len(b_list) >= 2:
            ac_vector = self.get_vector(c, np.asarray(b_list)[0], dimensions)
        else:
            ac_vector = self.get_vector(c, a, dimensions)

        unscaled_axis2 = np.cross(ac_vector, unscaled_axis1)
        unscaled_axis3 = np.cross(unscaled_axis2, unscaled_axis1)

        unscaled_custom_axes = np.array(
            (unscaled_axis1, unscaled_axis2, unscaled_axis3), dtype=float
        )
        mod = np.sqrt(np.sum(unscaled_custom_axes**2, axis=1))
        if np.any(np.isclose(mod, 0.0)):
            raise ValueError("Degenerate custom axes: cannot normalize (zero norm).")

        scaled_custom_axes = unscaled_custom_axes / mod[:, np.newaxis]
        return scaled_custom_axes

    def get_custom_moment_of_inertia(
        self,
        UA,
        custom_rotation_axes: np.ndarray,
        center_of_mass: np.ndarray,
        dimensions: np.ndarray,
    ) -> np.ndarray:
        """Compute moment of inertia around custom axes for a UA.

        Get the moment of inertia (specifically used for the united atom level)
        from a set of rotation axes and a given center of mass (COM is usually the
        heavy atom position in a UA).

        Original behaviour preserved:
        - Uses PBC-aware translated coordinates.
        - Sums contributions from each atom: |axis x r|^2 * mass.
        - Removes the lowest MOI degree of freedom if the UA only has a single
          bonded H (i.e. UA has 2 atoms total).

        Args:
            UA: MDAnalysis AtomGroup for the UA (heavy + bonded H atoms).
            custom_rotation_axes: (3, 3) array of rotation axes.
            center_of_mass: (3,) COM for the UA (typically HA position).
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: (3,) moment of inertia array.
        """
        translated_coords = self.get_vector(center_of_mass, UA.positions, dimensions)
        custom_moment_of_inertia = np.zeros(3, dtype=float)

        for coord, mass in zip(translated_coords, UA.masses):
            axis_component = np.sum(
                np.cross(custom_rotation_axes, coord) ** 2 * mass, axis=1
            )
            custom_moment_of_inertia += axis_component

        if len(UA) == 2:
            order = custom_moment_of_inertia.argsort()[::-1]  # descending order
            custom_moment_of_inertia[order[-1]] = 0.0

        return custom_moment_of_inertia

    def get_flipped_axes(
        self,
        UA,
        custom_axes: np.ndarray,
        center_of_mass: np.ndarray,
        dimensions: np.ndarray,
    ):
        """Flip custom axes to a consistent direction with respect to the UA.

        For a given set of custom axes, ensure the axes are pointing in the
        correct direction with respect to the heavy atom position and the chosen
        center of mass.

        Args:
            UA: MDAnalysis AtomGroup for the UA.
            custom_axes: (3, 3) array of rotation axes.
            center_of_mass: (3,) COM reference (usually HA position).
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: (3, 3) array of flipped/normalized axes.
        """
        rr_axis = self.get_vector(UA[0].position, center_of_mass, dimensions)

        axis_norm = np.sqrt(np.sum(custom_axes**2, axis=1))
        custom_axes_flipped = custom_axes / axis_norm[:, np.newaxis]

        for i in range(3):
            dot_prod = float(np.dot(custom_axes_flipped[i], rr_axis))
            if dot_prod < 0.0:
                custom_axes_flipped[i] *= -1.0

        return custom_axes_flipped

    def get_vector(self, a: np.ndarray, b: np.ndarray, dimensions: np.ndarray):
        """Compute PBC-wrapped displacement vector(s).

        For vector of two coordinates over periodic boundary conditions (PBCs).

        Args:
            a: (3,) or (N, 3) array of coordinates.
            b: (3,) or (N, 3) array of coordinates.
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: Wrapped displacement vector(s) with broadcasted shape.
        """
        delta = b - a
        delta -= dimensions * np.round(delta / dimensions)
        return delta

    def get_moment_of_inertia_tensor(
        self,
        center_of_mass: np.ndarray,
        positions: np.ndarray,
        masses: Sequence[float],
        dimensions: np.ndarray,
    ) -> np.ndarray:
        """Compute a custom moment of inertia tensor.

        Calculate a custom moment of inertia tensor.
        E.g., for cases where the mass list will contain masses of UAs rather than
        individual atoms and the positions will be those for the UAs only
        (excluding the H atoms coordinates).

        Args:
            center_of_mass: (3,) chosen centre for the tensor.
            positions: (N, 3) point positions.
            masses: (N,) point masses corresponding to positions.
            dimensions: (3,) simulation box dimensions.

        Returns:
            np.ndarray: (3, 3) moment of inertia tensor.
        """
        r = self.get_vector(center_of_mass, positions, dimensions)
        r2 = np.sum(r**2, axis=1)

        masses_arr = np.asarray(list(masses), dtype=float)
        moment_of_inertia_tensor = np.eye(3) * np.sum(masses_arr * r2)
        moment_of_inertia_tensor -= np.einsum("i,ij,ik->jk", masses_arr, r, r)

        return moment_of_inertia_tensor

    def get_custom_principal_axes(
        self, moment_of_inertia_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute principal axes and moments from a custom MOI tensor.

        Principal axes and centre of axes from the ordered eigenvalues and
        eigenvectors of a moment of inertia tensor. This function allows for a
        custom moment of inertia tensor to be used, which isn't possible with the
        built-in MDAnalysis principal_axes() function.

        Original behaviour preserved:
        - Eigenvalues are sorted by descending absolute magnitude.
        - Eigenvectors are transposed so axes are returned as rows.
        - Z axis is flipped to enforce the same handedness convention as the
          original implementation.

        Args:
            moment_of_inertia_tensor: (3, 3) custom inertia tensor.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - principal_axes: (3, 3) principal axes (rows).
                - moment_of_inertia: (3,) principal moments.
        """
        eigenvalues, eigenvectors = np.linalg.eig(moment_of_inertia_tensor)
        order = np.abs(eigenvalues).argsort()[::-1]  # descending order
        transposed = np.transpose(eigenvectors)  # columns -> rows
        moment_of_inertia = eigenvalues[order]
        principal_axes = transposed[order]

        # point z axis in correct direction, as per original code
        cross_xy = np.cross(principal_axes[0], principal_axes[1])
        dot_z = float(np.dot(cross_xy, principal_axes[2]))
        if dot_z < 0:
            principal_axes[2] *= -1

        return principal_axes, moment_of_inertia

    def get_UA_masses(self, molecule) -> list[float]:
        """Return united-atom (UA) masses for a molecule.

        For a given molecule, return a list of masses of UAs (combination of the
        heavy atoms + bonded hydrogen atoms). This list is used to get the moment
        of inertia tensor for molecules larger than one UA.

        Args:
            molecule: MDAnalysis AtomGroup representing the molecule.

        Returns:
            list[float]: UA masses for each heavy atom.
        """
        ua_masses: list[float] = []
        for atom in molecule:
            if atom.mass > 1.1:
                ua_mass = float(atom.mass)
                bonded_atoms = molecule.select_atoms(f"bonded index {atom.index}")
                bonded_h_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
                for h in bonded_h_atoms:
                    ua_mass += float(h.mass)
                ua_masses.append(ua_mass)
        return ua_masses
