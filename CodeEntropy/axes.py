import logging

import numpy as np
from MDAnalysis.lib.mdamath import make_whole

logger = logging.getLogger(__name__)


class AxesManager:
    """
    Manages the structural and dynamic levels involved in entropy calculations. This
    includes selecting relevant levels, computing axes for translation and rotation,
    and handling bead-based representations of molecular systems. Provides utility
    methods to extract averaged positions, convert coordinates to spherical systems,
    compute weighted forces and torques, and manipulate matrices used in entropy
    analysis.
    """

    def __init__(self):
        """
        Initializes the LevelManager with placeholders for level-related data,
        including translational and rotational axes, number of beads, and a
        general-purpose data container.
        """
        self.data_container = None
        self._levels = None
        self._trans_axes = None
        self._rot_axes = None
        self._number_of_beads = None

    def get_residue_axes(self, data_container, index):
        """
        The translational and rotational axes at the residue level.

        Args:
          data_container (MDAnalysis.Universe): the molecule and trajectory data
          index (int): residue index

        Returns:
          trans_axes : translational axes (3,3)
          rot_axes : rotational axes (3,3)
          center: center of mass (3,)
          moment_of_inertia: moment of inertia (3,)
        """
        # TODO refine selection so that it will work for branched polymers
        index_prev = index - 1
        index_next = index + 1
        atom_set = data_container.select_atoms(
            f"(resindex {index_prev} or resindex {index_next}) "
            f"and bonded resid {index}"
        )
        residue = data_container.select_atoms(f"resindex {index}")
        center = residue.atoms.center_of_mass(unwrap=True)

        if len(atom_set) == 0:
            # No bonds to other residues
            # Use a custom principal axes, from a MOI tensor
            # that uses positions of heavy atoms only, but including masses
            # of heavy atom + bonded hydrogens
            UAs = residue.select_atoms("mass 2 to 999")
            UA_masses = self.get_UA_masses(residue)
            moment_of_inertia_tensor = self.get_moment_of_inertia_tensor(
                center, UAs.positions, UA_masses, data_container.dimensions[:3]
            )
            rot_axes, moment_of_inertia = self.get_custom_principal_axes(
                moment_of_inertia_tensor
            )
            trans_axes = (
                rot_axes  # set trans axes to same as rot axes as per Jon's code
            )
        else:
            # if bonded to other residues, use default axes and MOI
            make_whole(data_container.atoms)
            trans_axes = data_container.atoms.principal_axes()
            rot_axes, moment_of_inertia = self.get_vanilla_axes(residue)
            center = residue.center_of_mass(unwrap=True)

        return trans_axes, rot_axes, center, moment_of_inertia

    def get_UA_axes(self, data_container, index):
        """
        The translational and rotational axes at the united-atom level.

        Args:
          data_container (MDAnalysis.Universe): the molecule and trajectory data
          index (int): residue index

        Returns:
          trans_axes : translational axes (3,3)
          rot_axes : rotational axes (3,3)
          center: center of mass (3,)
          moment_of_inertia: moment of inertia (3,)
        """

        index = int(index)

        # use the same customPI trans axes as the residue level
        UAs = data_container.select_atoms("mass 2 to 999")
        UA_masses = self.get_UA_masses(data_container.atoms)
        center = data_container.atoms.center_of_mass(unwrap=True)
        moment_of_inertia_tensor = self.get_moment_of_inertia_tensor(
            center, UAs.positions, UA_masses, data_container.dimensions[:3]
        )
        trans_axes, _moment_of_inertia = self.get_custom_principal_axes(
            moment_of_inertia_tensor
        )

        # look for heavy atoms in residue of interest
        heavy_atoms = data_container.select_atoms("prop mass > 1.1")
        heavy_atom_indices = []
        for atom in heavy_atoms:
            heavy_atom_indices.append(atom.index)
        # we find the nth heavy atom
        # where n is the bead index
        heavy_atom_index = heavy_atom_indices[index]
        heavy_atom = data_container.select_atoms(f"index {heavy_atom_index}")

        center = heavy_atom.positions[0]
        rot_axes, moment_of_inertia = self.get_bonded_axes(
            data_container, heavy_atom[0], data_container.dimensions[:3]
        )

        logger.debug(f"Translational Axes: {trans_axes}")
        logger.debug(f"Rotational Axes: {rot_axes}")
        logger.debug(f"Center: {center}")
        logger.debug(f"Moment of Inertia: {moment_of_inertia}")

        return trans_axes, rot_axes, center, moment_of_inertia

    def get_bonded_axes(self, system, atom, dimensions):
        """
        For a given heavy atom, use its bonded atoms to get the axes
        for rotating forces around. Few cases for choosing united atom axes,
        which are dependent on the bonds to the atom:

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
            system: mdanalysis instance of all atoms in current frame
            atom: mdanalysis instance of a heavy atom
            dimensions: dimensions of the simulation box (3,)

        Returns:
            custom_axes: custom axes for the UA, (3,3) array
            custom_moment_of_inertia
        """
        # check atom is a heavy atom
        if not atom.mass > 1.1:
            return None
        # set default values
        custom_moment_of_inertia = None
        custom_axes = None

        # find the heavy bonded atoms and light bonded atoms
        heavy_bonded, light_bonded = self.find_bonded_atoms(atom.index, system)
        UA = atom + light_bonded
        UA_all = atom + heavy_bonded + light_bonded

        # now find which atoms to select to find the axes for rotating forces:
        # case1
        if len(heavy_bonded) == 0:
            custom_axes, custom_moment_of_inertia = self.get_vanilla_axes(UA_all)
        # case2
        if len(heavy_bonded) == 1 and len(light_bonded) == 0:
            custom_axes = self.get_custom_axes(
                atom.position, [heavy_bonded[0].position], np.zeros(3), dimensions
            )
        # case3
        if len(heavy_bonded) == 1 and len(light_bonded) >= 1:
            custom_axes = self.get_custom_axes(
                atom.position,
                [heavy_bonded[0].position],
                light_bonded[0].position,
                dimensions,
            )
        # case4, not used in Jon's 2019 paper code, use case5 instead
        # case5
        if len(heavy_bonded) >= 2:
            custom_axes = self.get_custom_axes(
                atom.position,
                heavy_bonded.positions,
                heavy_bonded[1].position,
                dimensions,
            )

        if custom_moment_of_inertia is None:
            # find moment of inertia using custom axes and atom position as COM
            custom_moment_of_inertia = self.get_custom_moment_of_inertia(
                UA, custom_axes, atom.position, dimensions
            )

        # get the moment of inertia from the custom axes
        if custom_axes is not None:
            # flip axes to face correct way wrt COM
            custom_axes = self.get_flipped_axes(
                UA, custom_axes, atom.position, dimensions
            )

        return custom_axes, custom_moment_of_inertia

    def find_bonded_atoms(self, atom_idx: int, system):
        """
        for a given atom, find its bonded heavy and H atoms

        Args:
            atom_idx: atom index to find bonded heavy atom for
            system: mdanalysis instance of all atoms in current frame

        Returns:
            bonded_heavy_atoms: MDAnalysis instance of bonded heavy atoms
            bonded_H_atoms: MDAnalysis instance of bonded hydrogen atoms
        """
        bonded_atoms = system.select_atoms(f"bonded index {atom_idx}")
        bonded_heavy_atoms = bonded_atoms.select_atoms("mass 2 to 999")
        bonded_H_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
        return bonded_heavy_atoms, bonded_H_atoms

    def get_vanilla_axes(self, molecule):
        """
        From a selection of atoms, get the ordered principal axes (3,3) and
        the ordered moment of inertia axes (3,) for that selection of atoms

        Args:
            molecule: mdanalysis instance of molecule
            molecule_scale: the length scale of molecule

        Returns:
            principal_axes: the principal axes, (3,3) array
            moment_of_inertia: the moment of inertia, (3,) array
        """
        # default moment of inertia
        moment_of_inertia = molecule.moment_of_inertia(unwrap=True)
        make_whole(molecule.atoms)
        principal_axes = molecule.principal_axes()
        # diagonalise moment of inertia tensor here
        # pylint: disable=unused-variable
        eigenvalues, _eigenvectors = np.linalg.eig(moment_of_inertia)
        # sort eigenvalues of moi tensor by largest to smallest magnitude
        order = sorted(eigenvalues, reverse=True)  # decending order
        # principal_axes = principal_axes[order] # PI already ordered correctly
        moment_of_inertia = eigenvalues[order]

        return principal_axes, moment_of_inertia

    def get_custom_axes(
        self, a: np.ndarray, b_list: list, c: np.ndarray, dimensions: np.ndarray
    ):
        r"""
        For atoms a, b_list and c, calculate the axis to rotate forces around:

        - axis1: use the normalised vector ab as axis1. If there is more than one bonded
        heavy atom (HA), average over all the normalised vectors calculated from b_list
        and use this as axis1). b_list contains all the bonded heavy atom
        coordinates.

        - axis2: use the cross product of normalised vector ac and axis1 as axis2.
        If there are more than two bonded heavy atoms, then use normalised vector
        b[0]c to cross product with axis1, this gives the axis perpendicular
        (represented by |_ symbol below) to axis1.

        - axis3: the cross product of axis1 and axis2, which is perpendicular to
        axis1 and axis2.

        Args:
            a: central united-atom coordinates (3,)
            b_list: list of heavy bonded atom positions (3,N)
            c: atom coordinates of either a second heavy atom or a hydrogen atom
            if there are no other bonded heavy atoms in b_list (where N=1 in b_list)
            (3,)
            dimensions: dimensions of the simulation box (3,)

        ::

            a          1 = norm_ab
           / \         2 = |_ norm_ab and norm_ac (use bc if more than 2 HAs)
          /   \        3 = |_ 1 and 2
        b       c

        Returns:
            custom_axes: (3,3) array of the axes used to rotate forces
        """
        unscaled_axis1 = np.zeros(3)
        # average of all heavy atom covalent bond vectors for axis1
        for b in b_list:
            ab_vector = self.get_vector(a, b, dimensions)
            unscaled_axis1 += ab_vector
        if len(b_list) >= 2:
            # use the first heavy bonded atom as atom a
            ac_vector = self.get_vector(c, b_list[0], dimensions)
        else:
            ac_vector = self.get_vector(c, a, dimensions)

        unscaled_axis2 = np.cross(ac_vector, unscaled_axis1)
        unscaled_axis3 = np.cross(unscaled_axis2, unscaled_axis1)

        unscaled_custom_axes = np.array(
            (unscaled_axis1, unscaled_axis2, unscaled_axis3)
        )
        mod = np.sqrt(np.sum(unscaled_custom_axes**2, axis=1))
        scaled_custom_axes = unscaled_custom_axes / mod[:, np.newaxis]

        return scaled_custom_axes

    def get_custom_moment_of_inertia(
        self,
        UA,
        custom_rotation_axes: np.ndarray,
        center_of_mass: np.ndarray,
        dimensions: np.ndarray,
    ):
        """
        Get the moment of inertia (specifically used for the united atom level)
        from a set of rotation axes and a given center of mass
        (COM is usually the heavy atom position in a UA).

        Args:
            UA: MDAnalysis instance of a united-atom
            custom_rotation_axes: (3,3) arrray of rotation axes
            center_of_mass: (3,) center of mass for collection of atoms N

        Returns:
            custom_moment_of_inertia: (3,) array for moment of inertia
        """
        translated_coords = self.get_vector(center_of_mass, UA.positions, dimensions)
        custom_moment_of_inertia = np.zeros(3)
        for coord, mass in zip(translated_coords, UA.masses):
            axis_component = np.sum(
                np.cross(custom_rotation_axes, coord) ** 2 * mass, axis=1
            )
            custom_moment_of_inertia += axis_component

        # Remove lowest MOI degree of freedom if UA only has a single bonded H
        if len(UA) == 2:
            order = custom_moment_of_inertia.argsort()[::-1]  # decending order
            custom_moment_of_inertia[order[-1]] = 0

        return custom_moment_of_inertia

    def get_flipped_axes(self, UA, custom_axes, center_of_mass, dimensions):
        """
        For a given set of custom axes, ensure the axes are pointing in the
        correct direction wrt the heavy atom position and the chosen center
        of mass.

        Args:
            UA: MDAnalysis instance of a united-atom
            custom_axes: (3,3) array of the rotation axes
            center_of_mass: (3,) array for center of mass (usually HA position)
            dimensions: (3,) array of system box dimensions.
        """
        # sorting out PIaxes for MoI for UA fragment

        # get dot product of Paxis1 and CoM->atom1 vect
        # will just be [0,0,0]
        RRaxis = self.get_vector(UA[0].position, center_of_mass, dimensions)

        # flip each Paxis if its pointing out of UA
        custom_axis = np.sum(custom_axes**2, axis=1)
        custom_axes_flipped = custom_axes / custom_axis**0.5
        for i in range(3):
            dotProd1 = np.dot(custom_axes_flipped[i], RRaxis)
            custom_axes_flipped[i] = np.where(
                dotProd1 < 0, -custom_axes_flipped[i], custom_axes_flipped[i]
            )
        return custom_axes_flipped

    def get_vector(self, a: np.ndarray, b: np.ndarray, dimensions: np.ndarray):
        """
        For vector of two coordinates over periodic boundary conditions (PBCs).

        Args:
            a: (N,3) array of atom cooordinates
            b: (3,) array of atom cooordinates
            dimensions: (3,) array of system box dimensions.

        Returns:
            delta_wrapped: (N,3) array of the vector
        """
        delta = b - a
        delta -= dimensions * np.round(delta / dimensions)

        return delta

    def get_moment_of_inertia_tensor(
        self,
        center_of_mass: np.ndarray,
        positions: np.ndarray,
        masses: list,
        dimensions: np.array,
    ) -> np.ndarray:
        """
        Calculate a custom moment of inertia tensor.
        E.g., for cases where the mass list will contain masses of UAs rather than
        individual atoms and the postions will be those for the UAs only
        (excluding the H atoms coordinates).

        Args:
            center_of_mass: a (3,) array of the chosen center of mass
            positions: a (N,3) array of point positions
            masses: a (N,) list of point masses

        Returns:
            moment_of_inertia_tensor: a (3,3) moment of inertia tensor
        """
        r = self.get_vector(center_of_mass, positions, dimensions)
        r2 = np.sum(r**2, axis=1)
        moment_of_inertia_tensor = np.eye(3) * np.sum(masses * r2)
        moment_of_inertia_tensor -= np.einsum("i,ij,ik->jk", masses, r, r)

        return moment_of_inertia_tensor

    def get_custom_principal_axes(
        self, moment_of_inertia_tensor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Principal axes and centre of axes from the ordered eigenvalues
        and eigenvectors of a moment of inertia tensor. This function allows for
        a custom moment of inertia tensor to be used, which isn't possible with
        the built-in MDAnalysis principal_axes() function.

        Args:
            moment_of_inertia_tensor: a (3,3) array of a custom moment of
             inertia tensor

        Returns:
            principal_axes: a (3,3) array for the principal axes
            moment_of_inertia: a (3,) array of the principal axes center
        """
        eigenvalues, eigenvectors = np.linalg.eig(moment_of_inertia_tensor)
        order = abs(eigenvalues).argsort()[::-1]  # decending order
        transposed = np.transpose(eigenvectors)  # turn columns to rows
        moment_of_inertia = eigenvalues[order]
        principal_axes = transposed[order]

        # point z axis in correct direction, as per Jon's code
        cross_xy = np.cross(principal_axes[0], principal_axes[1])
        dot_z = np.dot(cross_xy, principal_axes[2])
        if dot_z < 0:
            principal_axes[2] *= -1

        return principal_axes, moment_of_inertia

    def get_UA_masses(self, molecule) -> list[float]:
        """
        For a given molecule, return a list of masses of UAs
        (combination of the heavy atoms + bonded hydrogen atoms. This list is used to
        get the moment of inertia tensor for molecules larger than one UA.

        Args:
            molecule: mdanalysis instance of molecule

        Returns:
            UA_masses: list of masses for each UA in a molecule
        """
        UA_masses = []
        for atom in molecule:
            if atom.mass > 1.1:
                UA_mass = atom.mass
                bonded_atoms = molecule.select_atoms(f"bonded index {atom.index}")
                bonded_H_atoms = bonded_atoms.select_atoms("mass 1 to 1.1")
                for H in bonded_H_atoms:
                    UA_mass += H.mass
                UA_masses.append(UA_mass)
            else:
                continue
        return UA_masses
