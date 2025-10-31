import logging

import numpy as np

logger = logging.getLogger(__name__)


class CoordinateSystem:
    """ """

    def __init__(self):
        """
        Initializes the CoordinateSystem with placeholders for level-related data,
        including translational and rotational axes, number of beads, and a
        general-purpose data container.
        """

    def get_axes(self, data_container, level, index=0):
        """
        Function to set the translational and rotational axes.
        The translational axes are based on the principal axes of the unit
        one level larger than the level we are interested in (except for
        the polymer level where there is no larger unit). The rotational
        axes use the covalent links between residues or atoms where possible
        to define the axes, or if the unit is not bonded to others of the
        same level the prinicpal axes of the unit are used.

        Args:
          data_container (MDAnalysis.Universe): the molecule and trajectory data
          level (str): the level (united atom, residue, or polymer) of interest
          index (int): residue index

        Returns:
          trans_axes : translational axes
          rot_axes : rotational axes
        """
        index = int(index)

        if level == "polymer":
            # for polymer use principle axis for both translation and rotation
            trans_axes = data_container.atoms.principal_axes()
            rot_axes = data_container.atoms.principal_axes()

        elif level == "residue":
            # Translation
            # for residues use principal axes of whole molecule for translation
            trans_axes = data_container.atoms.principal_axes()

            # Rotation
            # find bonds between atoms in residue of interest and other residues
            # we are assuming bonds only exist between adjacent residues
            # (linear chains of residues)
            # TODO refine selection so that it will work for branched polymers
            index_prev = index - 1
            index_next = index + 1
            atom_set = data_container.select_atoms(
                f"(resindex {index_prev} or resindex {index_next}) "
                f"and bonded resid {index}"
            )
            residue = data_container.select_atoms(f"resindex {index}")

            if len(atom_set) == 0:
                # if no bonds to other residues use pricipal axes of residue
                rot_axes = residue.atoms.principal_axes()

            else:
                # set center of rotation to center of mass of the residue
                center = residue.atoms.center_of_mass()

                # get vector for average position of bonded atoms
                vector = self.get_avg_pos(atom_set, center)

                # use spherical coordinates function to get rotational axes
                rot_axes = self.get_sphCoord_axes(vector)

        elif level == "united_atom":
            # Translation
            # for united atoms use principal axes of residue for translation
            trans_axes = data_container.residues.principal_axes()

            # Rotation
            # for united atoms use heavy atoms bonded to the heavy atom
            atom_set = data_container.select_atoms(
                f"(prop mass > 1.1) and bonded index {index}"
            )

            if len(atom_set) == 0:
                # if no bonds to other residues use pricipal axes of residue
                rot_axes = data_container.residues.principal_axes()
            else:
                # center at position of heavy atom
                atom_group = data_container.select_atoms(f"index {index}")
                center = atom_group.positions[0]

                # get vector for average position of bonded atoms
                vector = self.get_avg_pos(atom_set, center)

                # use spherical coordinates function to get rotational axes
                rot_axes = self.get_sphCoord_axes(vector)

        logger.debug(f"Translational Axes: {trans_axes}")
        logger.debug(f"Rotational Axes: {rot_axes}")

        return trans_axes, rot_axes

    def get_avg_pos(self, atom_set, center):
        """
        Function to get the average position of a set of atoms.

        Args:
            atom_set : MDAnalysis atom group
            center : position for center of rotation

        Returns:
            avg_position : three dimensional vector
        """
        # start with an empty vector
        avg_position = np.zeros((3))

        # get number of atoms
        number_atoms = len(atom_set.names)

        if number_atoms != 0:
            # sum positions for all atoms in the given set
            for atom_index in range(number_atoms):
                atom_position = atom_set.atoms[atom_index].position

                avg_position += atom_position

            avg_position /= number_atoms  # divide by number of atoms to get average

        else:
            # if no atoms in set the unit has no bonds to restrict its rotational
            # motion, so we can use a random vector to get spherical
            # coordinate axes
            avg_position = np.random.random(3)

        # transform the average position to a coordinate system with the origin
        # at center
        avg_position = avg_position - center

        logger.debug(f"Average Position: {avg_position}")

        return avg_position

    def get_sphCoord_axes(self, arg_r):
        """
        For a given vector in space, treat it is a radial vector rooted at
        0,0,0 and derive a curvilinear coordinate system according to the
        rules of polar spherical coordinates

        Args:
            arg_r: 3 dimensional vector

        Returns:
            spherical_basis: axes set (3 vectors)
        """

        x2y2 = arg_r[0] ** 2 + arg_r[1] ** 2
        r2 = x2y2 + arg_r[2] ** 2

        # Check for division by zero
        if r2 == 0.0:
            raise ValueError("r2 is zero, cannot compute spherical coordinates.")

        if x2y2 == 0.0:
            raise ValueError("x2y2 is zero, cannot compute sin_phi and cos_phi.")

        # These conditions are mathematically unreachable for real-valued vectors.
        # Marked as no cover to avoid false negatives in coverage reports.

        # Check for non-negative values inside the square root
        if x2y2 / r2 < 0:  # pragma: no cover
            raise ValueError(
                f"Negative value encountered for sin_theta calculation: {x2y2 / r2}. "
                f"Cannot take square root."
            )

        if x2y2 < 0:  # pragma: no cover
            raise ValueError(
                f"Negative value encountered for sin_phi and cos_phi "
                f"calculation: {x2y2}. "
                f"Cannot take square root."
            )

        if x2y2 != 0.0:
            sin_theta = np.sqrt(x2y2 / r2)
            cos_theta = arg_r[2] / np.sqrt(r2)

            sin_phi = arg_r[1] / np.sqrt(x2y2)
            cos_phi = arg_r[0] / np.sqrt(x2y2)

        else:  # pragma: no cover
            sin_theta = 0.0
            cos_theta = 1

            sin_phi = 0.0
            cos_phi = 1

        # if abs(sin_theta) > 1 or abs(sin_phi) > 1:
        #     print('Bad sine : T {} , P {}'.format(sin_theta, sin_phi))

        # cos_theta = np.sqrt(1 - sin_theta*sin_theta)
        # cos_phi = np.sqrt(1 - sin_phi*sin_phi)

        # print('{} {} {}'.format(*arg_r))
        # print('Sin T : {}, cos T : {}'.format(sin_theta, cos_theta))
        # print('Sin P : {}, cos P : {}'.format(sin_phi, cos_phi))

        spherical_basis = np.zeros((3, 3))

        # r^
        spherical_basis[0, :] = np.asarray(
            [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
        )

        # Theta^
        spherical_basis[1, :] = np.asarray(
            [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta]
        )

        # Phi^
        spherical_basis[2, :] = np.asarray([-sin_phi, cos_phi, 0.0])

        logger.debug(f"Spherical Basis: {spherical_basis}")

        return spherical_basis
