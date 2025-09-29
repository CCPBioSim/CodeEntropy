import logging

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logger = logging.getLogger(__name__)


class LevelManager:
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

    def select_levels(self, data_container):
        """
        Function to read input system and identify the number of molecules and
        the levels (i.e. united atom, residue and/or polymer) that should be used.
        The level refers to the size of the bead (atom or collection of atoms)
        that will be used in the entropy calculations.

        Args:
            arg_DataContainer: MDAnalysis universe object containing the system of
            interest

        Returns:
             number_molecules (int): Number of molecules in the system.
             levels (array): Strings describing the length scales for each molecule.
        """

        # fragments is MDAnalysis terminology for what chemists would call molecules
        number_molecules = len(data_container.atoms.fragments)
        logger.debug(f"The number of molecules is {number_molecules}.")

        fragments = data_container.atoms.fragments
        levels = [[] for _ in range(number_molecules)]

        for molecule in range(number_molecules):
            levels[molecule].append(
                "united_atom"
            )  # every molecule has at least one atom

            atoms_in_fragment = fragments[molecule].select_atoms("not name H*")
            number_residues = len(atoms_in_fragment.residues)

            if len(atoms_in_fragment) > 1:
                levels[molecule].append("residue")

                if number_residues > 1:
                    levels[molecule].append("polymer")

        logger.debug(f"levels {levels}")

        return number_molecules, levels

    def get_matrices(
        self,
        data_container,
        level,
        number_frames,
        highest_level,
        force_matrix,
        torque_matrix,
    ):
        """
        Compute and accumulate force/torque covariance matrices for a given level.

        Parameters:
          data_container (MDAnalysis.Universe): Data for a molecule or residue.
          level (str): 'polymer', 'residue', or 'united_atom'.
          number_frames (int): Number of frames being processed.
          highest_level (bool): Whether this is the top (largest bead size) level.
          force_matrix, torque_matrix (np.ndarray or None): Accumulated matrices to add
          to.

        Returns:
          force_matrix (np.ndarray): Accumulated force covariance matrix.
          torque_matrix (np.ndarray): Accumulated torque covariance matrix.
        """

        # Make beads
        list_of_beads = self.get_beads(data_container, level)

        # number of beads and frames in trajectory
        number_beads = len(list_of_beads)

        # initialize force and torque arrays
        weighted_forces = [None for _ in range(number_beads)]
        weighted_torques = [None for _ in range(number_beads)]

        # Calculate forces/torques for each bead
        for bead_index in range(number_beads):
            # Set up axes
            # translation and rotation use different axes
            # how the axes are defined depends on the level
            trans_axes, rot_axes = self.get_axes(data_container, level, bead_index)

            # Sort out coordinates, forces, and torques for each atom in the bead
            weighted_forces[bead_index] = self.get_weighted_forces(
                data_container, list_of_beads[bead_index], trans_axes, highest_level
            )
            weighted_torques[bead_index] = self.get_weighted_torques(
                data_container, list_of_beads[bead_index], rot_axes
            )

        # Create covariance submatrices
        force_submatrix = [
            [0 for _ in range(number_beads)] for _ in range(number_beads)
        ]
        torque_submatrix = [
            [0 for _ in range(number_beads)] for _ in range(number_beads)
        ]

        for i in range(number_beads):
            for j in range(i, number_beads):
                f_sub = self.create_submatrix(weighted_forces[i], weighted_forces[j])
                t_sub = self.create_submatrix(weighted_torques[i], weighted_torques[j])
                force_submatrix[i][j] = f_sub
                force_submatrix[j][i] = f_sub.T
                torque_submatrix[i][j] = t_sub
                torque_submatrix[j][i] = t_sub.T

        # Convert block matrices to full matrix
        force_block = np.block(
            [
                [force_submatrix[i][j] for j in range(number_beads)]
                for i in range(number_beads)
            ]
        )
        torque_block = np.block(
            [
                [torque_submatrix[i][j] for j in range(number_beads)]
                for i in range(number_beads)
            ]
        )

        # Enforce consistent shape before accumulation
        if force_matrix is None:
            force_matrix = np.zeros_like(force_block)
        elif force_matrix.shape != force_block.shape:
            raise ValueError(
                f"Inconsistent force matrix shape: existing "
                f"{force_matrix.shape}, new {force_block.shape}"
            )
        else:
            force_matrix = force_block

        if torque_matrix is None:
            torque_matrix = np.zeros_like(torque_block)
        elif torque_matrix.shape != torque_block.shape:
            raise ValueError(
                f"Inconsistent torque matrix shape: existing "
                f"{torque_matrix.shape}, new {torque_block.shape}"
            )
        else:
            torque_matrix = torque_block

        return force_matrix, torque_matrix

    def get_dihedrals(self, data_container, level):
        """
        Define the set of dihedrals for use in the conformational entropy function.
        If united atom level, the dihedrals are defined from the heavy atoms
        (4 bonded atoms for 1 dihedral).
        If residue level, use the bonds between residues to cast dihedrals.
        Note: not using improper dihedrals only ones with 4 atoms/residues
        in a linear arrangement.

        Args:
          data_container (MDAnalysis.Universe): system information
          level (str): level of the hierarchy (should be residue or polymer)

        Returns:
           dihedrals (array): set of dihedrals
        """
        # Start with empty array
        dihedrals = []

        # if united atom level, read dihedrals from MDAnalysis universe
        if level == "united_atom":
            dihedrals = data_container.dihedrals

        # if residue level, looking for dihedrals involving residues
        if level == "residue":
            num_residues = len(data_container.residues)
            logger.debug(f"Number Residues: {num_residues}")
            if num_residues < 4:
                logger.debug("no residue level dihedrals")

            else:
                # find bonds between residues N-3:N-2 and N-1:N
                for residue in range(4, num_residues + 1):
                    # Using MDAnalysis selection,
                    # assuming only one covalent bond between neighbouring residues
                    # TODO not written for branched polymers
                    atom_string = (
                        "resindex "
                        + str(residue - 4)
                        + " and bonded resindex "
                        + str(residue - 3)
                    )
                    atom1 = data_container.select_atoms(atom_string)

                    atom_string = (
                        "resindex "
                        + str(residue - 3)
                        + " and bonded resindex "
                        + str(residue - 4)
                    )
                    atom2 = data_container.select_atoms(atom_string)

                    atom_string = (
                        "resindex "
                        + str(residue - 2)
                        + " and bonded resindex "
                        + str(residue - 1)
                    )
                    atom3 = data_container.select_atoms(atom_string)

                    atom_string = (
                        "resindex "
                        + str(residue - 1)
                        + " and bonded resindex "
                        + str(residue - 2)
                    )
                    atom4 = data_container.select_atoms(atom_string)

                    atom_group = atom1 + atom2 + atom3 + atom4
                    dihedrals.append(atom_group.dihedral)

        logger.debug(f"Level: {level}, Dihedrals: {dihedrals}")

        return dihedrals

    def compute_dihedral_conformations(
        self,
        selector,
        level,
        number_frames,
        bin_width,
        start,
        end,
        step,
        ce,
    ):
        """
        Compute dihedral conformations for a given selector and entropy level.

        Parameters:
            selector (AtomGroup): Atom selection to compute dihedrals for.
            level (str): Entropy level ("united_atom" or "residue").
            number_frames (int): Number of frames to process.
            bin_width (float): Bin width for dihedral angle discretization.
            start (int): Start frame index.
            end (int): End frame index.
            step (int): Step size for frame iteration.
            ce : Conformational Entropy class

        Returns:
            states (list): List of conformation strings per frame.
        """
        # Identify the dihedral angles in the residue/molecule
        dihedrals = self.get_dihedrals(selector, level)

        # When there are no dihedrals, there is only one possible conformation
        # so the conformational states are not relevant
        if len(dihedrals) == 0:
            logger.debug("No dihedrals found; skipping conformation assignment.")
            states = []
        else:
            # Identify the conformational label for each dihedral at each frame
            num_dihedrals = len(dihedrals)
            conformation = np.zeros((num_dihedrals, number_frames))

            for i, dihedral in enumerate(dihedrals):
                conformation[i] = ce.assign_conformation(
                    selector, dihedral, number_frames, bin_width, start, end, step
                )

            # for all the dihedrals available concatenate the label of each
            # dihedral into the state for that frame
            states = [
                state
                for state in (
                    "".join(str(int(conformation[d][f])) for d in range(num_dihedrals))
                    for f in range(number_frames)
                )
                if state
            ]

        logger.debug(f"level: {level}, states: {states}")
        return states

    def get_beads(self, data_container, level):
        """
        Function to define beads depending on the level in the hierarchy.

        Args:
           data_container (MDAnalysis.Universe): the molecule data
           level (str): the heirarchy level (polymer, residue, or united atom)

        Returns:
           list_of_beads : the relevent beads
        """

        if level == "polymer":
            list_of_beads = []
            atom_group = "all"
            list_of_beads.append(data_container.select_atoms(atom_group))

        if level == "residue":
            list_of_beads = []
            num_residues = len(data_container.residues)
            for residue in range(num_residues):
                atom_group = "resindex " + str(residue)
                list_of_beads.append(data_container.select_atoms(atom_group))

        # NOTE this could cause problems for hydrogen or helium molecules
        if level == "united_atom":
            list_of_beads = []
            heavy_atoms = data_container.select_atoms("not name H*")
            for atom in heavy_atoms:
                atom_group = (
                    "index "
                    + str(atom.index)
                    + " or (name H* and bonded index "
                    + str(atom.index)
                    + ")"
                )
                list_of_beads.append(data_container.select_atoms(atom_group))

        logger.debug(f"List of beads: {list_of_beads}")

        return list_of_beads

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
                f"not name H* and bonded index {index}"
            )

            if len(atom_set) == 0:
                # if no bonds to other residues use pricipal axes of residue
                rot_axes = data_container.residues.principal_axes()
            else:
                # center at position of heavy atom
                atom_group = data_container.select_atoms(f"index {index}")
                center = atom_group.positions[0]

                # get vector for average position of hydrogens
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

    def get_weighted_forces(
        self, data_container, bead, trans_axes, highest_level, force_partitioning=0.5
    ):
        """
        Function to calculate the mass weighted forces for a given bead.

        Args:
           data_container (MDAnalysis.Universe): Contains atomic positions and forces.
           bead : The part of the molecule to be considered.
           trans_axes (np.ndarray): The axes relative to which the forces are located.
           highest_level (bool): Is this the largest level of the length scale hierarchy
           force_partitioning (float): Factor to adjust force contributions to avoid
           over counting correlated forces, default is 0.5.

        Returns:
            weighted_force (np.ndarray): The mass-weighted sum of the forces in the
            bead.
        """

        forces_trans = np.zeros((3,))

        # Sum forces from all atoms in the bead
        for atom in bead.atoms:
            # update local forces in translational axes
            forces_local = np.matmul(trans_axes, data_container.atoms[atom.index].force)
            forces_trans += forces_local

        if highest_level:
            # multiply by the force_partitioning parameter to avoid double counting
            # of the forces on weakly correlated atoms
            # the default value of force_partitioning is 0.5 (dividing by two)
            forces_trans = force_partitioning * forces_trans

        # divide the sum of forces by the mass of the bead to get the weighted forces
        mass = bead.total_mass()

        # Check that mass is positive to avoid division by 0 or negative values inside
        # sqrt
        if mass <= 0:
            raise ValueError(
                f"Invalid mass value: {mass}. Mass must be positive to compute the "
                f"square root."
            )

        weighted_force = forces_trans / np.sqrt(mass)

        logger.debug(f"Weighted Force: {weighted_force}")

        return weighted_force

    def get_weighted_torques(
        self, data_container, bead, rot_axes, force_partitioning=0.5
    ):
        """
        Function to calculate the moment of inertia weighted torques for a given bead.

        This function computes torques in a rotated frame and then weights them using
        the moment of inertia tensor. To prevent numerical instability, it treats
        extremely small diagonal elements of the moment of inertia tensor as zero
        (since values below machine precision are effectively zero). This avoids
        unnecessary use of extended precision (e.g., float128).

        Additionally, if the computed torque is already zero, the function skips
        the division step, reducing unnecessary computations and potential errors.

        Parameters
        ----------
        data_container : object
            Contains atomic positions and forces.
        bead : object
            The part of the molecule to be considered.
        rot_axes : np.ndarray
            The axes relative to which the forces and coordinates are located.
        force_partitioning : float, optional
            Factor to adjust force contributions, default is 0.5.

        Returns
        -------
        weighted_torque : np.ndarray
            The mass-weighted sum of the torques in the bead.
        """

        torques = np.zeros((3,))
        weighted_torque = np.zeros((3,))

        for atom in bead.atoms:

            # update local coordinates in rotational axes
            coords_rot = (
                data_container.atoms[atom.index].position - bead.center_of_mass()
            )
            coords_rot = np.matmul(rot_axes, coords_rot)
            # update local forces in rotational frame
            forces_rot = np.matmul(rot_axes, data_container.atoms[atom.index].force)

            # multiply by the force_partitioning parameter to avoid double counting
            # of the forces on weakly correlated atoms
            # the default value of force_partitioning is 0.5 (dividing by two)
            forces_rot = force_partitioning * forces_rot

            # define torques (cross product of coordinates and forces) in rotational
            # axes
            torques_local = np.cross(coords_rot, forces_rot)
            torques += torques_local

        # divide by moment of inertia to get weighted torques
        # moment of inertia is a 3x3 tensor
        # the weighting is done in each dimension (x,y,z) using the diagonal
        # elements of the moment of inertia tensor
        moment_of_inertia = bead.moment_of_inertia()

        for dimension in range(3):
            # Skip calculation if torque is already zero
            if np.isclose(torques[dimension], 0):
                weighted_torque[dimension] = 0
                continue

            # Check for zero moment of inertia
            if np.isclose(moment_of_inertia[dimension, dimension], 0):
                raise ZeroDivisionError(
                    f"Attempted to divide by zero moment of inertia in dimension "
                    f"{dimension}."
                )

            # Check for negative moment of inertia
            if moment_of_inertia[dimension, dimension] < 0:
                raise ValueError(
                    f"Negative value encountered for moment of inertia: "
                    f"{moment_of_inertia[dimension, dimension]} "
                    f"Cannot compute weighted torque."
                )

            # Compute weighted torque
            weighted_torque[dimension] = torques[dimension] / np.sqrt(
                moment_of_inertia[dimension, dimension]
            )

        logger.debug(f"Weighted Torque: {weighted_torque}")

        return weighted_torque

    def create_submatrix(self, data_i, data_j):
        """
        Function for making covariance matrices.

        Args
        -----
        data_i : values for bead i
        data_j : values for bead j

        Returns
        ------
        submatrix : 3x3 matrix for the covariance between i and j
        """

        # Start with 3 by 3 matrix of zeros
        submatrix = np.zeros((3, 3))

        # For each frame calculate the outer product (cross product) of the data from
        # the two beads and add the result to the submatrix
        outer_product_matrix = np.outer(data_i, data_j)
        submatrix = np.add(submatrix, outer_product_matrix)

        logger.debug(f"Submatrix: {submatrix}")

        return submatrix

    def build_covariance_matrices(
        self,
        entropy_manager,
        reduced_atom,
        levels,
        groups,
        start,
        end,
        step,
        number_frames,
    ):
        """
        Construct average force and torque covariance matrices for all molecules and
        entropy levels.

        Parameters
        ----------
        entropy_manager : EntropyManager
            Instance of the EntropyManager.
        reduced_atom : Universe
            The reduced atom selection.
        levels : dict
            Dictionary mapping molecule IDs to lists of entropy levels.
        groups : dict
            Dictionary mapping group IDs to lists of molecule IDs.
        start : int
            Start frame index.
        end : int
            End frame index.
        step : int
            Step size for frame iteration.
        number_frames : int
            Total number of frames to process.

        Returns
        -------
        tuple
            force_avg : dict
                Averaged force covariance matrices by entropy level.
            torque_avg : dict
                Averaged torque covariance matrices by entropy level.
        """
        number_groups = len(groups)

        force_avg = {
            "ua": {},
            "res": [None] * number_groups,
            "poly": [None] * number_groups,
        }
        torque_avg = {
            "ua": {},
            "res": [None] * number_groups,
            "poly": [None] * number_groups,
        }

        total_steps = len(reduced_atom.trajectory[start:end:step])
        total_items = (
            sum(len(levels[mol_id]) for mols in groups.values() for mol_id in mols)
            * total_steps
        )

        frame_counts = {
            "ua": {},
            "res": np.zeros(number_groups, dtype=int),
            "poly": np.zeros(number_groups, dtype=int),
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[title]}", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
        ) as progress:

            task = progress.add_task(
                "[green]Processing...",
                total=total_items,
                title="Starting...",
            )

            indices = list(range(number_frames))
            for time_index, _ in zip(indices, reduced_atom.trajectory[start:end:step]):
                for group_id, molecules in groups.items():
                    for mol_id in molecules:
                        mol = entropy_manager._get_molecule_container(
                            reduced_atom, mol_id
                        )
                        for level in levels[mol_id]:
                            mol = entropy_manager._get_molecule_container(
                                reduced_atom, mol_id
                            )

                            resname = mol.atoms[0].resname
                            resid = mol.atoms[0].resid
                            segid = mol.atoms[0].segid

                            mol_label = f"{resname}_{resid} (segid {segid})"

                            progress.update(
                                task,
                                title=f"Building covariance matrices | "
                                f"Timestep {time_index} | "
                                f"Molecule: {mol_label} | "
                                f"Level: {level}",
                            )

                            self.update_force_torque_matrices(
                                entropy_manager,
                                mol,
                                group_id,
                                level,
                                levels[mol_id],
                                time_index,
                                number_frames,
                                force_avg,
                                torque_avg,
                                frame_counts,
                            )

                            progress.advance(task)

        return force_avg, torque_avg, frame_counts

    def update_force_torque_matrices(
        self,
        entropy_manager,
        mol,
        group_id,
        level,
        level_list,
        time_index,
        num_frames,
        force_avg,
        torque_avg,
        frame_counts,
    ):
        """
        Update the running averages of force and torque covariance matrices
        for a given molecule and entropy level.

        This function computes the force and torque covariance matrices for the
        current frame and updates the existing averages in-place using the incremental
        mean formula:

            new_avg = old_avg + (value - old_avg) / n

        where n is the number of frames processed so far for that molecule/level
        combination. This ensures that the averages are maintained without storing
        all previous frame data.

        Parameters
        ----------
        entropy_manager : EntropyManager
            Instance of the EntropyManager.
        mol : AtomGroup
            The molecule to process.
        group_id : int
            Index of the group to which the molecule belongs.
        level : str
            Current entropy level ("united_atom", "residue", or "polymer").
        level_list : list
            List of entropy levels for the molecule.
        time_index : int
            Index of the current frame relative to the start of the trajectory slice.
        num_frames : int
            Total number of frames to process.
        force_avg : dict
            Dictionary holding the running average force matrices, keyed by entropy
            level.
        torque_avg : dict
            Dictionary holding the running average torque matrices, keyed by entropy
            level.
        frame_counts : dict
            Dictionary holding the count of frames processed for each molecule/level
            combination.

        Returns
        -------
        None
            Updates are performed in-place on `force_avg`, `torque_avg`, and
            `frame_counts`.
        """
        highest = level == level_list[-1]

        # United atom level calculations are done separately for each residue
        # This allows information per residue to be output and keeps the
        # matrices from becoming too large
        if level == "united_atom":
            for res_id, residue in enumerate(mol.residues):
                key = (group_id, res_id)
                res = entropy_manager._run_manager.new_U_select_atom(
                    mol, f"index {residue.atoms.indices[0]}:{residue.atoms.indices[-1]}"
                )

                # This is to get MDAnalysis to get the information from the
                # correct frame of the trajectory
                res.trajectory[time_index]

                # Build the matrices, adding data from each timestep
                # Being careful for the first timestep when data has not yet
                # been added to the matrices
                f_mat, t_mat = self.get_matrices(
                    res,
                    level,
                    num_frames,
                    highest,
                    None if key not in force_avg["ua"] else force_avg["ua"][key],
                    None if key not in torque_avg["ua"] else torque_avg["ua"][key],
                )

                if key not in force_avg["ua"]:
                    force_avg["ua"][key] = f_mat.copy()
                    torque_avg["ua"][key] = t_mat.copy()
                    frame_counts["ua"][key] = 1
                else:
                    frame_counts["ua"][key] += 1
                    n = frame_counts["ua"][key]
                    force_avg["ua"][key] += (f_mat - force_avg["ua"][key]) / n
                    torque_avg["ua"][key] += (t_mat - torque_avg["ua"][key]) / n

        elif level in ["residue", "polymer"]:
            # This is to get MDAnalysis to get the information from the
            # correct frame of the trajectory
            mol.trajectory[time_index]

            key = "res" if level == "residue" else "poly"

            # Build the matrices, adding data from each timestep
            # Being careful for the first timestep when data has not yet
            # been added to the matrices
            f_mat, t_mat = self.get_matrices(
                mol,
                level,
                num_frames,
                highest,
                None if force_avg[key][group_id] is None else force_avg[key][group_id],
                (
                    None
                    if torque_avg[key][group_id] is None
                    else torque_avg[key][group_id]
                ),
            )

            if force_avg[key][group_id] is None:
                force_avg[key][group_id] = f_mat.copy()
                torque_avg[key][group_id] = t_mat.copy()
                frame_counts[key][group_id] = 1
            else:
                frame_counts[key][group_id] += 1
                n = frame_counts[key][group_id]
                force_avg[key][group_id] += (f_mat - force_avg[key][group_id]) / n
                torque_avg[key][group_id] += (t_mat - torque_avg[key][group_id]) / n

        return frame_counts

    def filter_zero_rows_columns(self, arg_matrix):
        """
        function for removing rows and columns that contain only zeros from a matrix

        Args:
            arg_matrix : matrix

        Returns:
            arg_matrix : the reduced size matrix
        """

        # record the initial size
        init_shape = np.shape(arg_matrix)

        zero_indices = list(
            filter(
                lambda row: np.all(np.isclose(arg_matrix[row, :], 0.0)),
                np.arange(np.shape(arg_matrix)[0]),
            )
        )
        all_indices = np.ones((np.shape(arg_matrix)[0]), dtype=bool)
        all_indices[zero_indices] = False
        arg_matrix = arg_matrix[all_indices, :]

        all_indices = np.ones((np.shape(arg_matrix)[1]), dtype=bool)
        zero_indices = list(
            filter(
                lambda col: np.all(np.isclose(arg_matrix[:, col], 0.0)),
                np.arange(np.shape(arg_matrix)[1]),
            )
        )
        all_indices[zero_indices] = False
        arg_matrix = arg_matrix[:, all_indices]

        # get the final shape
        final_shape = np.shape(arg_matrix)

        if init_shape != final_shape:
            logger.debug(
                "A shape change has occurred ({},{}) -> ({}, {})".format(
                    *init_shape, *final_shape
                )
            )

        logger.debug(f"arg_matrix: {arg_matrix}")

        return arg_matrix

    def build_conformational_states(
        self,
        entropy_manager,
        reduced_atom,
        levels,
        groups,
        start,
        end,
        step,
        number_frames,
        bin_width,
        ce,
    ):
        """
        Construct the conformational states for each molecule at
        relevant levels.

        Parameters:
            entropy_manager (EntropyManager): Instance of the EntropyManager
            reduced_atom (Universe): The reduced atom selection.
            levels (list): List of entropy levels per molecule.
            groups (dict): Groups for averaging over molecules.
            start (int): Start frame index.
            end (int): End frame index.
            step (int): Step size for frame iteration.
            number_frames (int): Total number of frames to process.
            bin_width (int): Width of histogram bins.
            ce: Conformational Entropy object

        Returns:
            tuple: A tuple containing:
                - states_ua (dict): Conformational states at the united-atom level.
                - states_res (list): Conformational states at the residue level.
        """
        number_groups = len(groups)
        states_ua = {}
        states_res = [None] * number_groups

        total_items = sum(
            len(levels[mol_id]) for mols in groups.values() for mol_id in mols
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[title]}", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
        ) as progress:

            task = progress.add_task(
                "[green]Building Conformational States...",
                total=total_items,
                title="Starting...",
            )

            for group_id in groups.keys():
                molecules = groups[group_id]
                for mol_id in molecules:
                    mol = entropy_manager._get_molecule_container(reduced_atom, mol_id)

                    resname = mol.atoms[0].resname
                    resid = mol.atoms[0].resid
                    segid = mol.atoms[0].segid

                    mol_label = f"{resname}_{resid} (segid {segid})"

                    for level in levels[mol_id]:
                        progress.update(
                            task,
                            title=f"Building conformational states | "
                            f"Molecule: {mol_label} | "
                            f"Level: {level}",
                        )

                        if level == "united_atom":
                            for res_id, residue in enumerate(mol.residues):
                                key = (group_id, res_id)

                                res_container = (
                                    entropy_manager._run_manager.new_U_select_atom(
                                        mol,
                                        f"index {residue.atoms.indices[0]}:"
                                        f"{residue.atoms.indices[-1]}",
                                    )
                                )
                                heavy_res = (
                                    entropy_manager._run_manager.new_U_select_atom(
                                        res_container, "not name H*"
                                    )
                                )
                                states = self.compute_dihedral_conformations(
                                    heavy_res,
                                    level,
                                    number_frames,
                                    bin_width,
                                    start,
                                    end,
                                    step,
                                    ce,
                                )

                                if key in states_ua:
                                    states_ua[key].extend(states)
                                else:
                                    states_ua[key] = states

                        elif level == "residue":
                            states = self.compute_dihedral_conformations(
                                mol,
                                level,
                                number_frames,
                                bin_width,
                                start,
                                end,
                                step,
                                ce,
                            )

                            if states_res[group_id] is None:
                                states_res[group_id] = states
                            else:
                                states_res[group_id].extend(states)

                        progress.advance(task)

        logger.debug(f"states_ua {states_ua}")
        logger.debug(f"states_res {states_res}")

        return states_ua, states_res
