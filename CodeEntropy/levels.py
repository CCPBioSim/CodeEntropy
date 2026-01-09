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

    def __init__(self, universe_operations):
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
        self._universe_operations = universe_operations

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

            atoms_in_fragment = fragments[molecule].select_atoms("prop mass > 1.1")
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
        highest_level,
        force_matrix,
        torque_matrix,
        force_partitioning,
    ):
        """
        Compute and accumulate force/torque covariance matrices for a given level.

        Parameters:
          data_container (MDAnalysis.Universe): Data for a molecule or residue.
          level (str): 'polymer', 'residue', or 'united_atom'.
          highest_level (bool): Whether this is the top (largest bead size) level.
          force_matrix, torque_matrix (np.ndarray or None): Accumulated matrices to add
          to.
          force_partitioning (float): Factor to adjust force contributions,
          default is 0.5.

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
            bead = list_of_beads[bead_index]
            # Set up axes
            # translation and rotation use different axes
            # how the axes are defined depends on the level
            trans_axes = data_container.atoms.principal_axes()
            rot_axes = np.real(bead.principal_axes())

            # Sort out coordinates, forces, and torques for each atom in the bead
            weighted_forces[bead_index] = self.get_weighted_forces(
                data_container,
                bead,
                trans_axes,
                highest_level,
                force_partitioning,
            )
            weighted_torques[bead_index] = self.get_weighted_torques(
                data_container, bead, rot_axes, force_partitioning
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

        if level == "united_atom":
            list_of_beads = []
            heavy_atoms = data_container.select_atoms("prop mass > 1.1")
            if len(heavy_atoms) == 0:
                # molecule without heavy atoms would be a hydrogen molecule
                list_of_beads.append(data_container.select_atoms("all"))
            else:
                # Select one heavy atom and all light atoms bonded to it
                for atom in heavy_atoms:
                    atom_group = (
                        "index "
                        + str(atom.index)
                        + " or ((prop mass <= 1.1) and bonded index "
                        + str(atom.index)
                        + ")"
                    )
                    list_of_beads.append(data_container.select_atoms(atom_group))

        logger.debug(f"List of beads: {list_of_beads}")

        return list_of_beads

    def get_weighted_forces(
        self, data_container, bead, trans_axes, highest_level, force_partitioning
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

    def get_weighted_torques(self, data_container, bead, rot_axes, force_partitioning):
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
        moment_of_inertia = np.zeros(3)

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

            # multiply by the force_partitioning parameter to avoid double counting
            # of the forces on weakly correlated atoms
            # the default value of force_partitioning is 0.5 (dividing by two)
        #      torques = force_partitioning * torques

        # divide by moment of inertia to get weighted torques
        # moment of inertia is a 3x3 tensor
        # the weighting is done in each dimension (x,y,z) using
        # the sorted eigenvalues of the moment of inertia tensor
        eigenvalues, _ = np.linalg.eig(bead.moment_of_inertia())
        moments_of_inertia = sorted(eigenvalues, reverse=True)

        for dimension in range(3):
            # Skip calculation if torque is already zero
            if np.isclose(torques[dimension], 0):
                weighted_torque[dimension] = 0
                continue

            # Check for zero moment of inertia
            if np.isclose(moments_of_inertia[dimension], 0):
                # If moment of inertia is 0 there should be 0 torque
                weighted_torque[dimension] = 0
                logger.warning("Zero moment of inertia. Setting torque to 0")
                continue

            # Check for negative moment of inertia
            if moments_of_inertia[dimension] < 0:
                raise ValueError(
                    f"Negative value encountered for moment of inertia: "
                    f"{moment_of_inertia[dimension]} "
                    f"Cannot compute weighted torque."
                )

            # Compute weighted torque
            weighted_torque[dimension] = torques[dimension] / np.sqrt(
                moments_of_inertia[dimension]
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
        force_partitioning,
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
        force_partitioning : float
            Factor to adjust force contributions, default is 0.5.


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
                        mol = self._universe_operations.get_molecule_container(
                            reduced_atom, mol_id
                        )
                        for level in levels[mol_id]:
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
                                force_partitioning,
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
        force_partitioning,
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
        force_partitioning : float
         Factor to adjust force contributions, default is 0.5.
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
                res = self._universe_operations.new_U_select_atom(
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
                    highest,
                    None if key not in force_avg["ua"] else force_avg["ua"][key],
                    None if key not in torque_avg["ua"] else torque_avg["ua"][key],
                    force_partitioning,
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
                highest,
                None if force_avg[key][group_id] is None else force_avg[key][group_id],
                (
                    None
                    if torque_avg[key][group_id] is None
                    else torque_avg[key][group_id]
                ),
                force_partitioning,
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
