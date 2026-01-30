import logging

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from CodeEntropy.axes import AxesManager

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
        customised_axes,
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
            axes_manager = AxesManager()
            if level == "united_atom" and customised_axes:
                trans_axes, rot_axes, center, moment_of_inertia = (
                    axes_manager.get_UA_axes(data_container, bead_index)
                )
            elif level == "residue" and customised_axes:
                trans_axes, rot_axes, center, moment_of_inertia = (
                    axes_manager.get_residue_axes(data_container, bead_index)
                )
            else:
                trans_axes = data_container.atoms.principal_axes()
                rot_axes = np.real(bead.principal_axes())
                eigenvalues, _ = np.linalg.eig(bead.moment_of_inertia())
                moment_of_inertia = sorted(eigenvalues, reverse=True)
                center = bead.center_of_mass()

            # Sort out coordinates, forces, and torques for each atom in the bead
            weighted_forces[bead_index] = self.get_weighted_forces(
                data_container,
                bead,
                trans_axes,
                highest_level,
                force_partitioning,
            )
            weighted_torques[bead_index] = self.get_weighted_torques(
                bead,
                rot_axes,
                center,
                force_partitioning,
                moment_of_inertia,
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
            force_matrix = force_block  # add first set of forces
        elif force_matrix.shape != force_block.shape:
            raise ValueError(
                f"Inconsistent force matrix shape: existing "
                f"{force_matrix.shape}, new {force_block.shape}"
            )
        else:
            force_matrix = force_block

        if torque_matrix is None:
            torque_matrix = np.zeros_like(torque_block)
            torque_matrix = torque_block  # add first set of torques
        elif torque_matrix.shape != torque_block.shape:
            raise ValueError(
                f"Inconsistent torque matrix shape: existing "
                f"{torque_matrix.shape}, new {torque_block.shape}"
            )
        else:
            torque_matrix = torque_block

        return force_matrix, torque_matrix

    def get_combined_forcetorque_matrices(
        self,
        data_container,
        level,
        highest_level,
        forcetorque_matrix,
        force_partitioning,
        customised_axes,
    ):
        """
        Compute and accumulate combined force/torque covariance matrices for
        a given level.

        Parameters:
          data_container (MDAnalysis.Universe): Data for a molecule or residue.
          level (str): 'polymer', 'residue', or 'united_atom'.
          highest_level (bool): Whether this is the top (largest bead size) level.
          forcetorque_matrix (np.ndarray or None): Accumulated matrices to add
          to.
          force_partitioning (float): Factor to adjust force contributions,
          default is 0.5.

        Returns:
          forcetorque_matrix (np.ndarray): Accumulated torque covariance matrix.
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
            axes_manager = AxesManager()
            if level == "residue" and customised_axes:
                trans_axes, rot_axes, center, moment_of_inertia = (
                    axes_manager.get_residue_axes(data_container, bead_index)
                )
            else:
                trans_axes = data_container.atoms.principal_axes()
                rot_axes = np.real(bead.principal_axes())
                eigenvalues, _ = np.linalg.eig(bead.moment_of_inertia())
                moment_of_inertia = sorted(eigenvalues, reverse=True)
                center = bead.center_of_mass()

            # Sort out coordinates, forces, and torques for each atom in the bead
            weighted_forces[bead_index] = self.get_weighted_forces(
                data_container,
                bead,
                trans_axes,
                highest_level,
                force_partitioning,
            )
            weighted_torques[bead_index] = self.get_weighted_torques(
                bead,
                rot_axes,
                center,
                force_partitioning,
                moment_of_inertia,
            )

        # Create covariance submatrices
        forcetorque_submatrix = [
            [0 for _ in range(number_beads)] for _ in range(number_beads)
        ]

        for i in range(number_beads):
            for j in range(i, number_beads):
                ft_sub = self.create_FTsubmatrix(
                    np.concatenate((weighted_forces[i], weighted_torques[i])),
                    np.concatenate((weighted_forces[j], weighted_torques[j])),
                )
                forcetorque_submatrix[i][j] = ft_sub
                forcetorque_submatrix[j][i] = ft_sub.T

        # Convert block matrices to full matrix
        forcetorque_block = np.block(
            [
                [forcetorque_submatrix[i][j] for j in range(number_beads)]
                for i in range(number_beads)
            ]
        )

        # Enforce consistent shape before accumulation
        if forcetorque_matrix is None:
            forcetorque_matrix = np.zeros_like(forcetorque_block)
            forcetorque_matrix = forcetorque_block  # add first set of torques
        elif forcetorque_matrix.shape != forcetorque_block.shape:
            raise ValueError(
                f"Inconsistent forcetorque matrix shape: existing "
                f"{forcetorque_matrix.shape}, new {forcetorque_block.shape}"
            )
        else:
            forcetorque_matrix = forcetorque_block

        return forcetorque_matrix

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
        Compute mass-weighted translational forces for a bead.

        The forces acting on all atoms belonging to the bead are first transformed
        into the provided translational reference frame and summed. If this bead
        corresponds to the highest level of a hierarchical coarse-graining scheme,
        the total force is scaled by a force-partitioning factor to avoid double
        counting forces from weakly correlated atoms.

        The resulting force vector is then normalized by the square root of the
        bead's total mass.

        Parameters
        ----------
        data_container : MDAnalysis.Universe
            Container holding atomic positions and forces.
        bead : object
            Molecular subunit whose atoms contribute to the force.
        trans_axes : np.ndarray
            Transformation matrix defining the translational reference frame.
        highest_level : bool
            Whether this bead is the highest level in the length-scale hierarchy.
            If True, force partitioning is applied.
        force_partitioning : float
            Scaling factor applied to forces to avoid over-counting correlated
            contributions (typically 0.5).

        Returns
        -------
        weighted_force : np.ndarray
            Mass-weighted translational force acting on the bead.

        Raises
        ------
        ValueError
            If the bead mass is zero or negative.
        """
        forces_trans = np.zeros((3,))

        for atom in bead.atoms:
            forces_local = np.matmul(trans_axes, data_container.atoms[atom.index].force)
            forces_trans += forces_local

        if highest_level:
            forces_trans = force_partitioning * forces_trans

        mass = bead.total_mass()

        if mass <= 0:
            raise ValueError(
                f"Invalid mass value: {mass}. Mass must be positive to compute the "
                f"square root."
            )

        weighted_force = forces_trans / np.sqrt(mass)

        logger.debug(f"Weighted Force: {weighted_force}")

        return weighted_force

    def get_weighted_torques(
        self, bead, rot_axes, center, force_partitioning, moment_of_inertia
    ):
        """
        Compute moment-of-inertia weighted torques for a bead.

        Atomic coordinates and forces are transformed into the provided rotational
        reference frame. Torques are computed as the cross product of position
        vectors (relative to the bead center of mass) and forces, with a
        force-partitioning factor applied to reduce over-counting of correlated
        atomic contributions.

        The total torque vector is then weighted by the square root of the bead's
        principal moments of inertia. Weighting is performed component-wise using
        the sorted eigenvalues of the moment of inertia tensor.

        To ensure numerical stability:
        - Torque components that are effectively zero, zero or negative are skipped.

        Parameters
        ----------
        data_container : object
            Container holding atomic positions and forces.
        bead : object
            Molecular subunit whose atoms contribute to the torque.
        rot_axes : np.ndarray
            Transformation matrix defining the rotational reference frame.
        force_partitioning : float
            Scaling factor applied to forces to avoid over-counting correlated
            contributions (typically 0.5).
        moment_of_inertia : np.ndarray
            Moment of inertia (3,)

        Returns
        -------
        weighted_torque : np.ndarray
            Moment-of-inertia weighted torque acting on the bead.
        """

        # translate and rotate positions and forces
        translated_coords = bead.positions - center
        rotated_coords = np.tensordot(translated_coords, rot_axes.T, axes=1)
        rotated_forces = np.tensordot(bead.forces, rot_axes.T, axes=1)
        # scale forces
        rotated_forces *= force_partitioning
        # get torques here
        torques = np.cross(rotated_coords, rotated_forces)
        torques = np.sum(torques, axis=0)

        weighted_torque = np.zeros((3,))
        for dimension in range(3):
            if np.isclose(torques[dimension], 0):
                weighted_torque[dimension] = 0
                continue

            if moment_of_inertia[dimension] == 0:
                weighted_torque[dimension] = 0
                continue

            if moment_of_inertia[dimension] < 0:
                weighted_torque[dimension] = 0
                continue

            # Compute weighted torque
            weighted_torque[dimension] = torques[dimension] / np.sqrt(
                moment_of_inertia[dimension]
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

    def create_FTsubmatrix(self, data_i, data_j):
        """
        Function for making covariance matrices.

        Args
        -----
        data_i : values for bead i
        data_j : values for bead j

        Returns
        ------
        submatrix : 6x6 matrix for the covariance between i and j
        """

        # Start with 6 by 6 matrix of zeros
        submatrix = np.zeros((6, 6))

        # For each frame calculate the outer product (cross product) of the data from
        # the two beads and add the result to the submatrix
        outer_product_matrix = np.outer(data_i, data_j)
        submatrix = np.add(submatrix, outer_product_matrix)

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
        combined_forcetorque,
        customised_axes,
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
        combined_forcetorque : bool
         Whether to use combined forcetorque covariance matrix.

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

        forcetorque_avg = {
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
                                forcetorque_avg,
                                frame_counts,
                                force_partitioning,
                                combined_forcetorque,
                                customised_axes,
                            )

                            progress.advance(task)

        return force_avg, torque_avg, forcetorque_avg, frame_counts

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
        forcetorque_avg,
        frame_counts,
        force_partitioning,
        combined_forcetorque,
        customised_axes,
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
        combined_forcetorque : bool
            Whether to use combined forcetorque covariance matrix.
        customised_axes: bool
            Whether to use bonded axes for UA rovib calculations
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
                    customised_axes,
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
            if highest and combined_forcetorque:
                # use combined forcetorque covariance matrix for the highest level only
                ft_mat = self.get_combined_forcetorque_matrices(
                    mol,
                    level,
                    highest,
                    (
                        None
                        if forcetorque_avg[key][group_id] is None
                        else forcetorque_avg[key][group_id]
                    ),
                    force_partitioning,
                    customised_axes,
                )

                if forcetorque_avg[key][group_id] is None:
                    forcetorque_avg[key][group_id] = ft_mat.copy()
                    frame_counts[key][group_id] = 1
                else:
                    frame_counts[key][group_id] += 1
                    n = frame_counts[key][group_id]
                    forcetorque_avg[key][group_id] += (
                        ft_mat - forcetorque_avg[key][group_id]
                    ) / n
            else:
                f_mat, t_mat = self.get_matrices(
                    mol,
                    level,
                    highest,
                    (
                        None
                        if force_avg[key][group_id] is None
                        else force_avg[key][group_id]
                    ),
                    (
                        None
                        if torque_avg[key][group_id] is None
                        else torque_avg[key][group_id]
                    ),
                    force_partitioning,
                    customised_axes,
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
