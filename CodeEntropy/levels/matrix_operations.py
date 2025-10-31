import logging

import numpy as np

logger = logging.getLogger(__name__)


class MatrixOperations:
    """ """

    def __init__(self):
        """
        Initializes the MatrixOperations with placeholders for level-related data,
        including translational and rotational axes, number of beads, and a
        general-purpose data container.
        """

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
