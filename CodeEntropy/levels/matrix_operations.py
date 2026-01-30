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
