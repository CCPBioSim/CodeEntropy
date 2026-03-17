"""Matrix utilities used across covariance and entropy calculations.

This module contains small, focused helpers for matrix construction and cleanup.
All functions are pure (no side effects beyond logging) and operate on NumPy
arrays.

Key behaviors:
- `create_submatrix` computes a 3x3 outer-product block for two 3-vectors.
- `filter_zero_rows_columns` removes rows/columns that are all (near) zero.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MatrixUtils:
    """Utility operations for small matrix manipulations."""

    def create_submatrix(self, data_i: np.ndarray, data_j: np.ndarray) -> np.ndarray:
        """Create a 3x3 covariance-style submatrix from two 3-vectors.

        This computes the outer product of `data_i` and `data_j`:

            submatrix = outer(data_i, data_j)

        Args:
            data_i: Vector of shape (3,) representing bead i values.
            data_j: Vector of shape (3,) representing bead j values.

        Returns:
            A (3, 3) NumPy array corresponding to the outer product.

        Raises:
            ValueError: If either input cannot be reshaped to (3,).
        """
        v_i = np.asarray(data_i, dtype=float).reshape(-1)
        v_j = np.asarray(data_j, dtype=float).reshape(-1)

        if v_i.shape[0] != 3 or v_j.shape[0] != 3:
            raise ValueError(
                f"Expected 3-vectors for outer product, got {v_i.shape} "
                f"and {v_j.shape}."
            )

        submatrix = np.outer(v_i, v_j)
        logger.debug(f"Submatrix: {submatrix}")
        return submatrix

    def filter_zero_rows_columns(
        self, matrix: np.ndarray, atol: float = 0.0
    ) -> np.ndarray:
        """Remove rows and columns that are entirely (near) zero.

        A row (or column) is removed if all entries are close to zero according
        to `np.isclose(..., atol=atol)`.

        Args:
            matrix: Input 2D array.
            atol: Absolute tolerance used to determine "zero". Defaults to 0.0.

        Returns:
            A new matrix with all-zero rows and columns removed. If no such rows
            or columns exist, returns a view/copy of the original with consistent
            NumPy typing.

        Raises:
            ValueError: If `matrix` is not 2D.
        """
        mat = np.asarray(matrix, dtype=float)
        if mat.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got ndim={mat.ndim}.")

        init_shape = mat.shape

        row_mask = self._nonzero_row_mask(mat, atol=atol)
        mat = mat[row_mask, :]

        col_mask = self._nonzero_col_mask(mat, atol=atol)
        mat = mat[:, col_mask]

        final_shape = mat.shape
        if init_shape != final_shape:
            logger.debug(
                f"Matrix shape changed {init_shape}"
                f"-> {final_shape} after removing zero rows/cols."
            )

        logger.debug(f"Filtered matrix: {mat}")
        return mat

    @staticmethod
    def _nonzero_row_mask(matrix: np.ndarray, atol: float) -> np.ndarray:
        """Return a boolean mask selecting rows that are not all (near) zero."""
        is_zero_row = np.all(np.isclose(matrix, 0.0, atol=atol), axis=1)
        return ~is_zero_row

    @staticmethod
    def _nonzero_col_mask(matrix: np.ndarray, atol: float) -> np.ndarray:
        """Return a boolean mask selecting columns that are not all (near) zero."""
        is_zero_col = np.all(np.isclose(matrix, 0.0, atol=atol), axis=0)
        return ~is_zero_col
