import numpy as np
import pytest

from CodeEntropy.levels.linalg import MatrixUtils


def test_create_submatrix_outer_product_correct():
    mu = MatrixUtils()
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    out = mu.create_submatrix(a, b)

    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, np.outer(a, b))


def test_create_submatrix_rejects_non_3_vectors():
    mu = MatrixUtils()
    with pytest.raises(ValueError):
        mu.create_submatrix(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_filter_zero_rows_columns_removes_all_zero_rows_and_cols():
    mu = MatrixUtils()
    mat = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    out = mu.filter_zero_rows_columns(mat)

    assert out.shape == (1, 1)
    assert out[0, 0] == 2.0


def test_filter_zero_rows_columns_uses_atol():
    mu = MatrixUtils()
    mat = np.array(
        [
            [1e-9, 0.0],
            [0.0, 1.0],
        ]
    )

    out = mu.filter_zero_rows_columns(mat, atol=1e-8)
    assert out.shape == (1, 1)
    assert out[0, 0] == 1.0


def test_filter_zero_rows_columns_rejects_non_2d():
    mu = MatrixUtils()
    with pytest.raises(ValueError):
        mu.filter_zero_rows_columns(np.array([1.0, 2.0, 3.0]))
