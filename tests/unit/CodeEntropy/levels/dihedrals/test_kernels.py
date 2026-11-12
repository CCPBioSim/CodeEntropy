from __future__ import annotations

import numpy as np

from CodeEntropy.levels.dihedrals.kernels import (
    assign_peak_labels_and_count_flexible,
    histogram_counts_by_dihedral,
    wrap_degrees_positive,
)


def test_wrap_degrees_positive_returns_copy_and_wraps_negative_values():
    angles = np.array([[-10.0, 20.0], [-180.0, 0.0]], dtype=np.float64)

    wrapped = wrap_degrees_positive(angles)

    np.testing.assert_allclose(wrapped, np.array([[350.0, 20.0], [180.0, 0.0]]))
    np.testing.assert_allclose(angles, np.array([[-10.0, 20.0], [-180.0, 0.0]]))


def test_histogram_counts_by_dihedral_counts_each_dihedral_series():
    angles = np.array(
        [
            [0.0, 89.0],
            [90.0, 180.0],
            [359.0, 360.0],
        ],
        dtype=np.float64,
    )

    counts = histogram_counts_by_dihedral(angles, number_bins=4)

    np.testing.assert_array_equal(
        counts,
        np.array(
            [
                [1, 1, 0, 1],
                [1, 0, 1, 1],
            ],
            dtype=np.int64,
        ),
    )


def test_assign_peak_labels_uses_first_minimum_tie_and_counts_flexible():
    angles = np.array(
        [
            [5.0, 100.0],
            [15.0, 100.0],
        ],
        dtype=np.float64,
    )
    padded_peaks = np.array(
        [
            [0.0, 10.0],
            [100.0, 0.0],
        ],
        dtype=np.float64,
    )
    peak_counts = np.array([2, 1], dtype=np.int64)

    labels, flexible = assign_peak_labels_and_count_flexible(
        angles,
        padded_peaks,
        peak_counts,
    )

    np.testing.assert_array_equal(labels, np.array([[0, 0], [1, 0]], dtype=np.int64))
    assert flexible == 1


def test_assign_peak_labels_handles_dihedrals_with_no_peaks():
    angles = np.array([[10.0], [20.0]], dtype=np.float64)
    padded_peaks = np.zeros((1, 1), dtype=np.float64)
    peak_counts = np.array([0], dtype=np.int64)

    labels, flexible = assign_peak_labels_and_count_flexible(
        angles,
        padded_peaks,
        peak_counts,
    )

    np.testing.assert_array_equal(labels, np.zeros((2, 1), dtype=np.int64))
    assert flexible == 0


def test_histogram_counts_by_dihedral_clamps_negative_values_to_first_bin():
    angles = np.array([[-1.0], [10.0]], dtype=np.float64)

    counts = histogram_counts_by_dihedral(angles, number_bins=4)

    np.testing.assert_array_equal(counts, np.array([[2, 0, 0, 0]]))


def test_histogram_counts_by_dihedral_clamps_negative_bin_to_zero():
    angles = np.array([[-90.0]], dtype=np.float64)

    counts = histogram_counts_by_dihedral(
        angles=angles,
        number_bins=4,
    )

    np.testing.assert_array_equal(
        counts,
        np.array([[1, 0, 0, 0]], dtype=np.int64),
    )
