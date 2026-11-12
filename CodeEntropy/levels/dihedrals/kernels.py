"""Numba kernels for dihedral conformational-state analysis.

This module contains numeric kernels used by the serial chunked conformational
workflow. The kernels operate only on NumPy arrays and avoid MDAnalysis objects
so they remain safe to JIT compile and reuse inside future distributed workers.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def wrap_degrees_positive(angles: np.ndarray) -> np.ndarray:
    """Return dihedral angles wrapped into the positive degree range.

    Args:
        angles: Angle array in degrees. The expected shape is
            ``(n_frames, n_dihedrals)``.

    Returns:
        Copy of ``angles`` with negative values shifted by 360 degrees.
    """
    wrapped = angles.copy()

    for frame_i in range(wrapped.shape[0]):
        for dihedral_i in range(wrapped.shape[1]):
            if wrapped[frame_i, dihedral_i] < 0.0:
                wrapped[frame_i, dihedral_i] += 360.0

    return wrapped


@njit(cache=True)
def histogram_counts_by_dihedral(
    angles: np.ndarray,
    number_bins: int,
) -> np.ndarray:
    """Build histogram counts for each dihedral angle series.

    Args:
        angles: Positive-angle array with shape ``(n_frames, n_dihedrals)``.
        number_bins: Number of histogram bins spanning 0 to 360 degrees.

    Returns:
        Histogram counts with shape ``(n_dihedrals, number_bins)``.
    """
    n_frames = angles.shape[0]
    n_dihedrals = angles.shape[1]
    counts = np.zeros((n_dihedrals, number_bins), dtype=np.int64)
    bin_width = 360.0 / float(number_bins)

    for frame_i in range(n_frames):
        for dihedral_i in range(n_dihedrals):
            value = angles[frame_i, dihedral_i]
            bin_i = int(value / bin_width)

            if bin_i < 0:
                bin_i = 0
            elif bin_i >= number_bins:
                bin_i = number_bins - 1

            counts[dihedral_i, bin_i] += 1

    return counts


@njit(cache=True)
def assign_peak_labels_and_count_flexible(
    angles: np.ndarray,
    padded_peaks: np.ndarray,
    peak_counts: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Assign nearest-peak labels and count flexible dihedrals.

    Args:
        angles: Positive-angle array with shape ``(n_frames, n_dihedrals)``.
        padded_peaks: Peak values with shape ``(n_dihedrals, max_peaks)``.
        peak_counts: Number of valid peaks for each dihedral.

    Returns:
        Tuple containing an integer label array with shape
        ``(n_frames, n_dihedrals)`` and the number of flexible dihedrals.
    """
    n_frames = angles.shape[0]
    n_dihedrals = angles.shape[1]
    labels = np.zeros((n_frames, n_dihedrals), dtype=np.int64)
    flexible_count = 0

    for dihedral_i in range(n_dihedrals):
        n_peaks = peak_counts[dihedral_i]

        if n_peaks < 1:
            continue

        for frame_i in range(n_frames):
            value = angles[frame_i, dihedral_i]
            best_label = 0
            best_distance = abs(value - padded_peaks[dihedral_i, 0])

            for peak_i in range(1, n_peaks):
                distance = abs(value - padded_peaks[dihedral_i, peak_i])
                if distance < best_distance:
                    best_distance = distance
                    best_label = peak_i

            labels[frame_i, dihedral_i] = best_label

        first_label = labels[0, dihedral_i]
        for frame_i in range(1, n_frames):
            if labels[frame_i, dihedral_i] != first_label:
                flexible_count += 1
                break

    return labels, flexible_count
