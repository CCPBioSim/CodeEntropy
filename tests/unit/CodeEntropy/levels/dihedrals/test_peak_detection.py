from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.levels.dihedrals.angle_observations import (
    ConformationChunkTask,
    DihedralAngleObservable,
)
from CodeEntropy.levels.dihedrals.peak_detection import (
    ConformationPeakDetector,
    DihedralHistogramData,
    DihedralPeakData,
)
from CodeEntropy.trajectory.frames import FrameSelection


class _PeakDetector(ConformationPeakDetector):
    """Concrete peak detector for unit tests."""

    def __init__(self) -> None:
        """Initialize the test detector."""
        self._universe_operations = MagicMock()


def _make_task(
    molecule_order: int = 0,
    chunk_id: int = 0,
) -> ConformationChunkTask:
    """Build a minimal conformation chunk task.

    Args:
        molecule_order: Molecule order in the group.
        chunk_id: Frame-chunk id.

    Returns:
        ConformationChunkTask for one selected frame.
    """
    return ConformationChunkTask(
        group_id=0,
        molecule_id=molecule_order,
        molecule_order=molecule_order,
        chunk_id=chunk_id,
        frame_indices=(chunk_id,),
        frame_selection=FrameSelection(indices=(chunk_id,)),
    )


def test_find_histogram_peaks_hits_interior_and_wraparound_last_bin():
    popul = np.array([0, 2, 0, 3], dtype=np.int64)
    bin_value = [10.0, 20.0, 30.0, 40.0]

    peaks = _PeakDetector._find_histogram_peaks(popul=popul, bin_value=bin_value)

    assert peaks == [20.0, 40.0]


def test_reduce_angle_observables_to_peak_data_delegates_to_reducers():
    detector = _PeakDetector()
    histogram_data = DihedralHistogramData(0, [], 0, {}, [])
    peak_data = DihedralPeakData(peaks_ua=[], peaks_res=[])

    detector._reduce_angle_observables_to_histograms = MagicMock(
        return_value=histogram_data
    )
    detector._build_peak_data_from_histograms = MagicMock(return_value=peak_data)

    out = detector._reduce_angle_observables_to_peak_data(
        observables=[],
        level_list=["united_atom"],
        bin_width=30.0,
    )

    assert out is peak_data
    detector._reduce_angle_observables_to_histograms.assert_called_once_with(
        observables=[],
        level_list=["united_atom"],
        bin_width=30.0,
    )
    detector._build_peak_data_from_histograms.assert_called_once_with(
        histogram_data=histogram_data,
        level_list=["united_atom"],
        bin_width=30.0,
    )


def test_reduce_angle_observables_to_histograms_handles_empty_observables():
    detector = _PeakDetector()

    histogram_data = detector._reduce_angle_observables_to_histograms(
        observables=[],
        level_list=["united_atom", "residue"],
        bin_width=90.0,
    )

    assert histogram_data.num_residues == 0
    assert histogram_data.num_dihedrals_ua == []
    assert histogram_data.num_dihedrals_res == 0
    assert histogram_data.hist_ua == {}
    assert histogram_data.hist_res == []


def test_reduce_angle_observables_to_histograms_sums_chunk_counts():
    detector = _PeakDetector()
    observables = [
        DihedralAngleObservable(
            task=_make_task(molecule_order=0, chunk_id=1),
            num_residues=1,
            ua_angles_by_residue={0: np.array([[190.0]], dtype=np.float64)},
            residue_angles=np.array([[190.0]], dtype=np.float64),
        ),
        DihedralAngleObservable(
            task=_make_task(molecule_order=0, chunk_id=0),
            num_residues=1,
            ua_angles_by_residue={0: np.array([[10.0], [100.0]], dtype=np.float64)},
            residue_angles=np.array([[10.0], [100.0]], dtype=np.float64),
        ),
    ]

    histogram_data = detector._reduce_angle_observables_to_histograms(
        observables=observables,
        level_list=["united_atom", "residue"],
        bin_width=90.0,
    )

    np.testing.assert_array_equal(
        histogram_data.hist_ua[0][0],
        np.array([1, 1, 1, 0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        histogram_data.hist_res[0],
        np.array([1, 1, 1, 0], dtype=np.int64),
    )
    assert histogram_data.num_dihedrals_ua == [1]
    assert histogram_data.num_dihedrals_res == 1


def test_reduce_angle_observables_to_histograms_handles_empty_ua_angles():
    detector = _PeakDetector()
    observable = DihedralAngleObservable(
        task=_make_task(),
        num_residues=1,
        ua_angles_by_residue={0: np.empty((2, 0), dtype=np.float64)},
        residue_angles=None,
    )

    histogram_data = detector._reduce_angle_observables_to_histograms(
        observables=[observable],
        level_list=["united_atom"],
        bin_width=90.0,
    )

    assert histogram_data.num_residues == 1
    assert histogram_data.num_dihedrals_ua == [0]
    assert histogram_data.hist_ua == {0: []}


def test_reduce_angle_observables_to_histograms_skips_missing_residue_angles():
    detector = _PeakDetector()
    observable = DihedralAngleObservable(
        task=_make_task(),
        num_residues=1,
        ua_angles_by_residue={},
        residue_angles=None,
    )

    histogram_data = detector._reduce_angle_observables_to_histograms(
        observables=[observable],
        level_list=["residue"],
        bin_width=90.0,
    )

    assert histogram_data.num_dihedrals_res == 0
    assert histogram_data.hist_res == []


def test_reduce_angle_observables_to_histograms_skips_empty_residue_angles():
    detector = _PeakDetector()
    observable = DihedralAngleObservable(
        task=_make_task(),
        num_residues=1,
        ua_angles_by_residue={},
        residue_angles=np.empty((2, 0), dtype=np.float64),
    )

    histogram_data = detector._reduce_angle_observables_to_histograms(
        observables=[observable],
        level_list=["residue"],
        bin_width=90.0,
    )

    assert histogram_data.num_dihedrals_res == 0
    assert histogram_data.hist_res == []


def test_reduce_angle_observables_to_histograms_initialises_residue_histograms():
    detector = _PeakDetector()
    observable = DihedralAngleObservable(
        task=_make_task(),
        num_residues=1,
        ua_angles_by_residue={},
        residue_angles=np.array([[10.0], [20.0]], dtype=np.float64),
    )

    histogram_data = detector._reduce_angle_observables_to_histograms(
        observables=[observable],
        level_list=["residue"],
        bin_width=90.0,
    )

    assert histogram_data.num_dihedrals_res == 1
    assert isinstance(histogram_data.hist_res, dict)
    np.testing.assert_array_equal(
        histogram_data.hist_res[0],
        np.array([2, 0, 0, 0], dtype=np.int64),
    )


def test_build_peak_data_from_histograms_finds_ua_and_residue_peaks():
    detector = _PeakDetector()
    histogram_data = DihedralHistogramData(
        num_residues=1,
        num_dihedrals_ua=[1],
        num_dihedrals_res=1,
        hist_ua={0: {0: np.array([0, 2, 0, 1], dtype=np.int64)}},
        hist_res={0: np.array([1, 0, 3, 0], dtype=np.int64)},
    )

    peak_data = detector._build_peak_data_from_histograms(
        histogram_data=histogram_data,
        level_list=["united_atom", "residue"],
        bin_width=90.0,
    )

    assert peak_data == DihedralPeakData(
        peaks_ua=[[[135.0, 315.0]]],
        peaks_res=[[45.0, 225.0]],
    )


def test_build_peak_data_from_histograms_handles_empty_ua_histogram():
    detector = _PeakDetector()
    histogram_data = DihedralHistogramData(
        num_residues=1,
        num_dihedrals_ua=[0],
        num_dihedrals_res=0,
        hist_ua={0: []},
        hist_res=[],
    )

    peak_data = detector._build_peak_data_from_histograms(
        histogram_data=histogram_data,
        level_list=["united_atom"],
        bin_width=90.0,
    )

    assert peak_data.peaks_ua == [[]]
    assert peak_data.peaks_res == []
