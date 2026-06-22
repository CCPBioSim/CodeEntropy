"""Conformational peak detection from dihedral angle observations.

This module contains histogram and peak-identification logic for converting
chunk-local selected-frame dihedral angle observations into global
conformational peak definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from CodeEntropy.levels.dihedrals.angle_observations import (
    DihedralAngleCollector,
    DihedralAngleObservable,
)
from CodeEntropy.levels.dihedrals.kernels import (
    histogram_counts_by_dihedral,
)

logger = logging.getLogger(__name__)

HistogramValues = dict[int, np.ndarray]
HistogramContainer = dict[int, HistogramValues | list[Any]]


@dataclass
class DihedralPeakData:
    """Histogram peak definitions used for conformational state assignment.

    Attributes:
        peaks_ua: United-atom peak values by residue and dihedral index.
        peaks_res: Residue-level peak values by dihedral index.
    """

    peaks_ua: list[list[Any]]
    peaks_res: list[Any]


@dataclass
class DihedralHistogramData:
    """Reduced histogram counts for one conformational group.

    Attributes:
        num_residues: Number of residues in the representative molecule.
        num_dihedrals_ua: Number of united-atom dihedrals by residue index.
        num_dihedrals_res: Number of residue-level dihedrals.
        hist_ua: United-atom histogram counts by residue and dihedral index.
        hist_res: Residue-level histogram counts by dihedral index, or an empty
            list when no residue-level histograms are present.
    """

    num_residues: int
    num_dihedrals_ua: list[int]
    num_dihedrals_res: int
    hist_ua: HistogramContainer
    hist_res: HistogramValues | list[Any]


class ConformationPeakDetector(DihedralAngleCollector):
    """Identify conformational peak definitions from dihedral observations."""

    def _reduce_angle_observables_to_peak_data(
        self,
        observables: list[DihedralAngleObservable],
        level_list: list[Any],
        bin_width: float,
    ) -> DihedralPeakData:
        """Reduce chunk-local angle observables into global peak definitions.

        Args:
            observables: Chunk-local angle observables for one group.
            level_list: Enabled hierarchy levels.
            bin_width: Histogram bin width in degrees.

        Returns:
            Global peak definitions for the group.
        """
        histogram_data = self._reduce_angle_observables_to_histograms(
            observables=observables,
            level_list=level_list,
            bin_width=bin_width,
        )
        return self._build_peak_data_from_histograms(
            histogram_data=histogram_data,
            level_list=level_list,
            bin_width=bin_width,
        )

    def _reduce_angle_observables_to_histograms(
        self,
        observables: list[DihedralAngleObservable],
        level_list: list[Any],
        bin_width: float,
    ) -> DihedralHistogramData:
        """Reduce chunk-local angle arrays into summed histogram counts.

        Args:
            observables: Chunk-local angle observables for one group.
            level_list: Enabled hierarchy levels.
            bin_width: Histogram bin width in degrees.

        Returns:
            Reduced histogram counts for the group.
        """
        if not observables:
            return DihedralHistogramData(
                num_residues=0,
                num_dihedrals_ua=[],
                num_dihedrals_res=0,
                hist_ua={},
                hist_res=[],
            )

        ordered_observables = sorted(
            observables,
            key=lambda observable: (
                observable.task.molecule_order,
                observable.task.chunk_id,
            ),
        )
        number_bins = int(360 / bin_width)
        first = ordered_observables[0]
        num_residues = first.num_residues
        num_dihedrals_ua = [0 for _ in range(num_residues)]
        hist_ua: HistogramContainer = {}
        hist_res: HistogramValues | list[Any] = []
        num_dihedrals_res = 0

        if "united_atom" in level_list:
            for res_id in range(num_residues):
                for observable in ordered_observables:
                    angles = observable.ua_angles_by_residue.get(res_id)
                    if angles is None or angles.shape[1] == 0:
                        hist_ua.setdefault(res_id, [])
                        continue

                    num_dihedrals_ua[res_id] = angles.shape[1]
                    counts = histogram_counts_by_dihedral(angles, number_bins)

                    if res_id not in hist_ua or isinstance(hist_ua[res_id], list):
                        hist_ua[res_id] = {}

                    target = cast(HistogramValues, hist_ua[res_id])
                    for dihedral_index in range(counts.shape[0]):
                        if dihedral_index not in target:
                            target[dihedral_index] = counts[dihedral_index].copy()
                        else:
                            target[dihedral_index] = (
                                target[dihedral_index] + counts[dihedral_index]
                            )

        if "residue" in level_list:
            for observable in ordered_observables:
                if observable.residue_angles is None:
                    continue

                angles = observable.residue_angles
                if angles.shape[1] == 0:
                    continue

                num_dihedrals_res = angles.shape[1]
                counts = histogram_counts_by_dihedral(angles, number_bins)

                if isinstance(hist_res, list):
                    hist_res = {}

                target_res = cast(HistogramValues, hist_res)
                for dihedral_index in range(counts.shape[0]):
                    if dihedral_index not in target_res:
                        target_res[dihedral_index] = counts[dihedral_index].copy()
                    else:
                        target_res[dihedral_index] = (
                            target_res[dihedral_index] + counts[dihedral_index]
                        )

        return DihedralHistogramData(
            num_residues=num_residues,
            num_dihedrals_ua=num_dihedrals_ua,
            num_dihedrals_res=num_dihedrals_res,
            hist_ua=hist_ua,
            hist_res=hist_res,
        )

    def _build_peak_data_from_histograms(
        self,
        histogram_data: DihedralHistogramData,
        level_list: list[Any],
        bin_width: float,
    ) -> DihedralPeakData:
        """Build peak definitions from reduced histogram counts.

        Args:
            histogram_data: Reduced histogram counts for one group.
            level_list: Enabled hierarchy levels.
            bin_width: Histogram bin width in degrees.

        Returns:
            Peak definitions for united-atom and residue-level states.
        """
        peaks_ua: list[list[Any]] = [[] for _ in range(histogram_data.num_residues)]
        peaks_res: list[Any] = []
        number_bins = int(360 / bin_width)
        bin_edges = np.linspace(0.0, 360.0, number_bins + 1)
        bin_value = [
            0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(number_bins)
        ]

        if "united_atom" in level_list:
            for res_id in range(histogram_data.num_residues):
                hist_values = histogram_data.hist_ua.get(res_id)
                if not hist_values:
                    peaks_ua[res_id] = []
                    continue

                hist_values = cast(HistogramValues, hist_values)
                residue_peaks = []
                for dihedral_index in range(histogram_data.num_dihedrals_ua[res_id]):
                    counts = hist_values[dihedral_index]
                    residue_peaks.append(
                        self._find_histogram_peaks(
                            popul=counts,
                            bin_value=bin_value,
                        )
                    )
                peaks_ua[res_id] = residue_peaks

        if "residue" in level_list and histogram_data.hist_res:
            hist_res = cast(HistogramValues, histogram_data.hist_res)
            for dihedral_index in range(histogram_data.num_dihedrals_res):
                counts = hist_res[dihedral_index]
                peaks_res.append(
                    self._find_histogram_peaks(
                        popul=counts,
                        bin_value=bin_value,
                    )
                )

        return DihedralPeakData(peaks_ua=peaks_ua, peaks_res=peaks_res)

    @staticmethod
    def _find_histogram_peaks(
        popul: np.ndarray[Any, Any], bin_value: list[float]
    ) -> list[float]:
        """Return convex turning-point peaks from a histogram.

        Args:
            popul: Histogram bin populations.
            bin_value: Histogram bin centre values.

        Returns:
            List of peak positions.
        """
        number_bins = len(popul)
        peaks: list[float] = []

        for bin_index in range(number_bins):
            if popul[bin_index] == 0:
                continue

            left = popul[bin_index - 1]
            right = popul[0] if bin_index == number_bins - 1 else popul[bin_index + 1]

            if popul[bin_index] >= left and popul[bin_index] > right:
                peaks.append(bin_value[bin_index])

        return peaks
