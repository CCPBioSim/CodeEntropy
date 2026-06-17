"""Conformational state assignment from dihedral peak definitions.

This module contains the logic for converting positive-angle dihedral arrays and
global peak definitions into state labels and flexible-dihedral counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from CodeEntropy.levels.dihedrals.angle_observations import (
    ConformationChunkTask,
    DihedralAngleObservable,
)
from CodeEntropy.levels.dihedrals.kernels import (
    assign_peak_labels_and_count_flexible,
)
from CodeEntropy.levels.dihedrals.topology import MoleculeDihedralTopology

UAKey = tuple[int, int]


@dataclass
class ConformationStateData:
    """Serial conformational state data calculated for one molecule group.

    Attributes:
        state_res: Residue-level state labels for the group.
        flex_res: Number of flexible residue-level dihedrals for the group.
        states_ua_updates: United-atom state-label updates by ``(group, residue)``.
        flexible_ua_updates: United-atom flexible-dihedral updates by
            ``(group, residue)``.
    """

    state_res: list[str]
    flex_res: int
    states_ua_updates: dict[UAKey, list[str]]
    flexible_ua_updates: dict[UAKey, int]


@dataclass
class ConformationStatePartial:
    """Chunk-local conformational state labels and flexible counts.

    Attributes:
        task: Source molecule/frame chunk task.
        state_res: Residue-level state labels for this chunk.
        flex_res: Number of flexible residue-level dihedrals for this chunk.
        states_ua_updates: United-atom state-label updates by ``(group, residue)``.
        flexible_ua_updates: United-atom flexible-dihedral updates by
            ``(group, residue)``.
    """

    task: ConformationChunkTask
    state_res: list[str]
    flex_res: int
    states_ua_updates: dict[UAKey, list[str]]
    flexible_ua_updates: dict[UAKey, int]


class ConformationStateAssigner:
    """Assign conformational state labels from global dihedral peak definitions."""

    def _assign_state_partial_from_observable(
        self,
        observable: DihedralAngleObservable,
        topology: MoleculeDihedralTopology,
        level_list: list[Any],
        peaks_ua: list[list[Any]],
        peaks_res: list[Any],
    ) -> ConformationStatePartial:
        """Assign chunk-local states from cached angle arrays and global peaks.

        Args:
            observable: Chunk-local angle observable.
            topology: Static topology for the observable molecule.
            level_list: Enabled hierarchy levels.
            peaks_ua: Global united-atom peaks by residue.
            peaks_res: Global residue-level peaks.

        Returns:
            Chunk-local state partial.
        """
        state_res: list[str] = []
        flex_res = 0
        states_ua_updates: dict[UAKey, list[str]] = {}
        flexible_ua_updates: dict[UAKey, int] = {}

        if "united_atom" in level_list:
            for res_id in range(topology.num_residues):
                key = (topology.group_id, res_id)
                angles = observable.ua_angles_by_residue.get(res_id)

                if angles is None or angles.shape[1] == 0:
                    states_ua_updates[key] = []
                    flexible_ua_updates[key] = 0
                    continue

                states, flexible = self._process_conformations_from_angles(
                    peaks=peaks_ua[res_id],
                    angles=angles,
                )
                states_ua_updates[key] = states
                flexible_ua_updates[key] = flexible

        if "residue" in level_list and observable.residue_angles is not None:
            if observable.residue_angles.shape[1] > 0:
                state_res, flex_res = self._process_conformations_from_angles(
                    peaks=peaks_res,
                    angles=observable.residue_angles,
                )

        return ConformationStatePartial(
            task=observable.task,
            state_res=state_res,
            flex_res=flex_res,
            states_ua_updates=states_ua_updates,
            flexible_ua_updates=flexible_ua_updates,
        )

    def _reduce_state_partials(
        self,
        partials: list[ConformationStatePartial],
    ) -> ConformationStateData:
        """Merge chunk-local state partials into one group-level result.

        Args:
            partials: Chunk-local state partials for one group.

        Returns:
            Group-level state data using deterministic molecule/chunk ordering.
        """
        ordered_partials = sorted(
            partials,
            key=lambda partial: (
                partial.task.molecule_order,
                partial.task.chunk_id,
            ),
        )

        state_res: list[str] = []
        flex_res = 0
        states_ua_updates: dict[UAKey, list[str]] = {}
        flexible_ua_updates: dict[UAKey, int] = {}

        for partial in ordered_partials:
            for key, states in partial.states_ua_updates.items():
                if key not in states_ua_updates:
                    states_ua_updates[key] = list(states)
                    flexible_ua_updates[key] = partial.flexible_ua_updates[key]
                else:
                    states_ua_updates[key].extend(states)
                    flexible_ua_updates[key] = max(
                        flexible_ua_updates[key],
                        partial.flexible_ua_updates[key],
                    )

            state_res.extend(partial.state_res)
            flex_res = max(flex_res, partial.flex_res)

        return ConformationStateData(
            state_res=state_res,
            flex_res=flex_res,
            states_ua_updates=states_ua_updates,
            flexible_ua_updates=flexible_ua_updates,
        )

    @staticmethod
    def _merge_group_state_data(
        state_data: ConformationStateData,
        states_ua: dict[UAKey, list[str]],
        states_res: list[list[str]],
        flexible_ua: dict[UAKey, int],
        flexible_res: list[int],
    ) -> None:
        """Merge one group's state data into final output accumulators.

        Args:
            state_data: Serial conformational state data for one group.
            states_ua: UA state accumulator to mutate.
            states_res: Residue state accumulator to mutate.
            flexible_ua: UA flexible-dihedral accumulator to mutate.
            flexible_res: Residue flexible-dihedral accumulator to mutate.

        Returns:
            None. Mutates the provided accumulators.
        """
        for key, states in state_data.states_ua_updates.items():
            if key not in states_ua:
                states_ua[key] = states
                flexible_ua[key] = state_data.flexible_ua_updates[key]
            else:
                states_ua[key].extend(states)
                flexible_ua[key] = max(
                    flexible_ua[key],
                    state_data.flexible_ua_updates[key],
                )

        states_res.append(state_data.state_res)
        flexible_res.append(state_data.flex_res)

    def _process_conformations_from_angles(
        self,
        peaks: list[Any],
        angles: np.ndarray,
    ) -> tuple[list[str], int]:
        """Assign conformational states from a positive-angle NumPy array.

        Args:
            peaks: Histogram peaks by dihedral.
            angles: Positive-angle array with shape ``(n_frames, n_dihedrals)``.

        Returns:
            Tuple of ``(states, num_flexible)``.
        """
        if angles.size == 0 or angles.shape[1] == 0:
            return [], 0

        padded_peaks, peak_counts = self._pad_peak_values(peaks)
        labels, num_flexible = assign_peak_labels_and_count_flexible(
            angles,
            padded_peaks,
            peak_counts,
        )
        states = self._state_strings_from_labels(labels)
        return states, int(num_flexible)

    @staticmethod
    def _pad_peak_values(peaks: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Convert ragged peak lists into padded arrays for kernels.

        Args:
            peaks: Peak values by dihedral.

        Returns:
            Tuple of ``(padded_peaks, peak_counts)``.
        """
        if not peaks:
            return (
                np.zeros((0, 1), dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )

        max_peaks = max((len(dihedral_peaks) for dihedral_peaks in peaks), default=0)
        max_peaks = max(1, max_peaks)
        padded = np.zeros((len(peaks), max_peaks), dtype=np.float64)
        counts = np.zeros(len(peaks), dtype=np.int64)

        for dihedral_index, dihedral_peaks in enumerate(peaks):
            counts[dihedral_index] = len(dihedral_peaks)
            for peak_index, peak in enumerate(dihedral_peaks):
                padded[dihedral_index, peak_index] = float(peak)

        return padded, counts

    @staticmethod
    def _state_strings_from_labels(labels: np.ndarray) -> list[str]:
        """Convert integer per-frame labels into legacy state strings.

        Args:
            labels: Integer labels with shape ``(n_frames, n_dihedrals)``.

        Returns:
            Legacy state strings, one per frame.
        """
        states: list[str] = []
        number_frames = labels.shape[0]
        num_dihedrals = labels.shape[1]

        for frame_index in range(number_frames):
            state = "".join(
                str(int(labels[frame_index, dihedral_index]))
                for dihedral_index in range(num_dihedrals)
            )
            if state:
                states.append(state)

        return states
