"""Public conformational-state builder for dihedral analysis.

This module keeps the stable ``ConformationStateBuilder`` entry point used by
``ConformationDAG`` while the implementation is split across domain-specific
helpers for topology discovery, angle observation, peak detection, and state
assignment.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.progress import TaskID

from CodeEntropy.levels.dihedrals.peak_detection import ConformationPeakDetector
from CodeEntropy.levels.dihedrals.state_assignment import (
    ConformationStateAssigner,
    UAKey,
)
from CodeEntropy.results.reporter import _RichProgressSink
from CodeEntropy.trajectory.frames import FrameSelection

logger = logging.getLogger(__name__)


class ConformationStateBuilder(ConformationPeakDetector, ConformationStateAssigner):
    """Build conformational state labels from selected-frame dihedral angles."""

    def __init__(self, universe_operations: Any) -> None:
        """Initialise the analysis helper.

        Args:
            universe_operations: Object providing helper methods:
                - extract_fragment(data_container, molecule_id)
                - select_atoms(atomgroup, selection_string)
        """
        self._universe_operations = universe_operations

    def build_conformational_states(
        self,
        data_container: Any,
        levels: dict[Any, list[str]],
        groups: dict[int, list[Any]],
        bin_width: float,
        frame_selection: FrameSelection,
        progress: _RichProgressSink | None = None,
        chunk_size: int | None = None,
    ) -> tuple[dict[UAKey, list[str]], list[list[str]], dict[UAKey, int], list[int]]:
        """Build conformational state labels from selected trajectory frames.

        Args:
            data_container: MDAnalysis Universe or compatible container used to
                extract fragments and compute dihedral time series.
            levels: Mapping of molecule id to enabled level names.
            groups: Mapping of group id to molecule ids.
            bin_width: Histogram bin width in degrees used when identifying peak
                dihedral populations.
            frame_selection: FrameSelection controlling which absolute frames are
                analysed.
            progress: Optional progress sink.
            chunk_size: Optional internal frame chunk size. When omitted, the
                full selected-frame range is processed as a single chunk.

        Returns:
            Tuple ``(states_ua, states_res, flexible_ua, flexible_res)``.
        """
        if chunk_size is None:
            chunk_size = max(1, int(frame_selection.n_frames))

        return self._build_conformational_states_serial_chunked(
            data_container=data_container,
            levels=levels,
            groups=groups,
            bin_width=bin_width,
            frame_selection=frame_selection,
            chunk_size=chunk_size,
            progress=progress,
        )

    def _build_conformational_states_serial_chunked(
        self,
        data_container: Any,
        levels: dict[Any, list[str]],
        groups: dict[int, list[Any]],
        bin_width: float,
        frame_selection: FrameSelection,
        chunk_size: int,
        progress: _RichProgressSink | None = None,
    ) -> tuple[dict[UAKey, list[str]], list[list[str]], dict[UAKey, int], list[int]]:
        """Build conformational states with serial frame-chunk map-reduce.

        Args:
            data_container: MDAnalysis universe.
            levels: Mapping of molecule id to enabled level names.
            groups: Mapping of group id to molecule ids.
            bin_width: Histogram bin width in degrees.
            frame_selection: Selected absolute trajectory frames.
            chunk_size: Number of selected frames per chunk.
            progress: Optional progress sink.

        Returns:
            Tuple ``(states_ua, states_res, flexible_ua, flexible_res)``.

        Raises:
            ValueError: If ``chunk_size`` is less than one.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        number_groups = len(groups)
        states_ua: dict[UAKey, list[str]] = {}
        states_res: list[list[str]] = [[] for _ in range(number_groups)]
        flexible_ua: dict[UAKey, int] = {}
        flexible_res: list[int] = []

        task: TaskID | None = None
        if progress is not None:
            task = progress.add_task(
                "[green]Conformational states",
                total=max(1, len(groups)),
                title="Initializing",
            )

        if not groups:
            if progress is not None and task is not None:
                progress.update(task, title="No groups")
                progress.advance(task)

            return states_ua, states_res, flexible_ua, flexible_res

        for group_id, molecules in groups.items():
            if not molecules:
                if progress is not None and task is not None:
                    progress.update(task, title=f"Group {group_id} (empty)")
                    progress.advance(task)

                continue

            if progress is not None and task is not None:
                progress.update(task, title=f"Group {group_id}")

            level_list = levels[molecules[0]]

            topologies = self._discover_group_dihedral_topology(
                data_container=data_container,
                group_id=group_id,
                molecules=molecules,
                level_list=level_list,
            )
            tasks = self._build_conformation_chunk_tasks(
                topologies=topologies,
                frame_selection=frame_selection,
                chunk_size=chunk_size,
            )
            topology_by_order = {
                topology.molecule_order: topology for topology in topologies
            }

            observables = [
                self._collect_angle_observable(
                    topology=topology_by_order[task_item.molecule_order],
                    task=task_item,
                    level_list=level_list,
                )
                for task_item in tasks
            ]

            peak_data = self._reduce_angle_observables_to_peak_data(
                observables=observables,
                level_list=level_list,
                bin_width=bin_width,
            )

            state_partials = [
                self._assign_state_partial_from_observable(
                    observable=observable,
                    topology=topology_by_order[observable.task.molecule_order],
                    level_list=level_list,
                    peaks_ua=peak_data.peaks_ua,
                    peaks_res=peak_data.peaks_res,
                )
                for observable in observables
            ]

            state_data = self._reduce_state_partials(state_partials)

            self._merge_group_state_data(
                state_data=state_data,
                states_ua=states_ua,
                states_res=states_res,
                flexible_ua=flexible_ua,
                flexible_res=flexible_res,
            )

            if progress is not None and task is not None:
                progress.advance(task)

        logger.debug("States UA: %s", states_ua)
        logger.debug("Number of flexible dihedrals UA: %s", flexible_ua)
        logger.debug("States Res: %s", states_res)
        logger.debug("Number of flexible dihedrals Res: %s", flexible_res)

        return states_ua, states_res, flexible_ua, flexible_res
