"""Selected-frame dihedral angle observation helpers.

This module contains the frame-aware angle collection logic used by the
conformational state workflow. It preserves the MDAnalysis frame-index contract:
``Dihedral.run(...)`` receives active analysis-universe frame bounds, while the
returned ``results.angles`` array is indexed locally from zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from MDAnalysis.analysis.dihedrals import Dihedral

from CodeEntropy.levels.dihedrals.kernels import wrap_degrees_positive
from CodeEntropy.levels.dihedrals.topology import (
    DihedralTopologyDiscovery,
    MoleculeDihedralTopology,
)
from CodeEntropy.levels.execution.chunks import chunk_frame_indices
from CodeEntropy.trajectory.frames import FrameSelection


@dataclass(frozen=True)
class ConformationChunkTask:
    """Serial conformational work item for one molecule and frame chunk.

    Attributes:
        group_id: Molecule group id.
        molecule_id: Molecule id.
        molecule_order: Position of the molecule within the group.
        chunk_id: Deterministic frame-chunk index.
        frame_indices: Absolute analysis trajectory indices in this chunk.
        frame_selection: FrameSelection covering this chunk.
    """

    group_id: int
    molecule_id: Any
    molecule_order: int
    chunk_id: int
    frame_indices: tuple[int, ...]
    frame_selection: FrameSelection


@dataclass
class DihedralAngleObservable:
    """Chunk-local dihedral angle arrays for one molecule/frame chunk.

    Attributes:
        task: Source molecule/frame chunk task.
        num_residues: Number of residues in the molecule.
        ua_angles_by_residue: Positive-angle arrays by residue index. Each array
            has shape ``(n_chunk_frames, n_dihedrals)``.
        residue_angles: Positive-angle residue-level array with shape
            ``(n_chunk_frames, n_residue_dihedrals)``, or ``None`` when the
            residue level is disabled or has no dihedrals.
    """

    task: ConformationChunkTask
    num_residues: int
    ua_angles_by_residue: dict[int, np.ndarray]
    residue_angles: np.ndarray | None


class DihedralAngleCollector(DihedralTopologyDiscovery):
    """Collect dihedral angle observations from selected trajectory frames."""

    def _build_conformation_chunk_tasks(
        self,
        topologies: list[MoleculeDihedralTopology],
        frame_selection: FrameSelection,
        chunk_size: int,
    ) -> list[ConformationChunkTask]:
        """Build deterministic molecule/frame chunk tasks for conformations.

        Args:
            topologies: Per-molecule conformational topology entries.
            frame_selection: Selected frames in active analysis-universe index
                space.
            chunk_size: Number of selected frames per chunk.

        Returns:
            Conformation chunk tasks ordered by molecule order, then chunk id.
        """
        frame_indices = tuple(int(i) for i in frame_selection.analysis_indices)
        frame_chunks = chunk_frame_indices(list(frame_indices), int(chunk_size))
        tasks: list[ConformationChunkTask] = []

        for topology in topologies:
            for chunk_id, frame_chunk in enumerate(frame_chunks):
                chunk_indices = tuple(int(index) for index in frame_chunk)
                tasks.append(
                    ConformationChunkTask(
                        group_id=topology.group_id,
                        molecule_id=topology.molecule_id,
                        molecule_order=topology.molecule_order,
                        chunk_id=chunk_id,
                        frame_indices=chunk_indices,
                        frame_selection=self._frame_selection_from_chunk(chunk_indices),
                    )
                )

        return tasks

    @staticmethod
    def _frame_selection_from_chunk(frame_indices: tuple[int, ...]) -> FrameSelection:
        """Build a FrameSelection for a selected frame chunk.

        Args:
            frame_indices: Absolute trajectory frame indices in the chunk.

        Returns:
            FrameSelection containing exactly the chunk frame indices.

        Raises:
            ValueError: If the chunk is empty.
        """
        if not frame_indices:
            raise ValueError("Cannot build a frame selection from an empty chunk.")

        return FrameSelection(indices=tuple(int(index) for index in frame_indices))

    def _collect_angle_observable(
        self,
        topology: MoleculeDihedralTopology,
        task: ConformationChunkTask,
        level_list: list[Any],
    ) -> DihedralAngleObservable:
        """Collect chunk-local positive-angle arrays for one molecule.

        Args:
            topology: Static dihedral topology for the molecule.
            task: Molecule/frame chunk task.
            level_list: Enabled hierarchy levels.

        Returns:
            Chunk-local angle observable used by both conformational reductions.
        """
        number_frames = self._analysis_frame_count(task.frame_selection)
        ua_angles_by_residue: dict[int, np.ndarray] = {}
        residue_angles: np.ndarray | None = None

        if "united_atom" in level_list:
            for res_id in range(topology.num_residues):
                dihedrals = topology.ua_dihedrals_by_residue.get(res_id, [])
                if not dihedrals:
                    ua_angles_by_residue[res_id] = np.empty(
                        (number_frames, 0), dtype=np.float64
                    )
                    continue

                dihedral_results = self._run_dihedrals(
                    dihedrals=dihedrals,
                    frame_selection=task.frame_selection,
                )
                ua_angles_by_residue[res_id] = self._extract_positive_angle_array(
                    dihedral_results=dihedral_results,
                    num_dihedrals=len(dihedrals),
                    number_frames=number_frames,
                )

        if "residue" in level_list and topology.residue_dihedrals:
            dihedral_results = self._run_dihedrals(
                dihedrals=topology.residue_dihedrals,
                frame_selection=task.frame_selection,
            )
            residue_angles = self._extract_positive_angle_array(
                dihedral_results=dihedral_results,
                num_dihedrals=len(topology.residue_dihedrals),
                number_frames=number_frames,
            )

        return DihedralAngleObservable(
            task=task,
            num_residues=topology.num_residues,
            ua_angles_by_residue=ua_angles_by_residue,
            residue_angles=residue_angles,
        )

    def _extract_positive_angle_array(
        self,
        dihedral_results: Any,
        num_dihedrals: int,
        number_frames: int,
    ) -> np.ndarray:
        """Extract a positive-angle NumPy array from MDAnalysis results.

        Args:
            dihedral_results: Result of ``Dihedral(...).run(...)``.
            num_dihedrals: Number of dihedrals in the result.
            number_frames: Number of local result frames.

        Returns:
            Positive-angle array with shape ``(number_frames, num_dihedrals)``.
        """
        angles = np.asarray(
            dihedral_results.results.angles[:number_frames, :num_dihedrals],
            dtype=np.float64,
        )
        return wrap_degrees_positive(angles)

    def _run_dihedrals(
        self, dihedrals: list[Any], frame_selection: FrameSelection
    ) -> Any:
        """Run MDAnalysis dihedral analysis over selected analysis frames.

        Args:
            dihedrals: Dihedral AtomGroups.
            frame_selection: Selected trajectory frame selection.

        Returns:
            MDAnalysis Dihedral analysis result.

        Notes:
            ``Dihedral.run(start, stop, step)`` uses absolute active trajectory
            frame bounds. The returned ``results.angles`` array is indexed
            locally from zero.
        """
        if not dihedrals:
            raise ValueError("Cannot run Dihedral analysis with no dihedrals.")

        start, stop, step = self._analysis_run_bounds(frame_selection)
        return Dihedral(dihedrals).run(start=start, stop=stop, step=step)

    @staticmethod
    def _analysis_frame_count(frame_selection: FrameSelection) -> int:
        """Return the number of selected frames.

        Args:
            frame_selection: Selected trajectory frame selection.

        Returns:
            Number of selected frames.
        """
        return frame_selection.n_frames

    @staticmethod
    def _analysis_run_bounds(frame_selection: FrameSelection) -> tuple[int, int, int]:
        """Return MDAnalysis run bounds for selected analysis frames.

        Args:
            frame_selection: Selected trajectory frame selection.

        Returns:
            Tuple of ``(start, stop, step)`` in active analysis-universe index
            space.

        Raises:
            ValueError: If the selection is empty or irregularly spaced.
        """
        start = frame_selection.source_start
        stop = frame_selection.source_stop_exclusive

        if start is None or stop is None:
            raise ValueError("Frame selection is empty.")

        return start, stop, frame_selection.infer_source_step()
