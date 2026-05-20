"""Dihedral state assignment for conformational entropy.

This module converts selected-frame dihedral angle time series into discrete
conformational state labels. The resulting state labels are used downstream to
compute configurational entropy.

Frame-index contract:
    - ``FrameSelection.analysis_indices`` are used for MDAnalysis trajectory access
      in the active analysis universe.
    - ``Dihedral(...).run(start, stop, step)`` uses frame bounds in the active
      analysis-universe index space.
    - ``dihedral_results.results.angles`` is always indexed locally from zero.
      Never use an absolute/source frame index directly into that result array.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from MDAnalysis.analysis.dihedrals import Dihedral
from rich.progress import TaskID

from CodeEntropy.results.reporter import _RichProgressSink
from CodeEntropy.trajectory.frames import FrameSelection

logger = logging.getLogger(__name__)

UAKey = tuple[int, int]


class ConformationStateBuilder:
    """Build conformational state labels from selected-frame dihedral angles."""

    def __init__(self, universe_operations: Any) -> None:
        """Initialize the analysis helper.

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
    ) -> tuple[dict[UAKey, list[str]], list[list[str]], dict[UAKey, int], list[int]]:
        """Build conformational state labels from selected trajectory frames.

        Args:
            data_container: MDAnalysis Universe or compatible container used to
                extract fragments and compute dihedral time series.
            levels: Mapping of molecule id to enabled level names.
            groups: Mapping of group id to molecule ids.
            bin_width: Histogram bin width in degrees used when identifying peak
                dihedral populations.
            frame_selection: FrameSelection controlling which frames are analysed.
                During the current migration stage, ``analysis_indices`` are local
                indices into the physically frame-sliced analysis universe.
            progress: Optional progress sink.

        Returns:
            Tuple ``(states_ua, states_res, flexible_ua, flexible_res)``.
        """
        number_groups = len(groups)
        states_ua: dict[UAKey, list[str]] = {}
        states_res: list[list[str]] = [[] for _ in range(number_groups)]
        flexible_ua: dict[UAKey, int] = {}
        flexible_res: list[int] = []

        task: TaskID | None = None
        if progress is not None:
            total = max(1, len(groups))
            task = progress.add_task(
                "[green]Conformational states",
                total=total,
                title="Initializing",
            )

        if not groups:
            if progress is not None and task is not None:
                progress.update(task, title="No groups")
                progress.advance(task)
            return states_ua, states_res, flexible_ua, flexible_res

        for group_id in groups.keys():
            molecules = groups[group_id]
            if not molecules:
                if progress is not None and task is not None:
                    progress.update(task, title=f"Group {group_id} (empty)")
                    progress.advance(task)
                continue

            if progress is not None and task is not None:
                progress.update(task, title=f"Group {group_id}")

            level_list = levels[molecules[0]]

            peaks_ua, peaks_res = self._identify_peaks(
                data_container=data_container,
                molecules=molecules,
                bin_width=bin_width,
                level_list=level_list,
                frame_selection=frame_selection,
            )

            self._assign_states(
                data_container=data_container,
                group_id=group_id,
                molecules=molecules,
                level_list=level_list,
                peaks_ua=peaks_ua,
                peaks_res=peaks_res,
                states_ua=states_ua,
                states_res=states_res,
                flexible_ua=flexible_ua,
                flexible_res=flexible_res,
                frame_selection=frame_selection,
            )

            if progress is not None and task is not None:
                progress.advance(task)

        logger.debug("States UA: %s", states_ua)
        logger.debug("Number of flexible dihedrals UA: %s", flexible_ua)
        logger.debug("States Res: %s", states_res)
        logger.debug("Number of flexible dihedrals Res: %s", flexible_res)

        return states_ua, states_res, flexible_ua, flexible_res

    def _select_heavy_residue(self, mol: Any, res_id: int) -> Any:
        """Select heavy atoms in a residue by residue index.

        Args:
            mol: Representative molecule AtomGroup.
            res_id: Local residue index.

        Returns:
            AtomGroup containing heavy atoms in the residue selection.
        """
        selection1 = mol.residues[res_id].atoms.indices[0]
        selection2 = mol.residues[res_id].atoms.indices[-1]

        res_container = self._universe_operations.select_atoms(
            mol, f"index {selection1}:{selection2}"
        )
        return self._universe_operations.select_atoms(res_container, "prop mass > 1.1")

    def _get_dihedrals(self, data_container: Any, level: str) -> list[Any]:
        """Return dihedral AtomGroups for a container at a given level.

        Args:
            data_container: MDAnalysis container.
            level: Either ``"united_atom"`` or ``"residue"``.

        Returns:
            List of AtomGroups, each representing a dihedral definition.
        """
        atom_groups: list[Any] = []

        if level == "united_atom":
            for dihedral in data_container.dihedrals:
                atom_groups.append(dihedral.atoms)

        if level == "residue":
            num_residues = len(data_container.residues)
            if num_residues >= 4:
                for residue in range(4, num_residues + 1):
                    atom1 = data_container.select_atoms(
                        f"resindex {residue - 4} and bonded resindex {residue - 3}"
                    )
                    atom2 = data_container.select_atoms(
                        f"resindex {residue - 3} and bonded resindex {residue - 4}"
                    )
                    atom3 = data_container.select_atoms(
                        f"resindex {residue - 2} and bonded resindex {residue - 1}"
                    )
                    atom4 = data_container.select_atoms(
                        f"resindex {residue - 1} and bonded resindex {residue - 2}"
                    )
                    atom_groups.append(atom1 + atom2 + atom3 + atom4)

        logger.debug("Level: %s, Dihedrals: %s", level, atom_groups)
        return atom_groups

    def _identify_peaks(
        self,
        data_container: Any,
        molecules: list[Any],
        bin_width: float,
        level_list: list[Any],
        frame_selection: FrameSelection,
    ) -> tuple[list[list[Any]], list[Any]]:
        """Identify histogram peaks for each selected-frame dihedral series.

        Args:
            data_container: MDAnalysis universe.
            molecules: Molecule ids in the group.
            bin_width: Histogram bin width in degrees.
            level_list: Enabled hierarchy levels for the representative molecule.
            frame_selection: Selected frames in the active analysis-universe index
                space.

        Returns:
            Tuple of ``(peaks_ua, peaks_res)``.
        """
        rep_mol = self._universe_operations.extract_fragment(
            data_container, molecules[0]
        )
        number_frames = self._analysis_frame_count(frame_selection)
        num_residues = len(rep_mol.residues)

        num_dihedrals_ua: list[int] = [0 for _ in range(num_residues)]
        phi_ua: dict[int, Any] = {}
        phi_res: dict[int, list[float]] | list[Any] = {}
        peaks_ua: list[list[Any]] = [[] for _ in range(num_residues)]
        peaks_res: list[Any] = []
        num_dihedrals_res = 0

        for molecule in molecules:
            mol = self._universe_operations.extract_fragment(data_container, molecule)

            for level in level_list:
                if level == "united_atom":
                    for res_id in range(num_residues):
                        heavy_res = self._select_heavy_residue(mol, res_id)
                        dihedrals = self._get_dihedrals(heavy_res, level)
                        num_dihedrals_ua[res_id] = len(dihedrals)

                        if num_dihedrals_ua[res_id] == 0:
                            phi_ua[res_id] = []
                            continue

                        if res_id not in phi_ua or isinstance(phi_ua[res_id], list):
                            phi_ua[res_id] = {}

                        dihedral_results = self._run_dihedrals(
                            dihedrals=dihedrals,
                            frame_selection=frame_selection,
                        )
                        phi_ua[res_id] = self._process_dihedral_phi(
                            dihedral_results=dihedral_results,
                            num_dihedrals=num_dihedrals_ua[res_id],
                            number_frames=number_frames,
                            phi_values=phi_ua[res_id],
                        )

                elif level == "residue":
                    dihedrals = self._get_dihedrals(mol, level)
                    num_dihedrals_res = len(dihedrals)

                    if num_dihedrals_res == 0:
                        phi_res = []
                        continue

                    if isinstance(phi_res, list):
                        phi_res = {}

                    dihedral_results = self._run_dihedrals(
                        dihedrals=dihedrals,
                        frame_selection=frame_selection,
                    )
                    phi_res = self._process_dihedral_phi(
                        dihedral_results=dihedral_results,
                        num_dihedrals=num_dihedrals_res,
                        number_frames=number_frames,
                        phi_values=phi_res,
                    )

        logger.debug("phi_ua %s", phi_ua)
        logger.debug("phi_res %s", phi_res)

        for level in level_list:
            if level == "united_atom":
                for res_id in range(num_residues):
                    phi_values = phi_ua.get(res_id)
                    if not phi_values:
                        peaks_ua[res_id] = []
                    else:
                        peaks_ua[res_id] = self._process_histogram(
                            num_dihedrals=num_dihedrals_ua[res_id],
                            phi_values=phi_values,
                            bin_width=bin_width,
                        )

            elif level == "residue":
                if not phi_res:
                    peaks_res = []
                else:
                    peaks_res = self._process_histogram(
                        num_dihedrals=num_dihedrals_res,
                        phi_values=phi_res,
                        bin_width=bin_width,
                    )

        return peaks_ua, peaks_res

    def _process_dihedral_phi(
        self,
        dihedral_results: Any,
        num_dihedrals: int,
        number_frames: int,
        phi_values: dict[int, list[float]],
    ) -> dict[int, list[float]]:
        """Collect positive-angle dihedral values from a local result array.

        Args:
            dihedral_results: Result of ``MDAnalysis.analysis.dihedrals.Dihedral``.
            num_dihedrals: Number of dihedrals in the result.
            number_frames: Number of local frames in ``dihedral_results``.
            phi_values: Existing accumulator mapping dihedral index to values.

        Returns:
            Updated ``phi_values`` accumulator.

        Notes:
            ``dihedral_results.results.angles`` is indexed locally from zero.
        """
        for dihedral_index in range(num_dihedrals):
            phi: list[float] = []

            for local_i in range(number_frames):
                value = dihedral_results.results.angles[local_i][dihedral_index]
                if value < 0:
                    value += 360
                phi.append(float(value))

            if dihedral_index not in phi_values:
                phi_values[dihedral_index] = phi
            else:
                phi_values[dihedral_index].extend(phi)

        return phi_values

    def _process_histogram(
        self,
        num_dihedrals: int,
        phi_values: dict[int, list[float]],
        bin_width: float,
    ) -> list[Any]:
        """Find histogram peaks from dihedral angle values.

        Args:
            num_dihedrals: Number of dihedrals.
            phi_values: Mapping from dihedral index to angle values.
            bin_width: Histogram bin width in degrees.

        Returns:
            List of peak lists, one per dihedral.
        """
        peak_values = []
        for dihedral_index in range(num_dihedrals):
            phi = phi_values[dihedral_index]
            number_bins = int(360 / bin_width)
            popul, bin_edges = np.histogram(a=phi, bins=number_bins, range=(0, 360))

            logger.debug("Histogram: %s", popul)

            bin_value = [
                0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(0, len(popul))
            ]

            peaks = self._find_histogram_peaks(popul=popul, bin_value=bin_value)
            peak_values.append(peaks)

            logger.debug("Dihedral: %s Peaks: %s", dihedral_index, peaks)

        return peak_values

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

    def _assign_states(
        self,
        data_container: Any,
        group_id: int,
        molecules: list[Any],
        level_list: list[Any],
        peaks_ua: list[list[Any]],
        peaks_res: list[Any],
        states_ua: Any,
        states_res: Any,
        flexible_ua: Any,
        flexible_res: Any,
        frame_selection: FrameSelection,
    ) -> None:
        """Assign discrete state labels for selected-frame dihedrals.

        Args:
            data_container: MDAnalysis universe.
            group_id: Molecule group id.
            molecules: Molecule ids in the group.
            level_list: Enabled hierarchy levels.
            peaks_ua: UA-level peaks by residue.
            peaks_res: Residue-level peaks.
            states_ua: UA state accumulator.
            states_res: Residue state accumulator.
            flexible_ua: UA flexible-dihedral accumulator.
            flexible_res: Residue flexible-dihedral accumulator.
            frame_selection: Selected frames in the active analysis-universe index
                space.

        Returns:
            None. Mutates the provided state/flexible accumulators.
        """
        rep_mol = self._universe_operations.extract_fragment(
            data_container, molecules[0]
        )
        number_frames = self._analysis_frame_count(frame_selection)
        num_residues = len(rep_mol.residues)

        state_res = []
        flex_res = 0

        for molecule in molecules:
            mol = self._universe_operations.extract_fragment(data_container, molecule)

            for level in level_list:
                if level == "united_atom":
                    for res_id in range(num_residues):
                        key = (group_id, res_id)
                        heavy_res = self._select_heavy_residue(mol, res_id)
                        dihedrals = self._get_dihedrals(heavy_res, level)
                        num_dihedrals = len(dihedrals)

                        if num_dihedrals == 0:
                            states_ua[key] = []
                            flexible_ua[key] = 0
                            continue

                        dihedral_results = self._run_dihedrals(
                            dihedrals=dihedrals,
                            frame_selection=frame_selection,
                        )
                        states, flexible = self._process_conformations(
                            peaks=peaks_ua[res_id],
                            dihedral_results=dihedral_results,
                            num_dihedrals=num_dihedrals,
                            number_frames=number_frames,
                        )

                        if key not in states_ua:
                            states_ua[key] = states
                            flexible_ua[key] = flexible
                        else:
                            states_ua[key].extend(states)
                            flexible_ua[key] = max(flexible_ua[key], flexible)

                if level == "residue":
                    dihedrals = self._get_dihedrals(mol, level)
                    num_dihedrals = len(dihedrals)

                    if num_dihedrals == 0:
                        state_res = []
                        continue

                    dihedral_results = self._run_dihedrals(
                        dihedrals=dihedrals,
                        frame_selection=frame_selection,
                    )
                    states, flexible = self._process_conformations(
                        peaks=peaks_res,
                        dihedral_results=dihedral_results,
                        num_dihedrals=num_dihedrals,
                        number_frames=number_frames,
                    )
                    state_res.extend(states)
                    flex_res = max(flex_res, flexible)

        states_res.append(state_res)
        flexible_res.append(flex_res)

    def _process_conformations(
        self,
        peaks: list[Any],
        dihedral_results: Any,
        num_dihedrals: int,
        number_frames: int,
    ) -> tuple[list[str], int]:
        """Assign conformational state labels from local dihedral results.

        Args:
            peaks: Histogram peaks.
            dihedral_results: Result of ``Dihedral(...).run(...)``.
            num_dihedrals: Number of dihedrals.
            number_frames: Number of local result frames.

        Returns:
            Tuple of ``(states, num_flexible)``.

        Notes:
            ``dihedral_results.results.angles`` is indexed locally from zero.
        """
        states: list[str] = []
        conformations: list[list[Any]] = []
        num_flexible = 0

        for dihedral_index in range(num_dihedrals):
            conformation: list[Any] = []

            for local_i in range(number_frames):
                value = dihedral_results.results.angles[local_i][dihedral_index]
                if value < 0:
                    value += 360

                distances = [abs(value - peak) for peak in peaks[dihedral_index]]
                conformation.append(np.argmin(distances))

            unique = np.unique(conformation)
            if len(unique) > 1:
                num_flexible += 1

            conformations.append(conformation)

        mol_states = [
            state
            for state in (
                "".join(str(int(conformations[d][f])) for d in range(num_dihedrals))
                for f in range(number_frames)
            )
            if state
        ]

        states.extend(mol_states)

        return states, num_flexible

    def _run_dihedrals(self, dihedrals: list[Any], frame_selection: FrameSelection):
        """Run MDAnalysis dihedral analysis over selected analysis frames.

        Args:
            dihedrals: Dihedral AtomGroups.
            frame_selection: FrameSelection for the active analysis universe.

        Returns:
            MDAnalysis Dihedral analysis result.

        Notes:
            The returned ``results.angles`` array is indexed locally from zero.
        """
        if not dihedrals:
            raise ValueError("Cannot run Dihedral analysis with no dihedrals.")

        start, stop, step = self._analysis_run_bounds(frame_selection)
        return Dihedral(dihedrals).run(start=start, stop=stop, step=step)

    @staticmethod
    def _analysis_frame_count(frame_selection: FrameSelection) -> int:
        """Return the number of active analysis frames."""
        return len(frame_selection.analysis_indices)

    @staticmethod
    def _analysis_run_bounds(frame_selection: FrameSelection) -> tuple[int, int, int]:
        """Return MDAnalysis run bounds for the active analysis universe.

        Args:
            frame_selection: FrameSelection.

        Returns:
            Tuple of ``(start, stop, step)`` in analysis-index space.

        Raises:
            ValueError: If the selection is empty.
        """
        indices = frame_selection.analysis_indices
        if not indices:
            raise ValueError("Frame selection is empty.")

        start = indices[0]
        stop = indices[-1] + 1
        step = frame_selection.infer_analysis_step()
        return start, stop, step
