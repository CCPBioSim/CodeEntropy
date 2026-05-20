"""Dihedral state assignment for conformational entropy.

This module converts dihedral angle time series into discrete conformational
state labels. The resulting state labels are used downstream to compute
conformational entropy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from MDAnalysis.analysis.dihedrals import Dihedral
from rich.progress import TaskID

from CodeEntropy.levels.dihedrals import DihedralDefinitions
from CodeEntropy.results.reporter import _RichProgressSink

logger = logging.getLogger(__name__)

UAKey = tuple[int, int]


class ConformationStateBuilder:
    """Build conformational state labels from dihedral angles."""

    def __init__(self, universe_operations: Any) -> None:
        """Initializes the analysis helper.

        Args:
            universe_operations: Object providing helper methods:
                - extract_fragment(data_container, molecule_id)
                - select_atoms(atomgroup, selection_string)
        """
        self._universe_operations = universe_operations
        self._dihedral_definitions = DihedralDefinitions()

    def build_conformational_states(
        self,
        data_container: Any,
        levels: dict[Any, list[str]],
        groups: dict[int, list[Any]],
        bin_width: float,
        conf_type: str,
        progress: _RichProgressSink | None = None,
    ) -> tuple[dict[UAKey, list[str]], list[list[str]], dict[UAKey, int], list[int]]:
        """Build conformational state labels from trajectory dihedrals.

        This method constructs discrete conformational state descriptors used in
        configurational entropy calculations. It supports united-atom (UA) and
        residue-level state generation depending on which hierarchy levels are
        enabled per molecule.

        Progress reporting is optional and UI-agnostic. If a progress sink is
        provided, the method will create a single task and advance it once per
        molecule group.

        Args:
            data_container: MDAnalysis Universe (or compatible container) used to
                extract fragments and compute dihedral time series.
            levels: Mapping of molecule_id -> iterable of enabled level names
                (e.g., ["united_atom", "residue"]).
            groups: Mapping of group_id -> list of molecule_ids.
            bin_width: Histogram bin width in degrees used when identifying peak
                dihedral populations.
            progress: Optional progress sink (e.g., from
                ResultsReporter.progress()). Must expose add_task(), update(),
                and advance().

        Returns:
            tuple: (states_ua, states_res, flexible_ua, flexible_res)

            - states_ua: Dict mapping (group_id, local_residue_id) -> list of state
              labels (strings) across the analyzed trajectory.
            - states_res: Structure indexed by group_id (or equivalent) containing
              residue-level state labels (strings) across the analyzed trajectory.

        Notes:
            - This function advances progress once per group_id.
              helpers as implemented in this module.
        """
        states_ua: dict[UAKey, list[str]] = {}
        states_res: list[Any] = []
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
            return states_ua, states_res

        for group_id in groups.keys():
            molecules = groups[group_id]
            if not molecules:
                if progress is not None and task is not None:
                    progress.update(task, title=f"Group {group_id} (empty)")
                    progress.advance(task)
                continue

            if progress is not None and task is not None:
                progress.update(task, title=f"Group {group_id}")

            peaks_ua, peaks_res = self._identify_peaks(
                data_container=data_container,
                molecules=molecules,
                bin_width=bin_width,
                conf_type=conf_type,
                level_list=levels[molecules[0]],
            )

            self._assign_states(
                data_container=data_container,
                group_id=group_id,
                molecules=molecules,
                level_list=levels[molecules[0]],
                peaks_ua=peaks_ua,
                peaks_res=peaks_res,
                states_ua=states_ua,
                states_res=states_res,
                flexible_ua=flexible_ua,
                flexible_res=flexible_res,
                conf_type=conf_type,
            )

            if progress is not None and task is not None:
                progress.advance(task)

        logger.debug(f"States UA: {states_ua}")
        logger.debug(f"Number of flexible dihedrals UA: {flexible_ua}")
        logger.debug(f"States Res: {states_res}")
        logger.debug(f"Number of flexible dihedrals Res: {flexible_res}")

        return states_ua, states_res, flexible_ua, flexible_res

    def _select_heavy_residue(self, mol: Any, res_id: int) -> Any:
        """Select heavy atoms in a residue by residue index.

        Args:
            mol: Representative molecule AtomGroup.
            res_id: Residue index.

        Returns:
            AtomGroup containing heavy atoms in the residue selection.
        """
        selection1 = mol.residues[res_id].atoms.indices[0]
        selection2 = mol.residues[res_id].atoms.indices[-1]

        res_container = self._universe_operations.select_atoms(
            mol, f"index {selection1}:{selection2}"
        )
        return self._universe_operations.select_atoms(res_container, "prop mass > 1.1")

    def _get_dihedrals(
        self, data_container: Any, level: str, conf_type: str
    ) -> list[Any]:
        """Return dihedral AtomGroups for a container at a given level.

        Args:
            data_container: MDAnalysis container (AtomGroup/Universe).
            level: Either "united_atom" or "residue".

        Returns:
            List of AtomGroups (each representing a dihedral definition).
        """
        atom_groups: list[Any] = []

        if conf_type == "res_bonds":
            atom_groups = self._dihedral_definitions.method_res_bonds(
                data_container=data_container, level=level
            )

        elif conf_type == "res_points":
            atom_groups = self._dihedral_definitions.method_res_points(
                data_container=data_container, level=level
            )

        elif conf_type == "ua_only":
            atom_groups = self._dihedral_definitions.method_ua_only(
                data_container=data_container, level=level
            )

        return atom_groups

    def _identify_peaks(
        self,
        data_container: Any,
        molecules: list[Any],
        bin_width: float,
        conf_type: str,
        level_list: list[Any],
    ) -> list[list[float]]:
        """Identify histogram peaks ("convex turning points") for each dihedral.

        Important:
            This function intentionally preserves the legacy behavior:
            it samples over the full trajectory length for each molecule
            and does not apply start/end/step to the Dihedral run.

        Args:
            data_container: MDAnalysis universe.
            molecules: Molecule ids in the group.
            levels: Molecule levels.
            bin_width: Histogram bin width (degrees).

        Returns:
            List of peaks per dihedral (peak_values[dihedral_index] -> list of peaks).
        """
        rep_mol = self._universe_operations.extract_fragment(
            data_container, molecules[0]
        )
        number_frames = len(rep_mol.trajectory)
        num_residues = len(rep_mol.residues)

        num_dihedrals_ua: list[Any] = [0 for _ in range(num_residues)]
        phi_ua = {}
        phi_res: dict[list, list[float]] = {}
        peaks_ua: list[list[Any]] = [[] for _ in range(num_residues)]
        peaks_res: list[Any] = []

        for molecule in molecules:
            mol = self._universe_operations.extract_fragment(data_container, molecule)

            for level in level_list:
                if level == "united_atom":
                    for res_id in range(num_residues):
                        heavy_res = self._select_heavy_residue(mol, res_id)
                        dihedrals = self._get_dihedrals(
                            data_container=heavy_res,
                            level=level,
                            conf_type=conf_type,
                        )
                        num_dihedrals_ua[res_id] = len(dihedrals)
                        if num_dihedrals_ua[res_id] == 0:
                            # No dihedrals, no peaks
                            phi_ua[res_id] = []

                        else:
                            if res_id not in phi_ua:
                                phi_ua[res_id] = {}
                            dihedral_results = Dihedral(dihedrals).run()
                            phi_ua[res_id] = self._process_dihedral_phi(
                                dihedral_results,
                                num_dihedrals_ua[res_id],
                                number_frames,
                                phi_ua[res_id],
                            )

                elif level == "residue":
                    dihedrals = self._get_dihedrals(
                        data_container=mol,
                        level=level,
                        conf_type=conf_type,
                    )
                    num_dihedrals_res = len(dihedrals)
                    if num_dihedrals_res == 0:
                        # No dihedrals, no peaks
                        phi_res = []

                    else:
                        dihedral_results = Dihedral(dihedrals).run()
                        phi_res = self._process_dihedral_phi(
                            dihedral_results,
                            num_dihedrals_res,
                            number_frames,
                            phi_res,
                        )

        logger.debug(f"phi_ua {phi_ua}")
        logger.debug(f"phi_res {phi_res}")

        for level in level_list:
            if level == "united_atom":
                for res_id in range(num_residues):
                    if phi_ua[res_id] is None:
                        peaks_ua[res_id] = []
                    else:
                        peaks_ua[res_id] = self._process_histogram(
                            num_dihedrals_ua[res_id], phi_ua[res_id], bin_width
                        )

            elif level == "residue":
                if phi_res is None:
                    peaks_res = []
                else:
                    peaks_res = self._process_histogram(
                        num_dihedrals_res, phi_res, bin_width
                    )

        return peaks_ua, peaks_res

    def _process_dihedral_phi(
        self,
        dihedral_results,
        num_dihedrals,
        number_frames,
        phi_values,
    ):
        """
        Find array of dihedral angle values.

        Args:
            dihedral_results: the result of MDAnalysis Dihedrals.run.
            num_dihedrals: the number of dihedrals in the molecule or residue.

        Returns:
            peaks
        """
        for dihedral_index in range(num_dihedrals):
            phi: list[float] = []

            for timestep in range(number_frames):
                value = dihedral_results.results.angles[timestep][dihedral_index]
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
        num_dihedrals,
        phi_values,
        bin_width,
    ):
        """
        Find peaks from array of dihedral angle values.

        Args:
            dihedral_results: the result of MDAnalysis Dihedrals.run.
            num_dihedrals: the number of dihedrals in the molecule or residue.

        Returns:
            peaks
        """
        peak_values = []
        for dihedral_index in range(num_dihedrals):
            phi = phi_values[dihedral_index]
            number_bins = int(360 / bin_width)
            popul, bin_edges = np.histogram(a=phi, bins=number_bins, range=(0, 360))

            logger.debug(f"Histogram: {popul}")

            bin_value = [
                0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(0, len(popul))
            ]

            peaks = self._find_histogram_peaks(popul=popul, bin_value=bin_value)
            peak_values.append(peaks)

            logger.debug(f"Dihedral: {dihedral_index} Peaks: {peaks}")

        return peak_values

    @staticmethod
    def _find_histogram_peaks(
        popul: np.ndarray[Any, Any], bin_value: list[float]
    ) -> list[float]:
        """Return convex turning-point peaks from a histogram.

        The selection of the population of the right adjacent bin takes into
        account that the dihedral angles are circular.

        Args:
           popul: the array of counts for each bin
           bin_value: the array of dihedral angle value at the center of each
              bin.

        Returns:
           peaks: list of values associated with peaks.
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
        conf_type: str,
    ) -> list[str]:
        """Assign discrete state labels for the provided dihedrals.

        Important:
            This function intentionally preserves the legacy behavior:
            it samples over the full trajectory length for each molecule
            and does not apply start/end/step to the Dihedral run.

        Args:
            data_container: MDAnalysis universe.
            molecules: Molecule ids in the group.
            dihedrals: Dihedral AtomGroups.
            peaks: Peaks per dihedral.

        Returns:
            List of state labels (strings).
        """
        rep_mol = self._universe_operations.extract_fragment(
            data_container, molecules[0]
        )
        number_frames = len(rep_mol.trajectory)
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
                        dihedrals = self._get_dihedrals(heavy_res, level, conf_type)
                        num_dihedrals = len(dihedrals)
                        if num_dihedrals == 0:
                            # No dihedrals, no conformations
                            states_ua[key] = []
                            flexible_ua[key] = 0
                        else:
                            dihedral_results = Dihedral(dihedrals).run()
                            states, flexible = self._process_conformations(
                                peaks_ua[res_id],
                                dihedral_results,
                                num_dihedrals,
                                number_frames,
                            )
                            if key not in states_ua:
                                states_ua[key] = states
                                flexible_ua[key] = flexible
                            else:
                                states_ua[key].extend(states)
                                flexible_ua[key] = max(flexible_ua[key], flexible)

                if level == "residue":
                    dihedrals = self._get_dihedrals(mol, level, conf_type)
                    num_dihedrals = len(dihedrals)
                    if num_dihedrals == 0:
                        # No dihedrals, no conformations
                        state_res = []
                    else:
                        dihedral_results = Dihedral(dihedrals).run()
                        states, flexible = self._process_conformations(
                            peaks_res,
                            dihedral_results,
                            num_dihedrals,
                            number_frames,
                        )
                        state_res.extend(states)
                        flex_res = max(flex_res, flexible)

                    states_res.append(state_res)
                    if conf_type == "ua_only":
                        flex_res = 0
                    flexible_res.append(flex_res)

    def _process_conformations(
        self, peaks, dihedral_results, num_dihedrals, number_frames
    ):
        """
        Find conformations

        Args:
            peaks: Histogram peaks.
            num_dihedrals: Number of dihedral angles in the molecule or residue.
        Returns:
            conformations
        """
        states: list[list[Any]] = []
        conformations: list[list[Any]] = []
        num_flexible = 0
        for dihedral_index in range(num_dihedrals):
            conformation: list[Any] = []

            # Check for flexible dihedrals
            #      if len(peaks[dihedral_index]) > 1:
            #          num_flexible += 1

            # Get conformations
            for timestep in range(number_frames):
                value = dihedral_results.results.angles[timestep][dihedral_index]
                # We want postive values in range 0 to 360 to make
                # the peak assignment.
                # works using the fact that dihedrals have circular symmetry
                # (i.e. -15 degrees = +345 degrees)
                if value < 0:
                    value += 360

                # Find the peak closest to the dihedral value
                distances = [abs(value - peak) for peak in peaks[dihedral_index]]
                conformation.append(np.argmin(distances))

            unique = np.unique(conformation)
            if len(unique) > 1:
                num_flexible += 1

            conformations.append(conformation)

        # Concatenate all the dihedrals in the molecule into the state
        # for the frame.
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
