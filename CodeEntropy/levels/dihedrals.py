"""Dihedral state assignment for conformational entropy.

This module converts dihedral angle time series into discrete conformational
state labels. The resulting state labels are used downstream to compute
conformational entropy.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from MDAnalysis.analysis.dihedrals import Dihedral
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logger = logging.getLogger(__name__)

UAKey = Tuple[int, int]


class ConformationStateBuilder:
    """Build conformational state labels from dihedral angles."""

    def __init__(self, universe_operations=None):
        """Initializes the analysis helper.

        Args:
            universe_operations: Object providing helper methods:
                - extract_fragment(data_container, molecule_id)
                - select_atoms(atomgroup, selection_string)
        """
        self._universe_operations = universe_operations

    def build_conformational_states(
        self,
        data_container,
        levels,
        groups,
        start: int,
        end: int,
        step: int,
        bin_width: float,
        progress: object | None = None,
    ):
        """Build conformational state labels from trajectory dihedrals.

        This method constructs discrete conformational state descriptors used in
        configurational entropy calculations. It supports united-atom (UA) and
        residue-level state generation depending on which hierarchy levels are
        enabled per molecule.

        Progress reporting is optional and UI-agnostic: if a progress sink is
        provided, the method will create a single task and advance it once per
        molecule group.

        Args:
            data_container: MDAnalysis Universe (or compatible container) used to
                extract fragments and compute dihedral time series.
            levels: Mapping of molecule_id -> iterable of enabled level names
                (e.g., ["united_atom", "residue"]).
            groups: Mapping of group_id -> list of molecule_ids.
            start: Inclusive start frame index.
            end: Exclusive end frame index.
            step: Frame stride.
            bin_width: Histogram bin width in degrees used when identifying peak
                dihedral populations.
            progress: Optional progress sink (e.g., from ResultsReporter.progress()).
                Must expose add_task(), update(), and advance().

        Returns:
            Tuple of:
                states_ua: Dict mapping (group_id, local_residue_id) -> list of state
                    labels (strings) across the analyzed trajectory.
                states_res: List-like structure indexed by group_id (or equivalent)
                    containing residue-level state labels (strings) across the
                    analyzed trajectory.

        Notes:
            - This function advances progress once per group_id.
            - Frame slicing arguments (start/end/step) are forwarded to downstream
            helpers as implemented in this module.
        """
        number_groups = len(groups)
        states_ua: Dict[UAKey, List[str]] = {}
        states_res: List[List[str]] = [None] * number_groups

        task = None
        if progress is not None:
            total = max(1, len(groups))
            task = progress.add_task(
                "[green]Conformational states",
                total=total,
                title="Initializing",
            )

        if not groups:
            if task is not None:
                progress.update(task, title="No groups")
                progress.advance(task)
            return states_ua, states_res

        for group_id in groups.keys():
            molecules = groups[group_id]
            if not molecules:
                if task is not None:
                    progress.update(task, title=f"Group {group_id} (empty)")
                    progress.advance(task)
                continue

            if task is not None:
                progress.update(task, title=f"Group {group_id}")

            mol = self._universe_operations.extract_fragment(
                data_container, molecules[0]
            )

            dihedrals_ua, dihedrals_res = self._collect_dihedrals_for_group(
                mol=mol,
                level_list=levels[molecules[0]],
            )

            peaks_ua, peaks_res = self._collect_peaks_for_group(
                data_container=data_container,
                molecules=molecules,
                dihedrals_ua=dihedrals_ua,
                dihedrals_res=dihedrals_res,
                bin_width=bin_width,
                start=start,
                end=end,
                step=step,
                level_list=levels[molecules[0]],
            )

            self._assign_states_for_group(
                data_container=data_container,
                group_id=group_id,
                molecules=molecules,
                dihedrals_ua=dihedrals_ua,
                peaks_ua=peaks_ua,
                dihedrals_res=dihedrals_res,
                peaks_res=peaks_res,
                start=start,
                end=end,
                step=step,
                level_list=levels[molecules[0]],
                states_ua=states_ua,
                states_res=states_res,
            )

            if task is not None:
                progress.advance(task)

        return states_ua, states_res

    def _collect_dihedrals_for_group(self, mol, level_list):
        """Collect UA and residue dihedral AtomGroups for a group.

        Args:
            mol: Representative molecule AtomGroup.
            level_list: List of enabled hierarchy levels.

        Returns:
            Tuple:
                dihedrals_ua: List of per-residue dihedral AtomGroups.
                dihedrals_res: List of residue-level dihedral AtomGroups.
        """
        num_residues = len(mol.residues)
        dihedrals_ua: List[List] = [[] for _ in range(num_residues)]
        dihedrals_res: List = []

        for level in level_list:
            if level == "united_atom":
                for res_id in range(num_residues):
                    heavy_res = self._select_heavy_residue(mol, res_id)
                    dihedrals_ua[res_id] = self._get_dihedrals(heavy_res, level)

            elif level == "residue":
                dihedrals_res = self._get_dihedrals(mol, level)

        return dihedrals_ua, dihedrals_res

    def _select_heavy_residue(self, mol, res_id: int):
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

    def _get_dihedrals(self, data_container, level: str):
        """Return dihedral AtomGroups for a container at a given level.

        Args:
            data_container: MDAnalysis container (AtomGroup/Universe).
            level: Either "united_atom" or "residue".

        Returns:
            List of AtomGroups (each representing a dihedral definition).
        """
        atom_groups = []

        if level == "united_atom":
            dihedrals = data_container.dihedrals
            for d in dihedrals:
                atom_groups.append(d.atoms)

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

    def _collect_peaks_for_group(
        self,
        data_container,
        molecules,
        dihedrals_ua,
        dihedrals_res,
        bin_width,
        start,
        end,
        step,
        level_list,
    ):
        """Compute histogram peaks for UA and residue dihedral sets.

        Returns:
            Tuple:
                peaks_ua: list of peaks per residue
                (each item is list-of-peaks per dihedral)
                peaks_res: list-of-peaks per dihedral for residue level (or [])
        """
        peaks_ua = [{} for _ in range(len(dihedrals_ua))]
        peaks_res = {}

        for level in level_list:
            if level == "united_atom":
                for res_id in range(len(dihedrals_ua)):
                    if len(dihedrals_ua[res_id]) == 0:
                        peaks_ua[res_id] = []
                    else:
                        peaks_ua[res_id] = self._identify_peaks(
                            data_container=data_container,
                            molecules=molecules,
                            dihedrals=dihedrals_ua[res_id],
                            bin_width=bin_width,
                            start=start,
                            end=end,
                            step=step,
                        )

            elif level == "residue":
                if len(dihedrals_res) == 0:
                    peaks_res = []
                else:
                    peaks_res = self._identify_peaks(
                        data_container=data_container,
                        molecules=molecules,
                        dihedrals=dihedrals_res,
                        bin_width=bin_width,
                        start=start,
                        end=end,
                        step=step,
                    )

        return peaks_ua, peaks_res

    def _identify_peaks(
        self,
        data_container,
        molecules,
        dihedrals,
        bin_width,
        start,
        end,
        step,
    ):
        """Identify histogram peaks ("convex turning points") for each dihedral.

        Important:
            This function intentionally preserves the legacy behavior:
            it samples over the full trajectory length for each molecule
            and does not apply start/end/step to the Dihedral run.

        Args:
            data_container: MDAnalysis universe.
            molecules: Molecule ids in the group.
            dihedrals: Dihedral AtomGroups.
            bin_width: Histogram bin width (degrees).
            start: Unused in legacy sampling.
            end: Unused in legacy sampling.
            step: Unused in legacy sampling.

        Returns:
            List of peaks per dihedral (peak_values[dihedral_index] -> list of peaks).
        """
        peak_values = [] * len(dihedrals)

        for dihedral_index in range(len(dihedrals)):
            phi = []

            for molecule in molecules:
                mol = self._universe_operations.extract_fragment(
                    data_container, molecule
                )
                number_frames = len(mol.trajectory)

                dihedral_results = Dihedral(dihedrals).run()

                for timestep in range(number_frames):
                    value = dihedral_results.results.angles[timestep][dihedral_index]
                    if value < 0:
                        value += 360
                    phi.append(value)

            number_bins = int(360 / bin_width)
            popul, bin_edges = np.histogram(a=phi, bins=number_bins, range=(0, 360))
            bin_value = [
                0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(0, len(popul))
            ]

            peaks = self._find_histogram_peaks(popul=popul, bin_value=bin_value)
            peak_values.append(peaks)

            logger.debug("Dihedral: %s, Peak Values: %s", dihedral_index, peak_values)

        return peak_values

    @staticmethod
    def _find_histogram_peaks(popul, bin_value):
        """Return convex turning-point peaks from a histogram."""
        number_bins = len(popul)
        peaks = []

        for bin_index in range(number_bins):
            if popul[bin_index] == 0:
                continue

            if bin_index == number_bins - 1:
                if (
                    popul[bin_index] >= popul[bin_index - 1]
                    and popul[bin_index] >= popul[0]
                ):
                    peaks.append(bin_value[bin_index])
            else:
                if (
                    popul[bin_index] >= popul[bin_index - 1]
                    and popul[bin_index] >= popul[bin_index + 1]
                ):
                    peaks.append(bin_value[bin_index])

        return peaks

    def _assign_states_for_group(
        self,
        data_container,
        group_id,
        molecules,
        dihedrals_ua,
        peaks_ua,
        dihedrals_res,
        peaks_res,
        start,
        end,
        step,
        level_list,
        states_ua,
        states_res,
    ):
        """Assign UA and residue states for a group into output containers."""
        for level in level_list:
            if level == "united_atom":
                for res_id in range(len(dihedrals_ua)):
                    key = (group_id, res_id)
                    if len(dihedrals_ua[res_id]) == 0:
                        states_ua[key] = []
                    else:
                        states_ua[key] = self._assign_states(
                            data_container=data_container,
                            molecules=molecules,
                            dihedrals=dihedrals_ua[res_id],
                            peaks=peaks_ua[res_id],
                            start=start,
                            end=end,
                            step=step,
                        )

            elif level == "residue":
                if len(dihedrals_res) == 0:
                    states_res[group_id] = []
                else:
                    states_res[group_id] = self._assign_states(
                        data_container=data_container,
                        molecules=molecules,
                        dihedrals=dihedrals_res,
                        peaks=peaks_res,
                        start=start,
                        end=end,
                        step=step,
                    )

    def _assign_states(
        self,
        data_container,
        molecules,
        dihedrals,
        peaks,
        start,
        end,
        step,
    ):
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
            start: Unused in legacy sampling.
            end: Unused in legacy sampling.
            step: Unused in legacy sampling.

        Returns:
            List of state labels (strings).
        """
        states = None

        for molecule in molecules:
            conformations = []
            mol = self._universe_operations.extract_fragment(data_container, molecule)
            number_frames = len(mol.trajectory)

            dihedral_results = Dihedral(dihedrals).run()

            for dihedral_index in range(len(dihedrals)):
                conformation = []
                for timestep in range(number_frames):
                    value = dihedral_results.results.angles[timestep][dihedral_index]
                    if value < 0:
                        value += 360

                    distances = [abs(value - peak) for peak in peaks[dihedral_index]]
                    conformation.append(np.argmin(distances))

                conformations.append(conformation)

            mol_states = [
                state
                for state in (
                    "".join(
                        str(int(conformations[d][f])) for d in range(len(dihedrals))
                    )
                    for f in range(number_frames)
                )
                if state
            ]

            if states is None:
                states = mol_states
            else:
                states.extend(mol_states)

        logger.debug("States: %s", states)
        return states

    @staticmethod
    def _count_total_items(levels, groups) -> int:
        """Count total progress items."""
        return sum(len(levels[mol_id]) for mols in groups.values() for mol_id in mols)

    @staticmethod
    def _progress_bar(total_items: int) -> Progress:
        """Create a Rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[title]}", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
        )
