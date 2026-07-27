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

from CodeEntropy.levels.axes import AxesCalculator

logger = logging.getLogger(__name__)

UAKey = tuple[int, int]


class DihedralDefinitions:
    """Build conformational state labels from dihedral angles."""

    def __init__(self) -> None:
        """Initializes the analysis helper."""
        self._axes = AxesCalculator()

    def method_res_bonds(self, data_container: Any, level: str) -> list[Any]:
        """Return dihedral AtomGroups for a container at a given level.

        Args:
            data_container: MDAnalysis container.
            level: Either ``"united_atom"`` or ``"residue"``.

        Returns:
            List of AtomGroups, each representing a dihedral definition.
        """
        atom_groups: list[Any] = []

        if level == "united_atom":
            selected_indices = {int(index) for index in data_container.indices}

            for dihedral in data_container.dihedrals:
                dihedral_atoms = dihedral.atoms
                dihedral_indices = {int(index) for index in dihedral_atoms.indices}

                if len(dihedral_atoms) == 4 and dihedral_indices.issubset(
                    selected_indices
                ):
                    atom_groups.append(dihedral_atoms)

        if level == "residue":
            num_residues = len(data_container.residues)
            if num_residues >= 4:
                for residue in range(4, num_residues + 1):
                    residue1 = data_container.residues[residue - 4]
                    residue2 = data_container.residues[residue - 3]
                    residue3 = data_container.residues[residue - 2]
                    residue4 = data_container.residues[residue - 1]

                    atom1 = self._atoms_in_source_bonded_to_target(
                        residue1,
                        residue2,
                    )
                    atom2 = self._atoms_in_source_bonded_to_target(
                        residue2,
                        residue1,
                    )
                    atom3 = self._atoms_in_source_bonded_to_target(
                        residue3,
                        residue4,
                    )
                    atom4 = self._atoms_in_source_bonded_to_target(
                        residue4,
                        residue3,
                    )

                    dihedral_atoms = atom1 + atom2 + atom3 + atom4

                    if len(dihedral_atoms) == 4:
                        atom_groups.append(dihedral_atoms)
                    else:
                        logger.debug(
                            "Skipping residue-level dihedral for local residues "
                            "%s-%s-%s-%s because it produced %d atoms.",
                            residue - 4,
                            residue - 3,
                            residue - 2,
                            residue - 1,
                            len(dihedral_atoms),
                        )

        logger.debug("Level: %s, Dihedrals: %s", level, atom_groups)
        return atom_groups

    def method_res_points(self, data_container: Any, level: str) -> list[Any]:
        """Return dihedral AtomGroups for a container at a given level.

        Args:
            data_container: MDAnalysis container (AtomGroup/Universe).
            level: Either "united_atom" or "residue".

        Returns:
            List of AtomGroups (each representing a dihedral definition).
        """
        atom_groups: list[Any] = []

        if level == "united_atom":
            dihedrals = data_container.dihedrals
            for d in dihedrals:
                atom_groups.append(d.atoms)

        if level == "residue":
            num_residues = len(data_container.residues)
            point: list[Any] = {}
            if num_residues >= 4:
                for residue in range(num_residues):
                    atom_search = []
                    index = data_container.residues.resindices[residue]
                    edges = data_container.select_atoms(
                        f"resindex {index} and bonded not resindex {index}"
                    )
                    if len(edges) > 1:
                        center = edges.center_of_geometry()
                        atom_search = data_container.select_atoms(
                            f"point {center[0]} {center[1]} {center[2]} 1.3"
                            f" and resindex {residue}"
                        )
                    if len(atom_search) == 0:
                        point[residue] = edges[0]
                    else:
                        point[residue] = atom_search[0]

                for index in range(3, num_residues):
                    atom1 = point[index - 3]
                    atom2 = point[index - 2]
                    atom3 = point[index - 1]
                    atom4 = point[index]
                    atom_groups.append(atom1 + atom2 + atom3 + atom4)

        return atom_groups

    def method_ua_backbone(self, data_container: Any, level: str) -> list[Any]:
        """Return dihedral AtomGroups for a container at a given level.

        Args:
            data_container: MDAnalysis container (AtomGroup/Universe).
            level: Either "united_atom" or "residue".

        Returns:
            List of AtomGroups (each representing a dihedral definition).
        """
        atom_groups: list[Any] = []

        if level == "united_atom":
            dihedrals = data_container.dihedrals
            for d in dihedrals:
                atom_groups.append(d.atoms)

        if level == "residue":
            num_residues = len(data_container.residues)
            backbone = data_container.select_atoms("name H and not name H")
            if num_residues >= 4:
                for index in range(num_residues):
                    residue = data_container.residues[index]
                    edges = data_container.select_atoms(
                        f"resindex {index} and bonded not resindex {index}"
                    )
                    if len(edges) == 1:
                        backbone += edges
                    elif len(edges) == 2:
                        chain = self._axes.get_chain(residue, edges[0], edges[1])
                        backbone += chain

                logger.debug(f"backbone = {backbone}")
                dihedrals = backbone.dihedrals
                for d in dihedrals:
                    atom_groups.append(d.atoms)

        return atom_groups

    def method_ua_whole(self, data_container: Any, level: str) -> list[Any]:
        """Return dihedral AtomGroups for a container at a given level.

        Args:
            data_container: MDAnalysis container (AtomGroup/Universe).
            level: Either "united_atom" or "residue".

        Returns:
            List of AtomGroups (each representing a dihedral definition).
        """
        atom_groups: list[Any] = []

        if level == "united_atom":
            dihedrals = data_container.dihedrals
            for d in dihedrals:
                atom_groups.append(d.atoms)

        return atom_groups

    @staticmethod
    def _atoms_in_source_bonded_to_target(
        source_residue: Any,
        target_residue: Any,
    ) -> Any:
        """Return source-residue atoms bonded to atoms in a target residue.

        This helper is used when constructing residue-level dihedral definitions
        from lightweight molecule AtomGroups. It selects atoms from the source
        residue that are bonded to any atom in the target residue without using
        global ``resindex`` selection strings.

        Args:
            source_residue: Residue whose atoms should be tested for bonds.
            target_residue: Adjacent residue providing the target bonded atoms.

        Returns:
            MDAnalysis AtomGroup containing atoms from ``source_residue`` that are
            bonded to at least one atom in ``target_residue``. If no matching
            atoms are found, an empty AtomGroup is returned.
        """
        source_atoms = source_residue.atoms
        target_indices = {int(index) for index in target_residue.atoms.indices}
        selected_indices: list[int] = []

        for atom in source_atoms:
            bonded_atoms = getattr(atom, "bonded_atoms", None)
            if bonded_atoms is None:
                continue

            bonded_indices = {int(index) for index in bonded_atoms.indices}
            if bonded_indices.intersection(target_indices):
                selected_indices.append(int(atom.index))

        return source_atoms.universe.atoms[selected_indices]
