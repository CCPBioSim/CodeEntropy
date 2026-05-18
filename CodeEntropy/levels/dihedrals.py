"""Dihedral state assignment for conformational entropy.

This module converts dihedral angle time series into discrete conformational
state labels. The resulting state labels are used downstream to compute
conformational entropy.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

UAKey = tuple[int, int]


class DihedralDefinitions:
    """Build conformational state labels from dihedral angles."""

    def __init__(self) -> None:
        """Initializes the analysis helper."""

    def method_res_bonds(self, data_container: Any, level: str) -> list[Any]:
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

        logger.debug(f"Level: {level}, Dihedrals: {atom_groups}")
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
                    edges = data_container.select_atoms(
                        f"resindex {residue} and bonded not resindex {residue}"
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

        logger.debug(f"Level: {level}, Dihedrals: {atom_groups}")
        return atom_groups
