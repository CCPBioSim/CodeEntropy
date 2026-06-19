"""Dihedral topology discovery for conformational state analysis.

This module contains the static molecule/residue dihedral discovery logic used
by conformational entropy calculations. The methods here identify which
dihedrals should be analysed; they do not inspect trajectory frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MoleculeDihedralTopology:
    """Static conformational dihedral topology for one molecule.

    Attributes:
        group_id: Molecule group id.
        molecule_id: Molecule id.
        molecule_order: Position of the molecule within its group.
        num_residues: Number of residues in the molecule.
        ua_dihedrals_by_residue: United-atom dihedrals by residue index.
        residue_dihedrals: Residue-level dihedrals for the molecule.
    """

    group_id: int
    molecule_id: Any
    molecule_order: int
    num_residues: int
    ua_dihedrals_by_residue: dict[int, list[Any]]
    residue_dihedrals: list[Any]


class DihedralTopologyDiscovery:
    """Discover molecule-level dihedral definitions for conformational analysis."""

    def _discover_group_dihedral_topology(
        self,
        data_container: Any,
        group_id: int,
        molecules: list[Any],
        level_list: list[Any],
    ) -> list[MoleculeDihedralTopology]:
        """Discover static conformational topology for a molecule group.

        Args:
            data_container: MDAnalysis universe.
            group_id: Molecule group id.
            molecules: Molecule ids in the group.
            level_list: Enabled hierarchy levels.

        Returns:
            Static per-molecule dihedral topology used by both chunked passes.
        """
        topologies: list[MoleculeDihedralTopology] = []

        for molecule_order, molecule_id in enumerate(molecules):
            mol = self._universe_operations.extract_fragment(
                data_container, molecule_id
            )
            num_residues = len(mol.residues)
            ua_dihedrals_by_residue: dict[int, list[Any]] = {}
            residue_dihedrals: list[Any] = []

            if "united_atom" in level_list:
                for res_id in range(num_residues):
                    heavy_res = self._select_heavy_residue(mol, res_id)
                    ua_dihedrals_by_residue[res_id] = self._get_dihedrals(
                        heavy_res,
                        "united_atom",
                    )

            if "residue" in level_list:
                residue_dihedrals = self._get_dihedrals(mol, "residue")

            topologies.append(
                MoleculeDihedralTopology(
                    group_id=group_id,
                    molecule_id=molecule_id,
                    molecule_order=molecule_order,
                    num_residues=num_residues,
                    ua_dihedrals_by_residue=ua_dihedrals_by_residue,
                    residue_dihedrals=residue_dihedrals,
                )
            )

        return topologies

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
