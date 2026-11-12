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
            mol = self._extract_topology_fragment(data_container, molecule_id)
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

    def _extract_topology_fragment(self, data_container: Any, molecule_id: Any) -> Any:
        """Return a molecule fragment for topology discovery.

        This uses the lightweight AtomGroup extraction helper when available so
        static conformational topology discovery does not create a standalone
        in-memory universe or copy trajectory frames. The fallback preserves
        compatibility with older ``UniverseOperations`` implementations.

        Args:
            data_container: Source MDAnalysis universe or universe-like container.
            molecule_id: Fragment index identifying the molecule to extract.

        Returns:
            MDAnalysis AtomGroup for the selected molecule
        """
        return self._universe_operations.extract_fragment_atomgroup(
            data_container,
            int(molecule_id),
        )

    def _select_heavy_residue(self, mol: Any, res_id: int) -> Any:
        """Select heavy atoms in a residue by residue index.

        Args:
            mol: Representative molecule AtomGroup.
            res_id: Local residue index.

        Returns:
            AtomGroup containing heavy atoms in the residue selection.
        """
        residue_atoms = mol.residues[int(res_id)].atoms
        selection1 = residue_atoms.indices[0]
        selection2 = residue_atoms.indices[-1]

        res_container = mol.select_atoms(
            f"index {selection1}:{selection2}",
            updating=False,
        )
        return res_container.select_atoms("prop mass > 1.1", updating=False)

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
