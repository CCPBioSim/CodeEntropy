"""Frame-local neighbour observables for orientational entropy.

The frame execution layer calls :class:`Neighbors` once per selected trajectory
frame. Each call returns mergeable neighbour-count totals for every molecule
group. Static symmetry and linearity metadata is computed separately because it
does not vary by frame.
"""

from __future__ import annotations

from typing import Any

from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from rdkit import Chem

from CodeEntropy.levels.hbond_bias import HBondFactor
from CodeEntropy.levels.search import Search

NeighborCounts = dict[int, tuple[int, int]]


class Neighbors:
    """Compute neighbour-count and orientational metadata observables."""

    def __init__(self, search: Search | None = None) -> None:
        self._search = search or Search()
        self._bias = HBondFactor()

    def get_frame_neighbor_counts(
        self,
        *,
        universe: Any,
        levels: list[list[str]],
        groups: dict[int, list[int]],
        frame_source: Any,
        frame_index: int,
        search_type: str,
    ) -> NeighborCounts:
        """Return neighbour-count totals for one selected frame.

        The returned ``(total, sample_count)`` pairs are intentionally additive.
        Parent-side reducers combine them across frames and divide at finalisation.
        """
        frame_index = int(frame_index)
        frame_counts: NeighborCounts = {}

        self._universe = None
        self._groups = None
        self._levels = None
        self._search = Search()
        self._bias = HBondFactor()
        for group_id, molecule_ids in groups.items():
            if not molecule_ids:
                frame_counts[group_id] = (0, 0)
                continue

            highest_level = levels[molecule_ids[0]][-1]
            total_neighbors = 0
            sample_count = 0

            for molecule_id in molecule_ids:
                neighbors = self._get_neighbors_for_molecule(
                    universe=universe,
                    molecule_id=molecule_id,
                    highest_level=highest_level,
                    frame_source=frame_source,
                    frame_index=frame_index,
                    search_type=search_type,
                )
                total_neighbors += len(neighbors)
                sample_count += 1

            frame_counts[group_id] = (total_neighbors, sample_count)

        return frame_counts

    def get_bias(self, universe, groups):
        """
        Calculate orientational bias factors.

        Args:
            universe: MDAnalysis object
            groups: list of molecules sorted into groups

        Returns:
            hbond_factor: hydrogen bonding factor for each group
        """
        hbond_factor = {}

        # Get H-bonds from MDAnalysis for whole universe
        hbonds = HBA(universe=universe)
        hbonds.run()
        donors = hbonds.results.hbonds[:, 1].astype(int)
        acceptors = hbonds.results.hbonds[:, 3].astype(int)

        for group_id in groups.keys():
            hbond_factor[group_id] = self._bias.get_hbond_factor(
                universe, groups, group_id, donors, acceptors
            )

        return hbond_factor

    def get_symmetry(self, universe, groups):
        """
        Calculate symmetry number for the molecule.

        This function converts the MDAnalysis instance of a molecule into
        an RDKit object and then calculates the symmetry number and
        determines if the molecule is linear.

        Args:
            universe: MDAnalysis object
            groups: indices of the molecules for each group

        Returns:
            symmetry_number: dict, symmetry number of each group
            linear: dict, linear for each group
        """
        symmetry_number = {}
        linear = {}

        for group_id, molecule_ids in groups.items():
            if not molecule_ids:
                symmetry_number[group_id] = 0
                linear[group_id] = False
                continue

            rdkit_mol, number_heavy, number_hydrogen = self._get_rdkit_mol(
                universe,
                molecule_ids[0],
            )
            symmetry_number[group_id] = self._get_symmetry_number(
                rdkit_mol,
                number_heavy,
                number_hydrogen,
            )
            linear[group_id] = self._get_linear(rdkit_mol, number_heavy)

        return symmetry_number, linear

    def _get_neighbors_for_molecule(
        self,
        *,
        universe: Any,
        molecule_id: int,
        highest_level: str,
        frame_source: Any,
        frame_index: int,
        search_type: str,
    ) -> Any:
        """Run the configured neighbour search for one molecule and frame."""
        if search_type == "RAD":
            return self._search.get_RAD_neighbors(
                universe=universe,
                mol_id=molecule_id,
                frame_source=frame_source,
                frame_index=frame_index,
            )

        if search_type == "grid":
            return self._search.get_grid_neighbors(
                universe=universe,
                mol_id=molecule_id,
                highest_level=highest_level,
                frame_source=frame_source,
                frame_index=frame_index,
            )

        raise ValueError(f"unknown search_type {search_type}")

    @staticmethod
    def _get_rdkit_mol(universe: Any, molecule_id: int) -> tuple[Any, int, int]:
        """Convert one molecular fragment into an RDKit molecule."""
        if not hasattr(universe.atoms, "elements"):
            universe.guess_TopologyAttrs(to_guess=["elements"])

        molecule = universe.atoms.fragments[molecule_id]
        dummy_atoms = molecule.select_atoms("prop mass < 0.1")

        if len(dummy_atoms) > 0:
            fragment = molecule.select_atoms("prop mass > 0.1")
            rdkit_mol = fragment.convert_to("RDKIT", force=True, inferrer=None)
        else:
            try:
                rdkit_mol = molecule.convert_to("RDKIT", force=True)
            except Exception:
                rdkit_mol = molecule.convert_to("RDKIT", force=True, inferrer=None)

        number_heavy = rdkit_mol.GetNumHeavyAtoms()
        number_hydrogen = rdkit_mol.GetNumAtoms() - number_heavy
        return rdkit_mol, number_heavy, number_hydrogen

    @staticmethod
    def _get_symmetry_number(
        rdkit_mol: Any,
        number_heavy: int,
        number_hydrogen: int,
    ) -> int:
        """Calculate the molecular symmetry number used by orientational entropy."""
        if number_heavy > 1:
            heavy_atom_mol = Chem.RemoveHs(rdkit_mol)
            matches = rdkit_mol.GetSubstructMatches(
                heavy_atom_mol,
                uniquify=False,
                useChirality=True,
            )
            return len(matches)

        if number_hydrogen > 0:
            matches = rdkit_mol.GetSubstructMatches(
                rdkit_mol,
                uniquify=False,
                useChirality=True,
            )
            return len(matches)

        return 0

    @staticmethod
    def _get_linear(rdkit_mol: Any, number_heavy: int) -> bool:
        """Return whether a molecule is treated as linear."""
        if number_heavy == 1:
            return False

        if number_heavy == 2:
            return True

        heavy_atom_mol = Chem.RemoveHs(rdkit_mol)
        sp_count = sum(
            atom.GetHybridization() == Chem.HybridizationType.SP
            for atom in heavy_atom_mol.GetAtoms()
        )
        return sp_count >= number_heavy - 2
