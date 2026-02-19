"""Hierarchy level selection and bead construction.

This module defines `LevelHierarchy`, which is responsible for:
  1) Determining which hierarchy levels apply to each molecule.
  2) Constructing "beads" (AtomGroups) for a given molecule at a given level.

Notes:
- The "residue" bead construction must use residues attached to the provided
  AtomGroup/container. Using `resindex` selection strings is unsafe because
  `resindex` is global to the Universe and can produce empty/incorrect beads
  when operating on per-molecule containers beyond the first molecule.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class LevelHierarchy:
    """Determine applicable hierarchy levels and build beads for each level.

    A "level" represents a resolution scale used throughout the entropy workflow:
      - united_atom: heavy-atom-centered beads (plus bonded hydrogens)
      - residue:     residue beads
      - polymer:     whole-molecule bead

    This class intentionally does not perform any entropy calculations. It only
    provides structural information (levels and beads).
    """

    def select_levels(self, data_container) -> Tuple[int, List[List[str]]]:
        """Select applicable hierarchy levels for each molecule in the container.

        A molecule is always assigned the `united_atom` level.

        Additional levels are included if:
          - `residue`: the heavy-atom subset has more than one atom.
          - `polymer`: the heavy-atom subset spans more than one residue.

        Args:
            data_container: An MDAnalysis Universe (or compatible object) with
                `atoms.fragments` available.

        Returns:
            A tuple of:
              - number_molecules: Number of molecular fragments.
              - levels: List where `levels[mol_id]` is a list of level names
                (strings) for that molecule in increasing coarseness.
        """
        number_molecules = len(data_container.atoms.fragments)
        logger.debug("The number of molecules is %d.", number_molecules)

        fragments = data_container.atoms.fragments
        levels: List[List[str]] = [[] for _ in range(number_molecules)]

        for mol_id in range(number_molecules):
            levels[mol_id].append("united_atom")

            heavy_atoms = fragments[mol_id].select_atoms("prop mass > 1.1")
            if len(heavy_atoms) > 1:
                levels[mol_id].append("residue")

                number_residues = len(heavy_atoms.residues)
                if number_residues > 1:
                    levels[mol_id].append("polymer")

        logger.debug("Selected levels: %s", levels)
        return number_molecules, levels

    def get_beads(self, data_container, level: str) -> List:
        """Build beads for a given container at a given hierarchy level.

        Args:
            data_container: An MDAnalysis AtomGroup representing a molecule or
                other container that has `.select_atoms(...)` and `.residues`.
            level: One of {"united_atom", "residue", "polymer"}.

        Returns:
            A list of MDAnalysis AtomGroups representing beads at that level.

        Raises:
            ValueError: If `level` is not a supported hierarchy level.
        """
        if level == "polymer":
            return [data_container.select_atoms("all")]

        if level == "residue":
            return self._build_residue_beads(data_container)

        if level == "united_atom":
            return self._build_united_atom_beads(data_container)

        raise ValueError(f"Unknown level: {level}")

    # ------------------------------------------------------------------
    # Bead builders (single responsibility, testable)
    # ------------------------------------------------------------------

    def _build_residue_beads(self, data_container) -> List:
        """Build one bead per residue using the container's residues.

        Args:
            data_container: MDAnalysis AtomGroup with `.residues`.

        Returns:
            List of residue AtomGroups.
        """
        beads = [res.atoms for res in data_container.residues]
        logger.debug("Residue beads sizes: %s", [len(b) for b in beads])
        return beads

    def _build_united_atom_beads(self, data_container) -> List:
        """Build united-atom beads from heavy atoms and their bonded hydrogens.

        For each heavy atom, a bead is defined as:
          - that heavy atom, plus
          - any bonded atoms with mass <= 1.1 (hydrogen-like).

        If no heavy atoms exist in the container, the entire container becomes
        a single bead.

        Args:
            data_container: MDAnalysis AtomGroup representing a molecule.

        Returns:
            List of bead AtomGroups.
        """
        heavy_atoms = data_container.select_atoms("prop mass > 1.1")
        if len(heavy_atoms) == 0:
            return [data_container.select_atoms("all")]

        beads = []
        for atom in heavy_atoms:
            selection = (
                f"index {atom.index} "
                f"or ((prop mass <= 1.1) and bonded index {atom.index})"
            )
            beads.append(data_container.select_atoms(selection))

        logger.debug("United-atom bead sizes: %s", [len(b) for b in beads])
        return beads
