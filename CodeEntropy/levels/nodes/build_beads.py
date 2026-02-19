"""Build bead (AtomGroup index) definitions for each molecule and hierarchy level.

This module defines the `BuildBeadsNode`, a static DAG node that constructs bead
definitions once, in reduced-universe index space. These bead definitions are
used by later frame-level nodes (e.g., covariance construction) without needing
to re-run selection logic every frame.

Beads are stored as arrays of atom indices (in the reduced universe).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, MutableMapping, Tuple

import numpy as np

from CodeEntropy.levels.level_hierarchy import LevelHierarchy

logger = logging.getLogger(__name__)

BeadKey = Tuple[int, str] | Tuple[int, str, int]
BeadsMap = Dict[BeadKey, List[np.ndarray]]


@dataclass(frozen=True)
class UnitedAtomBead:
    """A united-atom bead associated with a residue bucket.

    Attributes:
        residue_id: Local residue index within the molecule (0..n_residues-1).
        atom_indices: Atom indices (reduced-universe index space) belonging to the bead.
    """

    residue_id: int
    atom_indices: np.ndarray


class BuildBeadsNode:
    """Build bead definitions once, in reduced-universe index space.

    Output contract:
        Writes `shared_data["beads"]` with keys:
          - (mol_id, "united_atom", res_id) -> list[np.ndarray]
          - (mol_id, "residue")             -> list[np.ndarray]
          - (mol_id, "polymer")             -> list[np.ndarray]

    Notes:
        United-atom beads are generated at the molecule level (preserving the
        underlying ordering provided by `LevelHierarchy.get_beads`) and then
        grouped into residue buckets based on the heavy atom that defines the bead.
    """

    def __init__(self, hierarchy: LevelHierarchy | None = None) -> None:
        """Initialize the node.

        Args:
            hierarchy: Optional `LevelHierarchy` dependency. If not provided,
                a default instance is created.
        """
        self._hier = hierarchy or LevelHierarchy()

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build bead definitions for all molecules and levels.

        Args:
            shared_data: Shared data dictionary. Requires:
                - "reduced_universe": MDAnalysis.Universe
                - "levels": list[list[str]]

        Returns:
            Dict containing the "beads" mapping (also written into shared_data).

        Raises:
            KeyError: If required keys are missing from `shared_data`.
        """
        u = shared_data["reduced_universe"]
        levels: List[List[str]] = shared_data["levels"]

        beads: BeadsMap = {}
        fragments = u.atoms.fragments

        for mol_id, level_list in enumerate(levels):
            mol = fragments[mol_id]

            if "united_atom" in level_list:
                self._add_united_atom_beads(beads=beads, mol_id=mol_id, mol=mol)

            if "residue" in level_list:
                self._add_residue_beads(beads=beads, mol_id=mol_id, mol=mol)

            if "polymer" in level_list:
                self._add_polymer_beads(beads=beads, mol_id=mol_id, mol=mol)

        shared_data["beads"] = beads
        return {"beads": beads}

    def _add_united_atom_beads(
        self, beads: MutableMapping[BeadKey, List[np.ndarray]], mol_id: int, mol
    ) -> None:
        """Compute and store united-atom beads grouped into residue buckets.

        Args:
            beads: Output bead mapping mutated in-place.
            mol_id: Molecule (fragment) index.
            mol: MDAnalysis AtomGroup representing the molecule.
        """
        ua_beads = self._hier.get_beads(mol, "united_atom")

        buckets: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
        for bead_i, bead in enumerate(ua_beads):
            atom_indices = self._validate_bead_indices(
                bead, mol_id=mol_id, level="united_atom", bead_i=bead_i
            )
            if atom_indices is None:
                continue

            residue_id = self._infer_local_residue_id(mol=mol, bead=bead)
            buckets[residue_id].append(atom_indices)

        for local_res_id in range(len(mol.residues)):
            beads[(mol_id, "united_atom", local_res_id)] = buckets.get(local_res_id, [])

    def _add_residue_beads(
        self, beads: MutableMapping[BeadKey, List[np.ndarray]], mol_id: int, mol
    ) -> None:
        """Compute and store residue beads.

        Args:
            beads: Output bead mapping mutated in-place.
            mol_id: Molecule (fragment) index.
            mol: MDAnalysis AtomGroup representing the molecule.
        """
        res_beads = self._hier.get_beads(mol, "residue")
        kept: List[np.ndarray] = []

        for bead_i, bead in enumerate(res_beads):
            atom_indices = self._validate_bead_indices(
                bead, mol_id=mol_id, level="residue", bead_i=bead_i
            )
            if atom_indices is None:
                continue
            kept.append(atom_indices)

        beads[(mol_id, "residue")] = kept

        if len(kept) == 0:
            logger.error(
                "[BuildBeadsNode] No residue beads kept for mol=%s. Residue-level "
                "entropy may be 0.0.",
                mol_id,
            )

    def _add_polymer_beads(
        self, beads: MutableMapping[BeadKey, List[np.ndarray]], mol_id: int, mol
    ) -> None:
        """Compute and store polymer beads.

        Args:
            beads: Output bead mapping mutated in-place.
            mol_id: Molecule (fragment) index.
            mol: MDAnalysis AtomGroup representing the molecule.
        """
        poly_beads = self._hier.get_beads(mol, "polymer")
        kept: List[np.ndarray] = []

        for bead_i, bead in enumerate(poly_beads):
            atom_indices = self._validate_bead_indices(
                bead, mol_id=mol_id, level="polymer", bead_i=bead_i
            )
            if atom_indices is None:
                continue
            kept.append(atom_indices)

        beads[(mol_id, "polymer")] = kept

    @staticmethod
    def _validate_bead_indices(
        bead, mol_id: int, level: str, bead_i: int
    ) -> np.ndarray | None:
        """Return a bead's atom indices, or None if the bead is empty.

        Args:
            bead: MDAnalysis AtomGroup representing the bead.
            mol_id: Molecule id used only for logging context.
            level: Level name used only for logging context.
            bead_i: Bead index used only for logging context.

        Returns:
            A copy of the bead indices as a NumPy array, or None if the bead is empty.
        """
        if len(bead) == 0:
            logger.warning(
                "[BuildBeadsNode] Empty bead skipped: mol=%s level=%s bead_i=%s",
                mol_id,
                level,
                bead_i,
            )
            return None
        return bead.indices.copy()

    @staticmethod
    def _infer_local_residue_id(mol, bead) -> int:
        """Infer the local residue bucket for a united-atom bead.

        Strategy:
            - Select heavy atoms in the bead (mass > 1.1).
            - Use the first heavy atom's `resindex` (universe-level).
            - Map that universe-level `resindex` back to the molecule's local residue
              index by scanning `mol.residues`.

        Args:
            mol: Molecule AtomGroup.
            bead: United-atom bead AtomGroup.

        Returns:
            Local residue index in [0, len(mol.residues) - 1]. Falls back to 0 if
            mapping cannot be determined.
        """
        heavy = bead.select_atoms("prop mass > 1.1")
        if len(heavy) == 0:
            return 0

        target_resindex = int(heavy[0].resindex)
        for local_i, res in enumerate(mol.residues):
            if int(res.resindex) == target_resindex:
                return local_i

        # Conservative fallback: bucket into residue 0 rather than dropping.
        return 0
