"""Utilities for grouping molecules for entropy analysis.

This module provides strategies for grouping molecular fragments from an
MDAnalysis Universe into deterministic groups used for statistical averaging
during entropy calculations.

Grouping strategies are designed to be stable and reproducible so that group
IDs remain consistent across runs given the same input system.

Available strategies:
    - each: Every molecule is treated as its own group.
    - molecules: Molecules are grouped by chemical signature
      (atom count and atom names in order).
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

logger = logging.getLogger(__name__)

GroupId = int
MoleculeId = int
MoleculeGroups = Dict[GroupId, List[MoleculeId]]
Signature = Tuple[int, Tuple[str, ...]]


@dataclass(frozen=True)
class GroupingConfig:
    """Configuration for molecule grouping.

    Attributes:
        strategy: Grouping strategy name. Supported values are:
            - "each": each molecule gets its own group.
            - "molecules": group molecules by chemical signature
              (atom count + atom names in order).
    """

    strategy: str


class GroupMolecules:
    """Build groups of molecules for averaging.

    This class provides strategies for grouping molecule fragments from an
    MDAnalysis Universe. Groups are returned as a mapping:

        group_id -> [molecule_id, molecule_id, ...]

    Group IDs are deterministic and stable.

    Supported strategies:
        - "each": Every molecule is its own group.
        - "molecules": Molecules are grouped by chemical signature
          (atom count, atom names in order). The group ID is the first molecule
          index observed for that signature.
    """

    def grouping_molecules(self, universe, grouping: str) -> MoleculeGroups:
        """Group molecules according to a selected strategy.

        Args:
            universe: MDAnalysis Universe containing atoms and fragments.
            grouping: Strategy name ("each" or "molecules").

        Returns:
            A dict mapping group IDs to molecule indices.

        Raises:
            ValueError: If `grouping` is not a supported strategy.
        """
        config = GroupingConfig(strategy=grouping)
        grouper = self._get_strategy(config.strategy)
        groups = grouper(universe)

        self._log_summary(groups)
        return groups

    def _get_strategy(self, strategy: str) -> Callable[[object], MoleculeGroups]:
        """Resolve a strategy name to a grouping implementation.

        Args:
            strategy: Strategy name.

        Returns:
            Callable that accepts a Universe and returns molecule groups.

        Raises:
            ValueError: If the strategy is unknown.
        """
        strategies: Mapping[str, Callable[[object], MoleculeGroups]] = {
            "each": self._group_each,
            "molecules": self._group_by_signature,
        }

        try:
            return strategies[strategy]
        except KeyError as exc:
            raise ValueError(f"Unknown grouping strategy: {strategy!r}") from exc

    def _group_each(self, universe) -> MoleculeGroups:
        """Create one group per molecule.

        Args:
            universe: MDAnalysis Universe.

        Returns:
            Dict where each molecule id maps to a singleton list [molecule id].
        """
        n_molecules = self._num_molecules(universe)
        return {mol_id: [mol_id] for mol_id in range(n_molecules)}

    def _group_by_signature(self, universe) -> MoleculeGroups:
        """Group molecules by chemical signature with stable group IDs.

        Signature is defined as:
            (atom_count, atom_names_in_order)

        Group ID selection is stable and matches the previous behavior:
        the first molecule index encountered for a signature is the group ID.

        Args:
            universe: MDAnalysis Universe.

        Returns:
            Dict mapping representative molecule id -> list of all molecule ids
            sharing the same signature.
        """
        fragments = self._fragments(universe)

        signature_to_rep: Dict[Signature, MoleculeId] = {}
        groups: MoleculeGroups = {}

        for mol_id, fragment in enumerate(fragments):
            signature = self._signature(fragment)
            rep_id = self._representative_id(signature_to_rep, signature, mol_id)
            groups.setdefault(rep_id, []).append(mol_id)

        return groups

    def _num_molecules(self, universe) -> int:
        """Return number of molecule fragments.

        Args:
            universe: MDAnalysis Universe.

        Returns:
            Number of fragments (molecules).
        """
        return len(self._fragments(universe))

    def _fragments(self, universe) -> Sequence[object]:
        """Return universe fragments (molecules).

        Args:
            universe: MDAnalysis Universe.

        Returns:
            Sequence of fragments.
        """
        return universe.atoms.fragments

    def _signature(self, fragment) -> Signature:
        """Build a chemical signature for a fragment.

        Args:
            fragment: MDAnalysis AtomGroup representing a fragment.

        Returns:
            A tuple of (atom_count, atom_names_in_order).
        """
        names = tuple(fragment.names)
        return (len(names), names)

    def _representative_id(
        self,
        signature_to_rep: Dict[Signature, MoleculeId],
        signature: Signature,
        candidate_id: MoleculeId,
    ) -> MoleculeId:
        """Return stable representative id for a signature.

        Args:
            signature_to_rep: Cache mapping signature -> representative id.
            signature: Chemical signature of current molecule.
            candidate_id: Current molecule id.

        Returns:
            Representative id for this signature (first seen molecule id).
        """
        rep_id = signature_to_rep.get(signature)
        if rep_id is None:
            signature_to_rep[signature] = candidate_id
            return candidate_id
        return rep_id

    def _log_summary(self, groups: MoleculeGroups) -> None:
        """Log grouping summary.

        Args:
            groups: Group mapping to summarize.
        """
        logger.info("Number of molecule groups: %d", len(groups))
        logger.debug("Molecule groups: %s", groups)
