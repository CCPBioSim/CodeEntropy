"""Detect molecules and build grouping definitions for the reduced universe.

This module defines a static DAG node responsible for ensuring a reduced
universe is available and generating molecule groupings using the configured
grouping strategy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from CodeEntropy.molecules.grouping import MoleculeGrouper

logger = logging.getLogger(__name__)

SharedData = Dict[str, Any]


class DetectMoleculesNode:
    """Static node that establishes molecule groups.

    Produces:
        shared_data["reduced_universe"]
        shared_data["groups"]
        shared_data["number_molecules"]
    """

    def __init__(self) -> None:
        """Initialize the node with a molecule grouping helper."""
        self._grouping = MoleculeGrouper()

    def run(self, shared_data: SharedData) -> Dict[str, Any]:
        """Detect molecules and create grouping definitions.

        Args:
            shared_data: Shared data dictionary. Requires:
                - "universe"
                - "args"

        Returns:
            Dict containing:
                - "groups": Molecule grouping dictionary.
                - "number_molecules": Total molecule count.

        Raises:
            KeyError: If required keys are missing.
        """
        universe = self._ensure_reduced_universe(shared_data)

        grouping_strategy = self._get_grouping_strategy(shared_data)

        groups = self._grouping.grouping_molecules(universe, grouping_strategy)
        number_molecules = self._count_molecules(universe)

        shared_data["groups"] = groups
        shared_data["number_molecules"] = number_molecules

        logger.info(
            "[DetectMoleculesNode] %s molecules detected (reduced_universe)",
            number_molecules,
        )

        return {
            "groups": groups,
            "number_molecules": number_molecules,
        }

    def _ensure_reduced_universe(self, shared_data: SharedData) -> Any:
        """Ensure reduced_universe exists in shared_data.

        Args:
            shared_data: Shared data dictionary.

        Returns:
            Reduced universe object.

        Raises:
            KeyError: If no universe is available.
        """
        universe = shared_data.get("reduced_universe")

        if universe is None:
            universe = shared_data.get("universe")
            if universe is None:
                raise KeyError("shared_data must contain 'universe'")
            shared_data["reduced_universe"] = universe

        return universe

    def _get_grouping_strategy(self, shared_data: SharedData) -> str:
        """Extract grouping strategy from args.

        Args:
            shared_data: Shared data dictionary.

        Returns:
            Grouping strategy string.
        """
        args = shared_data["args"]
        return getattr(args, "grouping", "each")

    @staticmethod
    def _count_molecules(universe: Any) -> int:
        """Count molecules in the universe.

        Args:
            universe: MDAnalysis universe.

        Returns:
            Number of molecular fragments.
        """
        return len(universe.atoms.fragments)
