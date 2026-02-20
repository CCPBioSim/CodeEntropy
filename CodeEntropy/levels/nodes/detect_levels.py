"""Detect hierarchy levels present for each molecule in the reduced universe.

This module defines a static DAG node responsible for determining which
hierarchical levels (united_atom, residue, polymer) apply to each molecule.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from CodeEntropy.levels.hierarchy import LevelHierarchy

SharedData = Dict[str, Any]
Levels = List[List[str]]


class DetectLevelsNode:
    """Static node that determines hierarchy levels per molecule.

    Produces:
        shared_data["levels"]
        shared_data["number_molecules"]
    """

    def __init__(self) -> None:
        """Initialize the node with a LevelHierarchy helper."""
        self._hierarchy = LevelHierarchy()

    def run(self, shared_data: SharedData) -> Dict[str, Any]:
        """Detect levels and store results in shared_data.

        Args:
            shared_data: Shared data dictionary. Requires:
                - "reduced_universe"

        Returns:
            Dict containing:
                - "levels": List of levels per molecule.
                - "number_molecules": Total molecule count.

        Raises:
            KeyError: If required keys are missing.
        """
        universe = shared_data["reduced_universe"]

        number_molecules, levels = self._detect_levels(universe)

        shared_data["levels"] = levels
        shared_data["number_molecules"] = number_molecules

        return {
            "levels": levels,
            "number_molecules": number_molecules,
        }

    def _detect_levels(self, universe: Any) -> Tuple[int, Levels]:
        """Delegate level detection to LevelHierarchy.

        Args:
            universe: Reduced MDAnalysis universe.

        Returns:
            Tuple of molecule count and levels list.
        """
        return self._hierarchy.select_levels(universe)
