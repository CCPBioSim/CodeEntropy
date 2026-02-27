"""Find neighbor and symmetry info for orientational entropy calculations.

This module defines a static DAG node that scans the trajectory and
finds neighbors and symmetry numbers. The resulting states are stored
in `shared_data` for later use by configurational entropy calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from CodeEntropy.levels.neighbors import Neighbors

SharedData = Dict[str, Any]
ConformationalStates = Dict[str, Any]


@dataclass(frozen=True)
class NeighborConfig:
    """Configuration for neighbor finding.

    Attributes:
        start: Start frame index (inclusive).
        end: End frame index (exclusive).
        step: Frame stride.
    """

    start: int
    end: int
    step: int


class ComputeNeighborsNode:
    """Static node that finds neighbors from trajectory.

    Produces:
        shared_data["neighbors"] = {}
        shared_data["symmetry_number"] = {}
        shared_data["linear"] = {}

    Where:
        - neighbors is the average number of neighbors
        - symmetry_number is the symmetry number of the molecule, int
        - linear is a boolean; True for linear, False for non-linear
    """

    def __init__(self) -> None:
        """Initialize the node."""
        self._neighbor_analysis = Neighbors()

    def run(
        self, shared_data: SharedData, *, progress: object | None = None
    ) -> SharedData:
        """Compute conformational states and store them in shared_data.

        Args:
            shared_data: Shared data dictionary. Requires:
                - "reduced_universe"
                - "levels"
                - "groups"
                - "start", "end", "step"
                - "args" with attribute "bin_width"
            progress: Optional progress sink provided by ResultsReporter.progress().

        Returns:
            shared_data: SharedData
        """
        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]
        groups = shared_data["groups"]
        search_type = shared_data["args"].search_type

        # Get average number of neighbors
        number_neighbors = self._neighbor_analysis.get_neighbors(
            universe=u,
            levels=levels,
            groups=groups,
            search_type=search_type,
        )

        # Get symmetry numbers and linearity
        symmetry_number, linear = self._neighbor_analysis.get_symmetry(
            universe=u,
            groups=groups,
        )

        # Add information to shared_data
        shared_data["neighbors"] = number_neighbors
        shared_data["symmetry_number"] = symmetry_number
        shared_data["linear"] = linear

        return shared_data
