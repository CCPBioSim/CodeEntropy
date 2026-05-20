"""Find neighbor and symmetry info for orientational entropy calculations.

This module defines a static DAG node that scans the trajectory and
finds neighbors and symmetry numbers. The resulting states are stored
in `shared_data` for later use by configurational entropy calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from CodeEntropy.levels.neighbors import Neighbors

SharedData = dict[str, Any]
ConformationalStates = dict[str, Any]


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
        """Compute neighbour and symmetry information.

        Args:
            shared_data: Shared data dictionary. Requires:
                - ``reduced_universe``
                - ``levels``
                - ``groups``
                - ``args.search_type``
                - ``frame_indices`` or ``n_frames``
            progress: Optional progress sink. Currently unused.

        Returns:
            The mutated shared data dictionary.
        """
        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]
        groups = shared_data["groups"]
        search_type = shared_data["args"].search_type

        frame_indices = list(
            shared_data.get(
                "frame_indices",
                range(int(shared_data["n_frames"])),
            )
        )

        number_neighbors = self._neighbor_analysis.get_neighbors(
            universe=u,
            levels=levels,
            groups=groups,
            frame_indices=frame_indices,
            search_type=search_type,
        )

        symmetry_number, linear = self._neighbor_analysis.get_symmetry(
            universe=u,
            groups=groups,
        )

        shared_data["neighbors"] = number_neighbors
        shared_data["symmetry_number"] = symmetry_number
        shared_data["linear"] = linear

        return shared_data
