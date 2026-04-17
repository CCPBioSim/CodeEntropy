"""Compute conformational states for configurational entropy calculations.

This module defines a static DAG node that scans the trajectory and builds
conformational state descriptors (united-atom and residue level). The resulting
states are stored in `shared_data` for later use by configurational entropy
calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from CodeEntropy.levels.dihedrals import ConformationStateBuilder

SharedData = dict[str, Any]
ConformationalStates = dict[str, Any]
FlexibleStates = dict[str, Any]


@dataclass(frozen=True)
class ConformationalStateConfig:
    """Configuration for conformational state construction.

    Attributes:
        n_frames: Number of frames to be analyised.
        bin_width: Histogram bin width in degrees.
    """

    n_frames: int
    bin_width: int


class ComputeConformationalStatesNode:
    """Static node that computes conformational states from trajectory dihedrals.

    Produces:
        shared_data["conformational_states"] = {"ua": states_ua, "res": states_res}
        shared_data["flexible_dihedrals"] = {"ua: flexible_ua, "res": flexible_res}

    Where:
        - states_ua is a dict keyed by (group_id, local_residue_id)
        - states_res is a list-like structure indexed by group_id (or equivalent)
        - flexible_ua is a dict keyed by (group_id, local_residue_id)
        - flexible_res is a list-like structure indexed by group_id (or equivalent)
    """

    def __init__(self, universe_operations: Any) -> None:
        """Initialize the node.

        Args:
            universe_operations: Object providing universe selection utilities used
                by `ConformationStateBuilder`.
        """
        self._dihedral_analysis = ConformationStateBuilder(
            universe_operations=universe_operations
        )

    def run(
        self, shared_data: SharedData, *, progress: object | None = None
    ) -> dict[str, ConformationalStates]:
        """Compute conformational states and store them in shared_data.

        Args:
            shared_data: Shared data dictionary. Requires:
                - "reduced_universe"
                - "levels"
                - "groups"
                - "n_frames"
                - "args" with attribute "bin_width"
            progress: Optional progress sink provided by ResultsReporter.progress().

        Returns:
            Dict containing "conformational_states" (also written into shared_data).
        """
        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]
        groups = shared_data["groups"]
        bin_width = int(shared_data["args"].bin_width)

        states_ua, states_res, flexible_ua, flexible_res = (
            self._dihedral_analysis.build_conformational_states(
                data_container=u,
                levels=levels,
                groups=groups,
                bin_width=bin_width,
                progress=progress,
            )
        )

        # Get state information into shared_data
        conformational_states: ConformationalStates = {
            "ua": states_ua,
            "res": states_res,
        }
        shared_data["conformational_states"] = conformational_states

        # Get flexible_dihedral data into shared_data
        flexible_states: FlexibleStates = {
            "ua": flexible_ua,
            "res": flexible_res,
        }
        shared_data["flexible_dihedrals"] = flexible_states

        return {"conformational_states": conformational_states}
