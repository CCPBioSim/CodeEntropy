"""Conformational-state DAG orchestration.

This module owns the conformational stage between static structural setup and
frame-local covariance/neighbour execution.
"""

from __future__ import annotations

from typing import Any

from CodeEntropy.levels.dihedrals import ConformationStateBuilder
from CodeEntropy.trajectory.frames import FrameSelection

SharedData = dict[str, Any]
ConformationalStates = dict[str, Any]
FlexibleStates = dict[str, Any]


class ConformationDAG:
    """Execute conformational-state construction for selected trajectory frames."""

    def __init__(self, universe_operations: Any | None = None) -> None:
        self._builder = ConformationStateBuilder(
            universe_operations=universe_operations
        )

    def build(self) -> ConformationDAG:
        """Build the conformational DAG topology.

        Returns:
            Self, to allow fluent construction.
        """
        return self

    def execute(
        self,
        shared_data: SharedData,
        *,
        progress: object | None = None,
    ) -> dict[str, ConformationalStates]:
        """Compute conformational states and store them in shared workflow data.

        Args:
            shared_data: Shared workflow data containing ``reduced_universe``,
                ``levels``, ``groups``, ``frame_selection``, and ``args.bin_width``.
            progress: Optional progress sink forwarded to the conformation builder.
        Returns:
            A dictionary containing the computed ``conformational_states`` mapping.
        """
        universe = shared_data["reduced_universe"]
        levels = shared_data["levels"]
        groups = shared_data["groups"]
        frame_selection: FrameSelection = shared_data["frame_selection"]
        bin_width = int(shared_data["args"].bin_width)

        states_ua, states_res, flexible_ua, flexible_res = (
            self._builder.build_conformational_states(
                data_container=universe,
                levels=levels,
                groups=groups,
                bin_width=bin_width,
                frame_selection=frame_selection,
                progress=progress,
            )
        )

        conformational_states: ConformationalStates = {
            "ua": states_ua,
            "res": states_res,
        }
        flexible_dihedrals: FlexibleStates = {
            "ua": flexible_ua,
            "res": flexible_res,
        }

        shared_data["conformational_states"] = conformational_states
        shared_data["flexible_dihedrals"] = flexible_dihedrals

        return {"conformational_states": conformational_states}
