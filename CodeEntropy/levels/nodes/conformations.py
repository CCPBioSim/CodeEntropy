"""Compute conformational states for configurational entropy calculations."""

from __future__ import annotations

from typing import Any

from CodeEntropy.levels.dihedrals import ConformationStateBuilder
from CodeEntropy.trajectory.frames import FrameSelection

SharedData = dict[str, Any]
ConformationalStates = dict[str, Any]
FlexibleStates = dict[str, Any]


class ComputeConformationalStatesNode:
    """Static node that computes conformational states from selected frames.

    Produces:
        shared_data["conformational_states"] = {"ua": states_ua, "res": states_res}
        shared_data["flexible_dihedrals"] = {"ua": flexible_ua, "res": flexible_res}
    """

    def __init__(self, universe_operations: Any | None = None) -> None:
        """Initialise the conformational-state node.

        Args:
            universe_operations: Optional universe-operation adapter passed to the
                underlying conformation-state builder.
        """
        self._builder = ConformationStateBuilder(
            universe_operations=universe_operations
        )

    def run(
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

        Raises:
            KeyError: If required entries are missing from ``shared_data``.
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
