"""Compute conformational states for configurational entropy calculations.

This module defines a static DAG node that scans the trajectory and builds
conformational state descriptors (united-atom and residue level). The resulting
states are stored in `shared_data` for later use by configurational entropy
calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from CodeEntropy.levels.dihedrals import DihedralAnalysis

SharedData = Dict[str, Any]
ConformationalStates = Dict[str, Any]


@dataclass(frozen=True)
class ConformationalStateConfig:
    """Configuration for conformational state construction.

    Attributes:
        start: Start frame index (inclusive).
        end: End frame index (exclusive).
        step: Frame stride.
        bin_width: Histogram bin width in degrees.
    """

    start: int
    end: int
    step: int
    bin_width: int


class ComputeConformationalStatesNode:
    """Static node that computes conformational states from trajectory dihedrals.

    Produces:
        shared_data["conformational_states"] = {"ua": states_ua, "res": states_res}

    Where:
        - states_ua is a dict keyed by (group_id, local_residue_id)
        - states_res is a list-like structure indexed by group_id (or equivalent)
    """

    def __init__(self, universe_operations: Any) -> None:
        """Initialize the node.

        Args:
            universe_operations: Object providing universe selection utilities used
                by `DihedralAnalysis`.
        """
        self._dihedral_analysis = DihedralAnalysis(
            universe_operations=universe_operations
        )

    def run(self, shared_data: SharedData) -> Dict[str, ConformationalStates]:
        """Compute conformational states and store them in shared_data.

        Args:
            shared_data: Shared data dictionary. Requires:
                - "reduced_universe"
                - "levels"
                - "groups"
                - "start", "end", "step"
                - "args" with attribute "bin_width"

        Returns:
            Dict containing "conformational_states" (also written into shared_data).

        Raises:
            KeyError: If required keys are missing.
            AttributeError: If `shared_data["args"]` lacks `bin_width`.
        """
        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]
        groups = shared_data["groups"]

        cfg = self._build_config(shared_data)

        states_ua, states_res = self._dihedral_analysis.build_conformational_states(
            data_container=u,
            levels=levels,
            groups=groups,
            start=cfg.start,
            end=cfg.end,
            step=cfg.step,
            bin_width=cfg.bin_width,
        )

        conformational_states: ConformationalStates = {
            "ua": states_ua,
            "res": states_res,
        }
        shared_data["conformational_states"] = conformational_states
        return {"conformational_states": conformational_states}

    @staticmethod
    def _build_config(shared_data: SharedData) -> ConformationalStateConfig:
        """Extract and validate configuration from shared_data.

        Args:
            shared_data: Shared data dictionary.

        Returns:
            ConformationalStateConfig with normalized integer fields.
        """
        start = int(shared_data["start"])
        end = int(shared_data["end"])
        step = int(shared_data["step"])
        bin_width = int(shared_data["args"].bin_width)
        return ConformationalStateConfig(
            start=start, end=end, step=step, bin_width=bin_width
        )
