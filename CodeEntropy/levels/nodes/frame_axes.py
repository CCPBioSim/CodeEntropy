"""Frame-local axes calculation.

This module defines a frame DAG node that computes per-molecule translational
axes for a single trajectory frame and exposes an AxesManager for downstream
nodes that need consistent axes and PBC-aware vectors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import numpy as np

from CodeEntropy.axes import AxesManager
from CodeEntropy.levels.mda_universe_operations import UniverseOperations

logger = logging.getLogger(__name__)

SharedData = MutableMapping[str, Any]
FrameContext = MutableMapping[str, Any]


class FrameAxesNode:
    """Compute per-frame translational axes for each molecule.

    This node operates in two modes:

    1) Frame-DAG mode:
       - Input is a frame context dict containing:
           ctx["shared"]      -> shared_data dict
           ctx["frame_index"] -> absolute trajectory frame index
       - Writes frame-local result to:
           ctx["frame_axes"]

    2) Static mode:
       - Input is the shared_data dict directly.
       - Uses shared_data["frame_index"] if present, otherwise defaults to 0.
       - Returns the computed axes payload (and also provides it in a synthetic
         frame context).

    Produces:
        - shared_data["axes_manager"]: AxesManager instance for downstream nodes.
        - frame_ctx["frame_axes"]: Dict containing:
            {
              "trans": {mol_id: np.ndarray shape (3, 3)},
              "custom": bool
            }
    """

    def __init__(self, universe_operations: UniverseOperations | None = None) -> None:
        """Initialize the node.

        Args:
            universe_operations: Helper for universe operations. If None, a default
                UniverseOperations instance is created.
        """
        self._universe_operations = universe_operations or UniverseOperations()
        self._axes_manager = AxesManager()

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Run the axes calculation for a single frame.

        Args:
            ctx: Either a frame context dict (Frame-DAG mode) or shared_data (static
                mode).

        Returns:
            A dict with translational axes and whether customized axes are enabled.

        Raises:
            KeyError: If required data is missing.
        """
        frame_ctx, shared, frame_index = self._resolve_context(ctx)

        universe = self._get_universe(shared)
        use_custom_axes = self._use_custom_axes(shared)

        # Expose the AxesManager for downstream nodes (e.g., torque with PBC vectors).
        shared["axes_manager"] = self._axes_manager

        # Ensure the universe is positioned at the correct frame before reading coords.
        universe.trajectory[frame_index]

        trans_axes = self._compute_trans_axes(universe)

        result = {"trans": trans_axes, "custom": use_custom_axes}
        frame_ctx["frame_axes"] = result
        return result

    def _resolve_context(
        self, ctx: Dict[str, Any]
    ) -> Tuple[FrameContext, SharedData, int]:
        """Resolve whether `ctx` is a frame context or shared_data.

        Args:
            ctx: Frame context dict or shared_data dict.

        Returns:
            Tuple of (frame_ctx, shared_data, frame_index).
        """
        if "shared" in ctx:
            shared = ctx["shared"]
            frame_index = int(ctx["frame_index"])
            return ctx, shared, frame_index

        shared = ctx
        frame_index = int(shared.get("frame_index", shared.get("time_index", 0)))
        frame_ctx: FrameContext = {"shared": shared, "frame_index": frame_index}
        return frame_ctx, shared, frame_index

    @staticmethod
    def _get_universe(shared: Mapping[str, Any]) -> Any:
        """Fetch the universe to operate on.

        Args:
            shared: Shared data mapping.

        Returns:
            MDAnalysis universe-like object.

        Raises:
            KeyError: If neither reduced_universe nor universe is present.
        """
        universe = shared.get("reduced_universe") or shared.get("universe")
        if universe is None:
            raise KeyError("shared_data must contain 'reduced_universe' or 'universe'")
        return universe

    @staticmethod
    def _use_custom_axes(shared: Mapping[str, Any]) -> bool:
        """Determine whether customized axes are enabled.

        Args:
            shared: Shared data mapping.

        Returns:
            True if customized axes are enabled.
        """
        args = shared["args"]
        return bool(getattr(args, "customised_axes", False))

    def _compute_trans_axes(self, universe: Any) -> Dict[int, np.ndarray]:
        """Compute translational axes for each molecule in the universe.

        Args:
            universe: MDAnalysis universe-like object.

        Returns:
            Mapping from molecule id to translational axes (3x3).
        """
        trans_axes: Dict[int, np.ndarray] = {}
        fragments = universe.atoms.fragments

        for mol_id, mol in enumerate(fragments):
            _, axes = self._axes_manager.get_vanilla_axes(mol)
            trans_axes[mol_id] = np.asarray(axes, dtype=float)

        return trans_axes
