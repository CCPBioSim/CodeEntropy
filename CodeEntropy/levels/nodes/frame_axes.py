from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from CodeEntropy.axes import AxesManager
from CodeEntropy.levels.mda_universe_operations import UniverseOperations

logger = logging.getLogger(__name__)


class FrameAxesNode:
    """
    Produces per-frame translational axes for each molecule.
    Also exports the AxesManager into shared_data so torque can use PBC vectors.
    """

    def __init__(self, universe_operations: UniverseOperations | None = None):
        self._universe_operations = universe_operations or UniverseOperations()
        self._axes_manager = AxesManager()

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        if "shared" in ctx:
            frame_ctx = ctx
            shared = frame_ctx["shared"]
            frame_index = frame_ctx["frame_index"]
        else:
            shared = ctx
            frame_index = shared.get("frame_index", shared.get("time_index", 0))
            frame_ctx = {"shared": shared, "frame_index": frame_index}

        u = shared.get("reduced_universe", shared.get("universe"))
        if u is None:
            raise KeyError("shared_data must contain 'reduced_universe' or 'universe'")

        args = shared["args"]
        use_custom = bool(getattr(args, "customised_axes", False))

        shared["axes_manager"] = self._axes_manager

        u.trajectory[frame_index]

        trans_axes: Dict[int, np.ndarray] = {}
        fragments = u.atoms.fragments

        for mol_id, mol in enumerate(fragments):
            _, trans_axes[mol_id] = self._axes_manager.get_vanilla_axes(mol)

        frame_ctx["frame_axes"] = {"trans": trans_axes, "custom": use_custom}
        return frame_ctx["frame_axes"]
