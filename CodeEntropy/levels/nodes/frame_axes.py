import logging
from typing import Any, Dict

import numpy as np

from CodeEntropy.axes import AxesManager
from CodeEntropy.levels.mda_universe_operations import UniverseOperations

logger = logging.getLogger(__name__)


class FrameAxesNode:
    """
    Produces per-frame translational axes for each molecule.

    Input:
      ctx["shared"] (preferred) or ctx (fallback)
      ctx["frame_index"]

    Output:
      ctx["frame_axes"] = {"trans": {mol_id: 3x3 np.ndarray}, "custom": bool}
    """

    def __init__(self, universe_operations: UniverseOperations | None = None):
        self._universe_operations = universe_operations or UniverseOperations()
        self._axes_manager = AxesManager()

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        shared = ctx.get("shared", ctx)

        u = shared.get("reduced_universe", shared.get("universe"))
        if u is None:
            raise KeyError("shared_data must contain 'reduced_universe' or 'universe'")

        args = shared["args"]
        use_custom = bool(getattr(args, "customised_axes", False))

        frame_index = ctx.get("frame_index", shared.get("frame_index", 0))
        u.trajectory[frame_index]

        trans_axes: Dict[int, np.ndarray] = {}
        fragments = u.atoms.fragments

        for mol_id, mol in enumerate(fragments):
            if use_custom:
                if hasattr(self._axes_manager, "get_translation_axes"):
                    A = np.asarray(self._axes_manager.get_translation_axes(mol))
                elif hasattr(self._axes_manager, "translation_axes"):
                    A = np.asarray(self._axes_manager.translation_axes(mol))
                else:
                    A = np.asarray(mol.principal_axes())
            else:
                A = np.asarray(mol.principal_axes())

            trans_axes[mol_id] = A

        ctx["frame_axes"] = {"trans": trans_axes, "custom": use_custom}
        return ctx["frame_axes"]
