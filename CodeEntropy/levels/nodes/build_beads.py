import logging
from typing import Any, Dict, List, Tuple

from CodeEntropy.config.run import RunManager
from CodeEntropy.levels.level_hierarchy import LevelHierarchy

logger = logging.getLogger(__name__)


class BuildBeadsNode:
    """
    Node to build collections of beads for each molecule and level.
    """

    def __init__(self, run_manager: RunManager):
        self._hier = LevelHierarchy()
        self._run_manager = run_manager

    def run(
        self, shared_data: Dict[str, Any], detect_levels: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        universe = shared_data["universe"]
        levels = detect_levels["levels"]
        beads_by_mol_level: Dict[Tuple[int, str], List[Any]] = {}

        for mol_id, level_list in enumerate(levels):
            # Create universal molecule container if needed
            mol_container = self._run_manager.new_U_select_atom(
                universe,
                f"index {universe.atoms.fragments[mol_id].indices[0]}:"
                f"{universe.atoms.fragments[mol_id].indices[-1]}",
            )
            for level in level_list:
                beads = self._hier.get_beads(mol_container, level)
                beads_by_mol_level[(mol_id, level)] = beads

        logger.debug(
            f"[BuildBeadsNode] built beads for {len(beads_by_mol_level)} combinations"
        )

        return {"beads_by_mol_level": beads_by_mol_level}
