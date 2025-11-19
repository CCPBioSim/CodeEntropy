import logging
from typing import Any, Dict

from CodeEntropy.levels.level_hierarchy import LevelHierarchy

logger = logging.getLogger(__name__)


class DetectLevelsNode:
    """
    Node to detect molecule count and assign levels.
    """

    def __init__(self):
        self._hier = LevelHierarchy()

    def run(self, shared_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        universe = shared_data["universe"]
        number_molecules, levels = self._hier.select_levels(universe)

        logger.debug(f"[DetectLevelsNode] number_molecules={number_molecules}")
        logger.debug(f"[DetectLevelsNode] levels={levels}")

        return {
            "number_molecules": number_molecules,
            "levels": levels,
        }
