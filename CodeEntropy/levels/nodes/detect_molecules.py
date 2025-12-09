import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DetectMoleculesNode:
    def run(self, shared_data: Dict[str, Any], **_):
        u = shared_data["universe"]
        fragments = u.atoms.fragments
        num_mol = len(fragments)

        logger.info(f"[DetectMoleculesNode] {num_mol} molecules detected")

        return {"number_molecules": num_mol, "fragments": fragments}
