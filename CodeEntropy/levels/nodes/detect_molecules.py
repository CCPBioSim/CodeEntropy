# CodeEntropy/levels/nodes/detect_molecules.py

import logging

logger = logging.getLogger(__name__)


class DetectMoleculesNode:
    def run(self, shared_data):
        u = shared_data["reduced_universe"]
        fragments = u.atoms.fragments
        num_mol = len(fragments)

        logger.info(f"[DetectMoleculesNode] {num_mol} molecules detected")

        shared_data["fragments"] = fragments
        shared_data["number_molecules"] = num_mol
