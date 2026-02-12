import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from CodeEntropy.levels.level_hierarchy import LevelHierarchy

logger = logging.getLogger(__name__)


class BuildBeadsNode:
    """
    Build bead definitions ONCE, in reduced_universe index space.

    shared_data["beads"] dict keys:
      (mol_id, "united_atom", res_id) -> list[np.ndarray] # UA beads grouped by residue
      (mol_id, "residue")             -> list[np.ndarray]
      (mol_id, "polymer")             -> list[np.ndarray]

    IMPORTANT:
      UA beads are generated at the MOLECULE level (mol) to preserve procedural ordering
      (molecule-heavy-atom ordinal), then assigned into residue buckets.
    """

    def __init__(self):
        self._hier = LevelHierarchy()

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]

        beads: Dict[Any, List[np.ndarray]] = {}
        fragments = u.atoms.fragments

        for mol_id, level_list in enumerate(levels):
            mol = fragments[mol_id]

            if "united_atom" in level_list:
                ua_beads_mol = self._hier.get_beads(mol, "united_atom")

                buckets: Dict[int, List[np.ndarray]] = defaultdict(list)

                for i, b in enumerate(ua_beads_mol):
                    if len(b) == 0:
                        logger.warning(
                            f"[BuildBeadsNode] EMPTY UA bead: mol={mol_id} bead_i={i}"
                        )
                        continue

                    heavy = b.select_atoms("prop mass > 1.1")
                    if len(heavy) == 0:
                        res_id = 0
                    else:
                        heavy_resindex = int(heavy[0].resindex)
                        res_id = None
                        for local_i, res in enumerate(mol.residues):
                            if int(res.resindex) == heavy_resindex:
                                res_id = local_i
                                break
                        if res_id is None:
                            res_id = 0

                    buckets[res_id].append(b.indices.copy())

                for res_id, res in enumerate(mol.residues):
                    kept = buckets.get(res_id, [])
                    beads[(mol_id, "united_atom", res_id)] = kept

            if "residue" in level_list:
                res_beads = self._hier.get_beads(mol, "residue")
                kept = []
                for i, b in enumerate(res_beads):
                    if len(b) == 0:
                        continue
                    kept.append(b.indices.copy())
                beads[(mol_id, "residue")] = kept

                if len(kept) == 0:
                    logger.error(
                        f"[BuildBeadsNode] NO residue beads kept for mol={mol_id}. "
                        "This will force residue entropy to 0.0."
                    )

            if "polymer" in level_list:
                poly_beads = self._hier.get_beads(mol, "polymer")
                kept = []
                for i, b in enumerate(poly_beads):
                    if len(b) == 0:
                        continue
                    kept.append(b.indices.copy())
                beads[(mol_id, "polymer")] = kept

        shared_data["beads"] = beads
        return {"beads": beads}
