from CodeEntropy.levels.level_hierarchy import LevelHierarchy


class BuildBeadsNode:
    """
    Build bead definitions ONCE, in reduced_universe index space.

    shared_data["beads"] dict keys:
      (mol_id, "united_atom", res_id) -> list[np.ndarray]
      (mol_id, "residue")             -> list[np.ndarray]
      (mol_id, "polymer")             -> list[np.ndarray]
    """

    def __init__(self):
        self._hier = LevelHierarchy()

    def run(self, shared_data):
        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]

        beads = {}
        fragments = u.atoms.fragments

        for mol_id, level_list in enumerate(levels):
            mol = fragments[mol_id]

            if "united_atom" in level_list:
                for res_id, residue in enumerate(mol.residues):
                    ua_beads = self._hier.get_beads(residue.atoms, "united_atom")
                    beads[(mol_id, "united_atom", res_id)] = [
                        b.indices.copy() for b in ua_beads if len(b) > 0
                    ]

            if "residue" in level_list:
                res_beads = self._hier.get_beads(mol, "residue")
                beads[(mol_id, "residue")] = [
                    b.indices.copy() for b in res_beads if len(b) > 0
                ]

            if "polymer" in level_list:
                poly_beads = self._hier.get_beads(mol, "polymer")
                beads[(mol_id, "polymer")] = [
                    b.indices.copy() for b in poly_beads if len(b) > 0
                ]

        shared_data["beads"] = beads
        return {"beads": beads}
