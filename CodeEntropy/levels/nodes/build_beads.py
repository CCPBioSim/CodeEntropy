from CodeEntropy.levels.level_hierarchy import LevelHierarchy
from CodeEntropy.levels.mda_universe_operations import UniverseOperations


class BuildBeadsNode:
    def __init__(self):
        self._hier = LevelHierarchy()
        self._mda = UniverseOperations()

    def run(self, shared_data, detect_levels):
        u = shared_data["universe"]
        levels = detect_levels["levels"]
        beads = {}

        for mol_id, level_list in enumerate(levels):
            mol_u = self._mda.get_molecule_container(u, mol_id)
            for level in level_list:
                beads[(mol_id, level)] = self._hier.get_beads(mol_u, level)

        return {"beads": beads}
