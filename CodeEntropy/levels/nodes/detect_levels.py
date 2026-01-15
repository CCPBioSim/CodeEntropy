from CodeEntropy.levels.level_hierarchy import LevelHierarchy


class DetectLevelsNode:
    def __init__(self):
        self._hier = LevelHierarchy()

    def run(self, shared_data):
        u = shared_data["universe"]

        num_mol, levels = self._hier.select_levels(u)

        shared_data["number_molecules"] = num_mol
        shared_data["levels"] = levels
