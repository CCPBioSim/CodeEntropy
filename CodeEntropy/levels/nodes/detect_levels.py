from CodeEntropy.levels.level_hierarchy import LevelHierarchy


class DetectLevelsNode:
    def __init__(self):
        self._hier = LevelHierarchy()

    def run(self, shared_data):
        u = shared_data["reduced_universe"]
        n_mol, levels = self._hier.select_levels(u)
        shared_data["levels"] = levels
        shared_data["number_molecules"] = n_mol
        return {"levels": levels, "number_molecules": n_mol}
