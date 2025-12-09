from CodeEntropy.levels.level_hierarchy import LevelHierarchy


class DetectLevelsNode:
    def __init__(self):
        self._hier = LevelHierarchy()

    def run(self, shared_data, detect_molecules):
        u = shared_data["universe"]
        num_mol, levels = self._hier.select_levels(u)

        return {
            "number_molecules": num_mol,
            "levels": levels,
            "fragments": detect_molecules["fragments"],
        }
