from CodeEntropy.levels.dihedral_tools import DihedralTools


class BuildConformationsNode:
    def __init__(self):
        self._dih = DihedralTools()

    def run(self, compute_dihedrals):
        return self._dih.build_conformational_states(compute_dihedrals["dihedrals"])
