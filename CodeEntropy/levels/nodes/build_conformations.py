from CodeEntropy.levels.dihedral_tools import DihedralTools


class BuildConformationsNode:
    def __init__(self):
        self._dih = DihedralTools()

    def run(self, shared_data):
        dihedrals = shared_data["dihedrals"]
        shared_data["conformations"] = self._dih.build_conformational_states(dihedrals)
