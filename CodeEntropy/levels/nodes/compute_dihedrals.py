from CodeEntropy.levels.dihedral_tools import DihedralTools


class ComputeDihedralsNode:
    def __init__(self):
        self._dih = DihedralTools()

    def run(self, shared_data):
        beads = shared_data["beads"]

        dihedrals = self._dih.get_dihedrals(beads)
        shared_data["dihedrals"] = dihedrals
