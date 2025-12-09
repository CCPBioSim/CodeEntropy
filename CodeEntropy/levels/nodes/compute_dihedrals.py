from CodeEntropy.levels.dihedral_tools import DihedralTools


class ComputeDihedralsNode:
    def __init__(self):
        self._dih = DihedralTools()

    def run(self, build_beads):
        return {"dihedrals": self._dih.get_dihedrals(build_beads["beads"])}
