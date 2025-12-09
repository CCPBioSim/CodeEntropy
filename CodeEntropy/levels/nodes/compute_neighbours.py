from CodeEntropy.levels.neighbours import NeighbourList


class ComputeNeighboursNode:
    def __init__(self):
        self._nb = NeighbourList()

    def run(self, build_beads):
        return {"neighbours": self._nb.compute(build_beads["beads"])}
