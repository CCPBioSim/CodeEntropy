from CodeEntropy.levels.neighbours import NeighbourList


class ComputeNeighboursNode:
    def __init__(self):
        self._nb = NeighbourList()

    def run(self, shared_data):
        beads = shared_data["beads"]

        neighbours = self._nb.compute(beads)
        shared_data["neighbours"] = neighbours
