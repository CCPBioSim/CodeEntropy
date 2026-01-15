from CodeEntropy.levels.neighbours import Neighbours


class ComputeNeighboursNode:
    def __init__(self):
        self._nb = Neighbours()

    def run(self, shared_data):
        beads = shared_data["beads"]
        shared_data["neighbours"] = self._nb.compute(beads)
