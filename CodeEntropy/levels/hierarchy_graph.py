from CodeEntropy.entropy.entropy_graph import EntropyGraph
from CodeEntropy.levels.nodes.build_beads import BuildBeadsNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode


class HierarchyGraph:
    """
    DAG for level / bead / structural preparation.
    """

    def __init__(self):
        self.graph = EntropyGraph()

    def build(self, run_manager):
        self.graph.add_node("detect_levels", DetectLevelsNode())
        self.graph.add_node(
            "build_beads", BuildBeadsNode(run_manager), depends_on=["detect_levels"]
        )
        return self.graph
