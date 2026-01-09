import networkx as nx

from CodeEntropy.levels.nodes.build_beads import BuildBeadsNode
from CodeEntropy.levels.nodes.build_conformations import BuildConformationsNode
from CodeEntropy.levels.nodes.build_covariance_matrices import (
    BuildCovarianceMatricesNode,
)
from CodeEntropy.levels.nodes.compute_axes import ComputeAxesNode
from CodeEntropy.levels.nodes.compute_dihedrals import ComputeDihedralsNode
from CodeEntropy.levels.nodes.compute_neighbours import ComputeNeighboursNode
from CodeEntropy.levels.nodes.compute_weighted_forces import ComputeWeightedForcesNode
from CodeEntropy.levels.nodes.compute_weighted_torques import ComputeWeightedTorquesNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode
from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode


class LevelDAG:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}

    def build(self):
        self.add("detect_molecules", DetectMoleculesNode())
        self.add("detect_levels", DetectLevelsNode(), ["detect_molecules"])
        self.add("build_beads", BuildBeadsNode(), ["detect_levels"])
        self.add("compute_axes", ComputeAxesNode(), ["build_beads"])
        self.add(
            "compute_weighted_forces", ComputeWeightedForcesNode(), ["compute_axes"]
        )
        self.add(
            "compute_weighted_torques", ComputeWeightedTorquesNode(), ["compute_axes"]
        )
        self.add(
            "build_covariance",
            BuildCovarianceMatricesNode(),
            ["compute_weighted_forces", "compute_weighted_torques"],
        )
        self.add("compute_dihedrals", ComputeDihedralsNode(), ["build_beads"])
        self.add("build_conformations", BuildConformationsNode(), ["compute_dihedrals"])
        self.add("compute_neighbours", ComputeNeighboursNode(), ["build_beads"])

        return self

    def add(self, name, obj, deps=None):
        self.nodes[name] = obj
        self.graph.add_node(name)
        if deps:
            for d in deps:
                self.graph.add_edge(d, name)

    def execute(self, shared_data):
        for node in nx.topological_sort(self.graph):
            self.nodes[node].run(shared_data)
