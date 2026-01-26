# CodeEntropy/levels/hierarchy_graph.py

import logging

import networkx as nx

from CodeEntropy.levels.nodes.build_beads import BuildBeadsNode
from CodeEntropy.levels.nodes.build_covariance_matrices import (
    BuildCovarianceMatricesNode,
)
from CodeEntropy.levels.nodes.compute_dihedrals import ComputeConformationalStatesNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode
from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode

logger = logging.getLogger(__name__)


class LevelDAG:
    """
    Level-processing DAG.

    IMPORTANT:
    - This DAG is NOT "per frame".
    - Covariances require averaging over frames, so the frame-loop stays inside
      ForceTorqueManager.build_covariance_matrices(), preserving your original math.
    """

    def __init__(self, universe_operations):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self._uops = universe_operations

    def build(self):
        self.add("detect_molecules", DetectMoleculesNode())
        self.add("detect_levels", DetectLevelsNode(), ["detect_molecules"])

        self.add("build_beads", BuildBeadsNode(), ["detect_levels"])

        self.add(
            "compute_conformational_states",
            ComputeConformationalStatesNode(self._uops),
            ["detect_levels"],
        )

        self.add(
            "build_covariance",
            BuildCovarianceMatricesNode(self._uops),
            ["detect_levels"],
        )

        return self

    def add(self, name, obj, deps=None):
        self.nodes[name] = obj
        self.graph.add_node(name)
        if deps:
            for d in deps:
                self.graph.add_edge(d, name)

    def execute(self, shared_data):
        for node_name in nx.topological_sort(self.graph):
            logger.info(f"[LevelDAG] Running node: {node_name}")
            self.nodes[node_name].run(shared_data)

        return shared_data
