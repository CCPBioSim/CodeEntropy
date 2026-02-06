# CodeEntropy/entropy/entropy_graph.py

from __future__ import annotations

import networkx as nx

from CodeEntropy.entropy.nodes.aggregate_entropy_node import AggregateEntropyNode
from CodeEntropy.entropy.nodes.configurational_entropy_node import (
    ConfigurationalEntropyNode,
)
from CodeEntropy.entropy.nodes.vibrational_entropy_node import VibrationalEntropyNode


class EntropyGraph:
    """
    Entropy DAG.

    Nodes operate on shared_data (produced by EntropyManager + LevelDAG) and
    write results to DataLogger.

    Graph:
      vibrational_entropy  ----\
                                -> aggregate_entropy
      configurational_entropy --/
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}

    def build(self) -> "EntropyGraph":
        self.nodes["vibrational_entropy"] = VibrationalEntropyNode()
        self.nodes["configurational_entropy"] = ConfigurationalEntropyNode()
        self.nodes["aggregate_entropy"] = AggregateEntropyNode()

        for n in self.nodes:
            self.graph.add_node(n)

        self.graph.add_edge("vibrational_entropy", "aggregate_entropy")
        self.graph.add_edge("configurational_entropy", "aggregate_entropy")

        return self

    def execute(self, shared_data):
        results = {}
        for node in nx.topological_sort(self.graph):
            preds = list(self.graph.predecessors(node))
            kwargs = {p: results[p] for p in preds}
            results[node] = self.nodes[node].run(shared_data, **kwargs)
        return results
