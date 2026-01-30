# CodeEntropy/entropy/entropy_graph.py

import networkx as nx

from CodeEntropy.entropy.nodes.aggregate_entropy_node import AggregateEntropyNode
from CodeEntropy.entropy.nodes.configurational_entropy_node import (
    ConfigurationalEntropyNode,
)
from CodeEntropy.entropy.nodes.vibrational_entropy_node import VibrationalEntropyNode


class EntropyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}

    def build(self):
        self.add("vibrational_entropy", VibrationalEntropyNode())
        self.add(
            "configurational_entropy",
            ConfigurationalEntropyNode(),
            depends_on=["vibrational_entropy"],
        )
        self.add(
            "aggregate_entropy",
            AggregateEntropyNode(),
            depends_on=["configurational_entropy"],
        )
        return self

    def add(self, name, obj, depends_on=None):
        self.nodes[name] = obj
        self.graph.add_node(name)
        for dep in depends_on or []:
            self.graph.add_edge(dep, name)

    def execute(self, shared_data):
        results = {}
        for node in nx.topological_sort(self.graph):
            preds = list(self.graph.predecessors(node))
            kwargs = {p: results[p] for p in preds}
            results[node] = self.nodes[node].run(shared_data, **kwargs)
        return results
