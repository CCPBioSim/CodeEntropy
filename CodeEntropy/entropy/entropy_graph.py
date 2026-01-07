import networkx as nx

from CodeEntropy.entropy.nodes.configurational_entropy import ConfigurationalEntropyNode
from CodeEntropy.entropy.nodes.entropy_aggregator import EntropyAggregatorNode
from CodeEntropy.entropy.nodes.orientational_entropy import OrientationalEntropyNode
from CodeEntropy.entropy.nodes.vibrational_entropy import VibrationalEntropyNode


class EntropyGraph:
    """
    DAG representing the entropy computation pipeline:

        1. Vibrational entropy
        2. Rotational (orientational) entropy
        3. Conformational entropy
        4. Aggregate entropy across levels and groups
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}

    def build(self):
        self.add("vibrational_entropy", VibrationalEntropyNode())

        self.add(
            "orientational_entropy",
            OrientationalEntropyNode(),
            depends_on=["vibrational_entropy"],
        )

        self.add(
            "configurational_entropy",
            ConfigurationalEntropyNode(),
            depends_on=["orientational_entropy"],
        )

        self.add(
            "aggregate_entropy",
            EntropyAggregatorNode(),
            depends_on=["configurational_entropy"],
        )

        return self

    def add(self, name, obj, depends_on=None):
        self.nodes[name] = obj
        self.graph.add_node(name)
        if depends_on:
            for dep in depends_on:
                self.graph.add_edge(dep, name)

    def execute(self, shared_data):
        results = {}

        for node in nx.topological_sort(self.graph):
            preds = list(self.graph.predecessors(node))
            kwargs = {p: results[p] for p in preds}

            output = self.nodes[node].run(shared_data, **kwargs)
            results[node] = output

        return results
