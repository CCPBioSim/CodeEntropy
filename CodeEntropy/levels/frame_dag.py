# CodeEntropy/levels/frame_dag.py

import networkx as nx


class FrameDAG:
    """
    Per-frame DAG (MAP stage).
    Should NOT mutate global running averages.
    It returns a per-frame result dictionary.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}

    def add(self, name, node, deps=None):
        self.nodes[name] = node
        self.graph.add_node(name)
        for d in deps or []:
            self.graph.add_edge(d, name)

    def build(self, frame_covariance_node):
        self.add("frame_covariance", frame_covariance_node)
        return self

    def execute(self, shared_data):
        results = {}
        for node_name in nx.topological_sort(self.graph):
            out = self.nodes[node_name].run(shared_data)
            results[node_name] = out
        return results
