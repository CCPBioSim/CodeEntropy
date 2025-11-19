import logging
from typing import Any, Dict

import networkx as nx

logger = logging.getLogger(__name__)


class EntropyGraph:
    """
    A Directed Acyclic Graph (DAG) for managing entropy calculation nodes.

    Each node must implement:
        run(shared_data: dict, **kwargs) -> dict
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Any] = {}

    def add_node(self, name: str, node_obj: Any, depends_on=None):
        if not hasattr(node_obj, "run"):
            raise TypeError(f"Node '{name}' must implement run(shared_data, **kwargs)")

        self.nodes[name] = node_obj
        self.graph.add_node(name)
        if depends_on:
            for dep in depends_on:
                self.graph.add_edge(dep, name)

    def execute(self, shared_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        for node_name in nx.topological_sort(self.graph):
            preds = list(self.graph.predecessors(node_name))
            kwargs = {p: results[p] for p in preds}

            node = self.nodes[node_name]
            output = node.run(shared_data, **kwargs)

            if not isinstance(output, dict):
                raise TypeError(
                    f"Node '{node_name}' returned {type(output)} (expected dict)"
                )

            results[node_name] = output
            shared_data.update(output)

        return results
