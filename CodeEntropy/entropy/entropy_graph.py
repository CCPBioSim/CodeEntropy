import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
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
        self.nodes = {}

    def add_node(self, name: str, obj: Any, depends_on=None):
        """Add a computational node to the DAG."""
        if not hasattr(obj, "run"):
            raise TypeError(
                f"Node '{name}' must implement a `run(shared_data, **kwargs)` method."
            )

        self.nodes[name] = obj
        self.graph.add_node(name)

        if depends_on:
            for dep in depends_on:
                if dep not in self.graph:
                    raise ValueError(f"Dependency '{dep}' not found for node '{name}'.")
                self.graph.add_edge(dep, name)

        logger.debug(f"Added node '{name}' with dependencies: {depends_on}")

    def execute(self, shared_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Execute nodes in topological order."""
        logger.info("Executing EntropyGraph DAG...")
        results: Dict[str, Dict[str, Any]] = {}

        for node in nx.topological_sort(self.graph):
            preds = list(self.graph.predecessors(node))
            kwargs = {p: results[p] for p in preds}

            node_obj = self.nodes[node]
            logger.info(f"Running node: {node} (depends on: {preds})")

            output = node_obj.run(shared_data, **kwargs)

            if not isinstance(output, dict):
                raise TypeError(
                    f"Node '{node}' returned {type(output)}; must return dict."
                )

            results[node] = output

        logger.info("DAG execution complete.")
        return results

    def visualize(self, show=True, figsize=(8, 6)):
        """Visualize the DAG using matplotlib."""

        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=figsize)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=2800,
            arrows=True,
            font_weight="bold",
        )
        plt.title("Entropy Computation Graph", fontsize=14)
        if show:
            plt.show()
