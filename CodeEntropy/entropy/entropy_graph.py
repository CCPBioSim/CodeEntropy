import logging
from typing import Any, Dict, Optional

import networkx as nx

from CodeEntropy.entropy.nodes.aggregate_entropy_node import AggregateEntropyNode
from CodeEntropy.entropy.nodes.configurational_entropy_node import (
    ConfigurationalEntropyNode,
)
from CodeEntropy.entropy.nodes.vibrational_entropy_node import VibrationalEntropyNode

logger = logging.getLogger(__name__)


class EntropyGraph:
    """
    Entropy DAG (simple, stable):
        vibrational_entropy
        configurational_entropy
        aggregate_entropy
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Any] = {}

    def build(self) -> "EntropyGraph":
        self._add("vibrational_entropy", VibrationalEntropyNode())
        self._add("configurational_entropy", ConfigurationalEntropyNode())
        self._add(
            "aggregate_entropy",
            AggregateEntropyNode(),
            deps=["vibrational_entropy", "configurational_entropy"],
        )
        return self

    def _add(self, name: str, node: Any, deps: Optional[list[str]] = None) -> None:
        self.nodes[name] = node
        self.graph.add_node(name)
        for d in deps or []:
            self.graph.add_edge(d, name)

    def execute(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for node_name in nx.topological_sort(self.graph):
            logger.info(f"[EntropyGraph] node: {node_name}")
            out = self.nodes[node_name].run(shared_data)
            if isinstance(out, dict):
                results.update(out)
        return results
