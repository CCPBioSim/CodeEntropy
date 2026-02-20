"""Entropy graph orchestration.

This module defines `EntropyGraph`, a small directed acyclic graph (DAG) that
executes entropy calculation nodes in dependency order.

The graph is intentionally simple:
  * Vibrational entropy
  * Configurational entropy
  * Aggregation of results

The nodes themselves encapsulate the detailed calculations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import networkx as nx

from CodeEntropy.entropy.nodes.aggregate import AggregateEntropyNode
from CodeEntropy.entropy.nodes.configurational import ConfigurationalEntropyNode
from CodeEntropy.entropy.nodes.vibrational import VibrationalEntropyNode

logger = logging.getLogger(__name__)


SharedData = Dict[str, Any]


@dataclass(frozen=True)
class NodeSpec:
    """Specification for a node within the entropy graph.

    Attributes:
        name: Unique node name.
        node: Node instance. Must implement `run(shared_data, **kwargs)`.
        deps: Optional list of node names that must run before this node.
    """

    name: str
    node: Any
    deps: tuple[str, ...] = ()


class EntropyGraph:
    """Build and execute the entropy calculation DAG.

    The graph is built once via `build()` and executed via `execute()`.

    Examples:
        graph = EntropyGraph().build()
        results = graph.execute(shared_data)
    """

    def __init__(self) -> None:
        """Initialize an empty entropy graph."""
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, Any] = {}

    def build(self) -> "EntropyGraph":
        """Populate the graph with the standard entropy workflow.

        Returns:
            Self for fluent chaining.
        """
        specs = (
            NodeSpec("vibrational_entropy", VibrationalEntropyNode()),
            NodeSpec("configurational_entropy", ConfigurationalEntropyNode()),
            NodeSpec(
                "aggregate_entropy",
                AggregateEntropyNode(),
                deps=("vibrational_entropy", "configurational_entropy"),
            ),
        )

        for spec in specs:
            self._add_node(spec)

        return self

    def execute(self, shared_data: SharedData) -> Dict[str, Any]:
        """Execute the entropy graph in topological order.

        Args:
            shared_data: Mutable shared data dictionary passed to each node.

        Returns:
            Dictionary containing the merged outputs of all nodes (only including
            outputs that are dict-like).

        Raises:
            KeyError: If a node name is missing from the internal node registry.
        """
        results: Dict[str, Any] = {}
        for node_name in nx.topological_sort(self._graph):
            node = self._nodes[node_name]
            logger.info("[EntropyGraph] node: %s", node_name)
            out = node.run(shared_data)
            if isinstance(out, dict):
                results.update(out)
        return results

    def _add_node(self, spec: NodeSpec) -> None:
        """Add a node and its dependencies to the graph.

        Args:
            spec: Node specification.

        Raises:
            ValueError: If a duplicate node name is added.
        """
        if spec.name in self._nodes:
            raise ValueError(f"Duplicate node name: {spec.name}")

        self._nodes[spec.name] = spec.node
        self._graph.add_node(spec.name)

        for dep in spec.deps:
            self._graph.add_edge(dep, spec.name)
