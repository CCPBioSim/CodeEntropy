"""Frame-local DAG execution.

This module defines the frame-scoped DAG used during the MAP stage of the
hierarchy workflow. Each frame is processed independently to produce
frame-local outputs (e.g., axes and covariance data), which are later reduced
outside this DAG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import networkx as nx

from CodeEntropy.levels.nodes.covariance import FrameCovarianceNode

logger = logging.getLogger(__name__)


@dataclass
class FrameContext:
    """Container for per-frame execution context.

    Attributes:
        shared: Shared workflow data (mutated across the full workflow).
        frame_index: Absolute trajectory frame index being processed.
        frame_covariance: Frame-local covariance output produced by FrameCovarianceNode.
        data: Additional frame-local scratch space for nodes, if needed.
    """

    shared: Dict[str, Any]
    frame_index: int
    frame_covariance: Any = None
    data: Dict[str, Any] = None


class FrameDAG:
    """Execute a frame-local directed acyclic graph.

    The graph is run once per trajectory frame. Nodes may read shared inputs from
    `ctx["shared"]` and must write only frame-local outputs into the frame context.

    Expected node outputs:
      - "frame_covariance"
    """

    def __init__(self, universe_operations: Optional[Any] = None) -> None:
        """Initialise a FrameDAG.

        Args:
            universe_operations: Optional adapter providing universe operations used
                by frame-level nodes (e.g., selections / molecule containers).
        """
        self._universe_operations = universe_operations
        self._graph = nx.DiGraph()
        self._nodes: Dict[str, Any] = {}

    def build(self) -> "FrameDAG":
        """Build the default frame DAG topology.

        Returns:
            Self, to allow fluent chaining.
        """
        self._add("frame_covariance", FrameCovarianceNode())
        return self

    def execute_frame(self, shared_data: Dict[str, Any], frame_index: int) -> Any:
        """Execute the frame DAG for a single trajectory frame.

        Args:
            shared_data: Shared workflow data dict.
            frame_index: Absolute trajectory frame index.

        Returns:
            Frame-local covariance payload produced by FrameCovarianceNode.
        """
        ctx = self._make_frame_ctx(shared_data=shared_data, frame_index=frame_index)

        for node_name in nx.topological_sort(self._graph):
            logger.debug("[FrameDAG] running %s @ frame=%s", node_name, frame_index)
            self._nodes[node_name].run(ctx)

        return ctx["frame_covariance"]

    def _add(self, name: str, node: Any, deps: Optional[list[str]] = None) -> None:
        """Register a node and its dependencies in the DAG."""
        self._nodes[name] = node
        self._graph.add_node(name)
        for dep in deps or []:
            self._graph.add_edge(dep, name)

    @staticmethod
    def _make_frame_ctx(
        shared_data: Dict[str, Any], frame_index: int
    ) -> Dict[str, Any]:
        """Create a frame context dictionary for node execution.

        Notes:
            - The context includes a reference to `shared_data` via "shared".
            - The context is intentionally frame-scoped and should not be used as
              a replacement for shared workflow state.

        Args:
            shared_data: Shared workflow data dict.
            frame_index: Absolute trajectory frame index.

        Returns:
            Frame context dict with required keys.
        """
        return {
            "shared": shared_data,
            "frame_index": frame_index,
            "frame_covariance": None,
        }
