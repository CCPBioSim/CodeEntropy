"""Frame-local DAG execution.

This module defines the frame-scoped DAG used during the MAP stage of the
hierarchy workflow. Each frame is processed independently to produce
frame-local outputs (e.g., axes and covariance data), which are later reduced
outside this DAG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

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

    shared: dict[str, Any]
    frame_index: int
    frame_covariance: Any = None
    data: dict[str, Any] | None = None


class FrameGraph:
    """Execute a frame-local directed acyclic graph.

    The graph is run once per trajectory frame. Nodes may read shared inputs from
    `ctx["shared"]` and must write only frame-local outputs into the frame context.

    Expected node outputs:
      - "frame_covariance"
    """

    def __init__(self, universe_operations: Any | None = None) -> None:
        """Initialise a FrameGraph.

        Args:
            universe_operations: Optional adapter providing universe operations used
                by frame-level nodes (e.g., selections / molecule containers).
        """
        self._universe_operations = universe_operations
        self._graph = nx.DiGraph()
        self._nodes: dict[str, Any] = {}

    def build(self) -> FrameGraph:
        """Build the default frame DAG topology.

        Returns:
            Self, to allow fluent chaining.
        """
        self._add("frame_covariance", FrameCovarianceNode())
        return self

    def execute_frame(self, shared_data: dict[str, Any], frame_index: int) -> Any:
        """Execute the frame DAG for a single trajectory frame.

        FrameGraph owns MDAnalysis trajectory positioning for frame-local execution.
        Higher-level orchestration passes frame indices, but must not rely on hidden
        trajectory cursor state.

        Args:
            shared_data: Shared workflow data dictionary. Must contain
                ``"reduced_universe"``.
            frame_index: Frame index to process. At this migration stage this is a
                local index into the already frame-reduced universe.

        Returns:
            Frame-local covariance payload produced by ``FrameCovarianceNode``.

        Raises:
            KeyError: If ``"reduced_universe"`` is missing from ``shared_data``.
            IndexError: If ``frame_index`` is outside the trajectory bounds.
        """
        universe = shared_data["reduced_universe"]
        frame_index = int(frame_index)

        try:
            universe.trajectory[frame_index]
        except IndexError as exc:
            n_frames = len(universe.trajectory)
            raise IndexError(
                f"Frame index {frame_index} is outside trajectory bounds "
                f"for trajectory with {n_frames} frames."
            ) from exc

        ctx = self._make_frame_ctx(
            shared_data=shared_data,
            frame_index=frame_index,
        )

        for node_name in nx.topological_sort(self._graph):
            logger.debug("[FrameGraph] running %s @ frame=%s", node_name, frame_index)
            self._nodes[node_name].run(ctx)

        return ctx["frame_covariance"]

    def _add(self, name: str, node: Any, deps: list[str] | None = None) -> None:
        """Register a node and its dependencies in the DAG."""
        self._nodes[name] = node
        self._graph.add_node(name)
        for dep in deps or []:
            self._graph.add_edge(dep, name)

    @staticmethod
    def _make_frame_ctx(
        shared_data: dict[str, Any], frame_index: int
    ) -> dict[str, Any]:
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
