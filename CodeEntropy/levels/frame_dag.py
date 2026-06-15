"""Frame-local DAG execution.

This module defines the frame-scoped DAG used during the MAP stage of the
hierarchy workflow. Each selected frame is processed independently to produce
frame-local observable outputs, which are reduced outside this DAG.

FrameGraph owns trajectory positioning. It does not own scheduling, chunking, or
reduction.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from CodeEntropy.levels.nodes.covariance import FrameCovarianceNode


class FrameGraph:
    """Execute the frame-local directed acyclic graph."""

    def __init__(self, universe_operations: Any | None = None) -> None:
        self._universe_operations = universe_operations
        self._graph = nx.DiGraph()
        self._nodes: dict[str, Any] = {}

    def build(self) -> FrameGraph:
        """Build the default frame DAG topology."""
        self._add("frame_covariance", FrameCovarianceNode())
        return self

    def execute_frame(self, shared_data: dict[str, Any], frame_index: int) -> Any:
        """Execute the frame DAG for one selected analysis frame.

        Args:
            shared_data: Shared workflow data. Must contain ``"frame_source"``.
            frame_index: Frame index in the active analysis frame-source space.

        Returns:
            Frame-local covariance payload produced by ``FrameCovarianceNode``.

        Raises:
            KeyError: If ``"frame_source"`` is missing from ``shared_data``.
            IndexError: If ``frame_index`` is outside the active trajectory bounds.
        """
        frame_source = shared_data["frame_source"]
        frame_index = int(frame_index)

        try:
            frame_source.seek(frame_index)
        except IndexError as exc:
            n_frames = len(frame_source.universe.trajectory)
            raise IndexError(
                f"Frame index {frame_index} is outside analysis trajectory bounds "
                f"for trajectory with {n_frames} frames."
            ) from exc

        ctx = self._make_frame_ctx(shared_data=shared_data, frame_index=frame_index)

        for node_name in nx.topological_sort(self._graph):
            self._nodes[node_name].run(ctx)

        return ctx["frame_covariance"]

    def _add(self, name: str, node: Any, deps: list[str] | None = None) -> None:
        """Register a frame-local node and its dependencies."""
        self._nodes[name] = node
        self._graph.add_node(name)
        for dep in deps or []:
            self._graph.add_edge(dep, name)

    @staticmethod
    def _make_frame_ctx(
        shared_data: dict[str, Any],
        frame_index: int,
    ) -> dict[str, Any]:
        """Create a frame-local execution context."""
        return {
            "shared": shared_data,
            "frame_index": frame_index,
            "frame_covariance": None,
        }
