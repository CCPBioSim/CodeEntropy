import logging
from typing import Any, Dict, Optional

import networkx as nx

from CodeEntropy.levels.nodes.frame_axes import FrameAxesNode
from CodeEntropy.levels.nodes.frame_covariance import FrameCovarianceNode

logger = logging.getLogger(__name__)


class FrameDAG:
    """
    Frame-local DAG (MAP stage).

    Contract:
      - execute_frame(shared_data, frame_index) builds a frame_ctx
      - frame_ctx ALWAYS contains:
          frame_ctx["shared"]      -> the shared_data dict
          frame_ctx["frame_index"] -> absolute trajectory frame index
      - nodes write only frame-local outputs:
          frame_ctx["frame_axes"], frame_ctx["frame_covariance"]
      - reduction/averaging happens outside this DAG (in LevelDAG)
    """

    def __init__(self, universe_operations=None):
        self._universe_operations = universe_operations
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Any] = {}

    def build(self) -> "FrameDAG":
        self._add("frame_axes", FrameAxesNode(self._universe_operations))
        self._add("frame_covariance", FrameCovarianceNode(), deps=["frame_axes"])
        return self

    def _add(self, name: str, node: Any, deps: Optional[list[str]] = None) -> None:
        self.nodes[name] = node
        self.graph.add_node(name)
        for d in deps or []:
            self.graph.add_edge(d, name)

    def execute_frame(
        self, shared_data: Dict[str, Any], frame_index: int
    ) -> Dict[str, Any]:
        frame_ctx: Dict[str, Any] = dict(shared_data)
        frame_ctx["shared"] = shared_data
        frame_ctx["frame_index"] = frame_index

        frame_ctx["frame_axes"] = None
        frame_ctx["frame_covariance"] = None

        for node_name in nx.topological_sort(self.graph):
            logger.debug(f"[FrameDAG] running {node_name} @ frame={frame_index}")
            self.nodes[node_name].run(frame_ctx)

        return frame_ctx["frame_covariance"]
