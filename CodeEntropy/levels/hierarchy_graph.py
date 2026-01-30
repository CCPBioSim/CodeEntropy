import logging
from typing import Any, Dict, Optional

import networkx as nx

from CodeEntropy.levels.nodes.build_beads import BuildBeadsNode
from CodeEntropy.levels.nodes.compute_dihedrals import ComputeConformationalStatesNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode
from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode
from CodeEntropy.levels.nodes.frame_covariance import FrameCovarianceNode
from CodeEntropy.levels.nodes.init_covariance_accumulators import (
    InitCovarianceAccumulatorsNode,
)

logger = logging.getLogger(__name__)


class LevelDAG:
    """
    Baseline two-stage DAG that matches original procedural behavior.

    Stage 1 (static): detect molecules, levels, build beads, init accumulators,
    conformational states
    Stage 2 (frame loop): for each frame, update running covariance means
    (forces/torques)
    """

    def __init__(self, universe_operations=None):
        self._universe_operations = universe_operations

        self.static_graph = nx.DiGraph()
        self.static_nodes: Dict[str, Any] = {}

        self.frame_graph = nx.DiGraph()
        self.frame_nodes: Dict[str, Any] = {}

    def build(self) -> "LevelDAG":
        # ---- static ----
        self._add_static("detect_molecules", DetectMoleculesNode())
        self._add_static("detect_levels", DetectLevelsNode(), deps=["detect_molecules"])
        self._add_static("build_beads", BuildBeadsNode(), deps=["detect_levels"])

        self._add_static(
            "init_covariance", InitCovarianceAccumulatorsNode(), deps=["detect_levels"]
        )

        # conformational states (trajectory scan inside)
        self._add_static(
            "compute_conformational_states",
            ComputeConformationalStatesNode(self._universe_operations),
            deps=["detect_levels"],
        )

        # ---- per-frame ----
        self._add_frame("frame_covariance", FrameCovarianceNode())

        return self

    def _add_static(
        self, name: str, node: Any, deps: Optional[list[str]] = None
    ) -> None:
        self.static_nodes[name] = node
        self.static_graph.add_node(name)
        for d in deps or []:
            self.static_graph.add_edge(d, name)

    def _add_frame(
        self, name: str, node: Any, deps: Optional[list[str]] = None
    ) -> None:
        self.frame_nodes[name] = node
        self.frame_graph.add_node(name)
        for d in deps or []:
            self.frame_graph.add_edge(d, name)

    def execute(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        # ---- run static stage ----
        for node_name in nx.topological_sort(self.static_graph):
            logger.info(f"[LevelDAG] Running static node: {node_name}")
            self.static_nodes[node_name].run(shared_data)

        # ---- frame loop (ONLY place trajectory advances) ----
        u = shared_data["reduced_universe"]
        start, end, step = shared_data["start"], shared_data["end"], shared_data["step"]

        for ts in u.trajectory[start:end:step]:
            shared_data["frame_index"] = ts.frame  # informational/debug only

            for node_name in nx.topological_sort(self.frame_graph):
                self.frame_nodes[node_name].run(shared_data)

        # outputs already accumulated in shared_data by FrameCovarianceNode
        return {
            "levels": shared_data["levels"],
            "beads": shared_data["beads"],
            "force_covariances": shared_data["force_covariances"],
            "torque_covariances": shared_data["torque_covariances"],
            "frame_counts": shared_data["frame_counts"],
            "conformational_states": shared_data["conformational_states"],
        }
