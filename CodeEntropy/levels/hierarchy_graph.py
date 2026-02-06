import logging
from typing import Any, Dict, Optional

import networkx as nx

from CodeEntropy.levels.frame_dag import FrameDAG
from CodeEntropy.levels.nodes.build_beads import BuildBeadsNode
from CodeEntropy.levels.nodes.compute_dihedrals import ComputeConformationalStatesNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode
from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode
from CodeEntropy.levels.nodes.init_covariance_accumulators import (
    InitCovarianceAccumulatorsNode,
)

logger = logging.getLogger(__name__)


class LevelDAG:
    """
    STATIC DAG:
      detect_molecules -> detect_levels -> build_beads
                       -> init_covariance_accumulators
                       -> compute_conformational_states

    FRAME MAP DAG (parallelisable later):
      frame_axes -> frame_covariance

    REDUCE:
      incremental mean reduction into shared_data accumulators
    """

    def __init__(self, universe_operations=None):
        self._universe_operations = universe_operations

        self.static_graph = nx.DiGraph()
        self.static_nodes: Dict[str, Any] = {}

        self.frame_dag = FrameDAG(universe_operations=universe_operations)

    def build(self) -> "LevelDAG":
        self._add_static("detect_molecules", DetectMoleculesNode())
        self._add_static("detect_levels", DetectLevelsNode(), deps=["detect_molecules"])
        self._add_static("build_beads", BuildBeadsNode(), deps=["detect_levels"])
        self._add_static(
            "init_covariance_accumulators",
            InitCovarianceAccumulatorsNode(),
            deps=["detect_levels"],
        )
        self._add_static(
            "compute_conformational_states",
            ComputeConformationalStatesNode(self._universe_operations),
            deps=["detect_levels"],
        )

        self.frame_dag.build()
        return self

    def _add_static(
        self, name: str, node: Any, deps: Optional[list[str]] = None
    ) -> None:
        self.static_nodes[name] = node
        self.static_graph.add_node(name)
        for d in deps or []:
            self.static_graph.add_edge(d, name)

    @staticmethod
    def _inc_mean(old, new, n: int):
        return new.copy() if old is None else old + (new - old) / float(n)

    def _reduce_one_frame(
        self, shared_data: Dict[str, Any], frame_out: Dict[str, Any]
    ) -> None:
        f_cov = shared_data["force_covariances"]
        t_cov = shared_data["torque_covariances"]
        counts = shared_data["frame_counts"]
        gid2i = shared_data["group_id_to_index"]

        f_frame = frame_out["force"]
        t_frame = frame_out["torque"]

        # UA keyed by (group_id, res_id)
        for key, F in f_frame["ua"].items():
            counts["ua"][key] = counts["ua"].get(key, 0) + 1
            n = counts["ua"][key]
            f_cov["ua"][key] = self._inc_mean(f_cov["ua"].get(key), F, n)

        for key, T in t_frame["ua"].items():
            n = counts["ua"][key]
            t_cov["ua"][key] = self._inc_mean(t_cov["ua"].get(key), T, n)

        # residue / polymer indexed by contiguous group index
        for gid, F in f_frame["res"].items():
            gi = gid2i[gid]
            counts["res"][gi] += 1
            n = counts["res"][gi]
            f_cov["res"][gi] = self._inc_mean(f_cov["res"][gi], F, n)

        for gid, T in t_frame["res"].items():
            gi = gid2i[gid]
            n = counts["res"][gi]
            t_cov["res"][gi] = self._inc_mean(t_cov["res"][gi], T, n)

        for gid, F in f_frame["poly"].items():
            gi = gid2i[gid]
            counts["poly"][gi] += 1
            n = counts["poly"][gi]
            f_cov["poly"][gi] = self._inc_mean(f_cov["poly"][gi], F, n)

        for gid, T in t_frame["poly"].items():
            gi = gid2i[gid]
            n = counts["poly"][gi]
            t_cov["poly"][gi] = self._inc_mean(t_cov["poly"][gi], T, n)

    def execute(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        # --- STATIC DAG ---
        for node_name in nx.topological_sort(self.static_graph):
            logger.info(f"[LevelDAG] static node: {node_name}")
            self.static_nodes[node_name].run(shared_data)

        # --- FRAME MAP + REDUCE ---
        u = shared_data["reduced_universe"]
        start, end, step = shared_data["start"], shared_data["end"], shared_data["step"]

        for ts in u.trajectory[start:end:step]:
            frame_index = ts.frame
            frame_out = self.frame_dag.execute_frame(
                shared_data, frame_index=frame_index
            )
            self._reduce_one_frame(shared_data, frame_out)

        return shared_data
