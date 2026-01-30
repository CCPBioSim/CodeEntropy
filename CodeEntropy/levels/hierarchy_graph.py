# CodeEntropy/levels/hierarchy_graph.py

import logging

import networkx as nx

from CodeEntropy.levels.frame_dag import FrameDAG
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
    Full level pipeline:

      STATIC DAG (once)
        -> prepares molecules, levels, beads, conformational states,
        and accumulator shapes

      FRAME MAP DAG (per frame, parallelisable)
        -> returns per-frame covariance contributions

      REDUCE step (once)
        -> reduces per-frame contributions into running means identical to original
    """

    def __init__(self, universe_operations):
        self._universe_operations = universe_operations

        self.static_graph = nx.DiGraph()
        self.static_nodes = {}

        # A separate per-frame DAG
        self.frame_dag = FrameDAG().build(FrameCovarianceNode())

    def add_static(self, name, node, deps=None):
        self.static_nodes[name] = node
        self.static_graph.add_node(name)
        for d in deps or []:
            self.static_graph.add_edge(d, name)

    def build(self):
        # STATIC DAG
        self.add_static("detect_molecules", DetectMoleculesNode())
        self.add_static("detect_levels", DetectLevelsNode(), deps=["detect_molecules"])
        self.add_static("build_beads", BuildBeadsNode(), deps=["detect_levels"])
        self.add_static(
            "init_covariance_accumulators",
            InitCovarianceAccumulatorsNode(),
            deps=["detect_levels"],
        )

        # Conformational states scans the trajectory internally (not frame-local)
        self.add_static(
            "compute_conformational_states",
            ComputeConformationalStatesNode(self._universe_operations),
            deps=["detect_levels"],
        )
        return self

    @staticmethod
    def _inc_mean(avg, new, n):
        return new.copy() if avg is None else avg + (new - avg) / float(n)

    def _reduce_one_frame(self, shared_data, frame_out):
        """
        Reduce MAP output into running means (matches your original).
        """
        f_cov = shared_data["force_covariances"]
        t_cov = shared_data["torque_covariances"]
        counts = shared_data["frame_counts"]

        f_frame = frame_out["force"]
        t_frame = frame_out["torque"]

        # UA: dict keyed by (group_id, res_id)
        for key, F in f_frame["ua"].items():
            counts["ua"][key] = counts["ua"].get(key, 0) + 1
            n = counts["ua"][key]
            f_cov["ua"][key] = self._inc_mean(f_cov["ua"].get(key), F, n)

        for key, T in t_frame["ua"].items():
            # same counter as force for UA key
            n = counts["ua"][key]
            t_cov["ua"][key] = self._inc_mean(t_cov["ua"].get(key), T, n)

        # residue/poly: arrays indexed by group_id
        for gid, F in f_frame["res"].items():
            counts["res"][gid] += 1
            n = counts["res"][gid]
            f_cov["res"][gid] = self._inc_mean(f_cov["res"][gid], F, n)

        for gid, T in t_frame["res"].items():
            n = counts["res"][gid]
            t_cov["res"][gid] = self._inc_mean(t_cov["res"][gid], T, n)

        for gid, F in f_frame["poly"].items():
            counts["poly"][gid] += 1
            n = counts["poly"][gid]
            f_cov["poly"][gid] = self._inc_mean(f_cov["poly"][gid], F, n)

        for gid, T in t_frame["poly"].items():
            n = counts["poly"][gid]
            t_cov["poly"][gid] = self._inc_mean(t_cov["poly"][gid], T, n)

    def execute(self, shared_data):
        # --- Run STATIC DAG ---
        for node_name in nx.topological_sort(self.static_graph):
            logger.info(f"[LevelDAG] STATIC: {node_name}")
            self.static_nodes[node_name].run(shared_data)

        # --- Frame MAP loop ---
        u = shared_data["reduced_universe"]
        start, end, step = shared_data["start"], shared_data["end"], shared_data["step"]

        for ts in u.trajectory[start:end:step]:
            frame_results = self.frame_dag.execute(shared_data)
            frame_cov = frame_results["frame_covariance"]

            self._reduce_one_frame(shared_data, frame_cov)

        return {
            "force_covariances": shared_data["force_covariances"],
            "torque_covariances": shared_data["torque_covariances"],
            "frame_counts": shared_data["frame_counts"],
            "conformational_states": shared_data.get("conformational_states"),
            "levels": shared_data["levels"],
        }

    def describe(self):
        static_edges = list(self.static_graph.edges())
        frame_edges = list(self.frame_dag.graph.edges())
        return {
            "static_nodes": list(self.static_graph.nodes()),
            "static_edges": static_edges,
            "frame_nodes": list(self.frame_dag.graph.nodes()),
            "frame_edges": frame_edges,
        }
