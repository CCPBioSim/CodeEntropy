"""Hierarchy-level DAG orchestration and reduction.

This module defines the `LevelDAG`, which coordinates two stages of the hierarchy
workflow:

1) Static stage (runs once):
   - Detect molecules and available resolution levels.
   - Build beads for each (molecule, level) definition.
   - Initialise accumulators used during per-frame reduction.
   - Compute conformational state descriptors required later by entropy nodes.

2) Frame stage (runs for each trajectory frame):
   - Execute the `FrameGraph` to produce frame-local covariance outputs.
   - Reduce frame-local outputs into running (incremental) means.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import networkx as nx

from CodeEntropy.levels.axes import AxesCalculator
from CodeEntropy.levels.frame_dag import FrameGraph
from CodeEntropy.levels.nodes.accumulators import InitCovarianceAccumulatorsNode
from CodeEntropy.levels.nodes.beads import BuildBeadsNode
from CodeEntropy.levels.nodes.conformations import ComputeConformationalStatesNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode
from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode

logger = logging.getLogger(__name__)


class LevelDAG:
    """Execute hierarchy detection, per-frame covariance calculation, and reduction.

    The LevelDAG is responsible for:
      - Running a static DAG (once) to prepare shared inputs.
      - Running a per-frame DAG (for each frame) to compute frame-local outputs.
      - Reducing frame-local outputs into shared running means.

    The reduction performed here is an incremental mean across frames (and across
    molecules within a group when frame nodes average within-frame first).
    """

    def __init__(self, universe_operations: Optional[Any] = None) -> None:
        """Initialise a LevelDAG.

        Args:
            universe_operations: Optional adapter providing universe operations.
                Passed to the FrameGraph and the conformational-state node.
        """
        self._universe_operations = universe_operations

        self._static_graph = nx.DiGraph()
        self._static_nodes: Dict[str, Any] = {}

        self._frame_dag = FrameGraph(universe_operations=universe_operations)

    def build(self) -> "LevelDAG":
        """Build the static and frame DAG topology.

        Returns:
            Self, to allow fluent chaining.
        """
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

        self._frame_dag.build()
        return self

    def execute(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full hierarchy workflow and mutate shared_data.

        Args:
            shared_data: Shared workflow data dict that will be mutated by the DAG.

        Returns:
            The mutated shared_data dict.
        """
        shared_data.setdefault("axes_manager", AxesCalculator())
        self._run_static_stage(shared_data)
        self._run_frame_stage(shared_data)
        return shared_data

    def _run_static_stage(self, shared_data: Dict[str, Any]) -> None:
        """Run all static nodes in dependency order."""
        for node_name in nx.topological_sort(self._static_graph):
            self._static_nodes[node_name].run(shared_data)

    def _add_static(
        self, name: str, node: Any, deps: Optional[list[str]] = None
    ) -> None:
        """Register a static node and its dependencies in the static DAG."""
        self._static_nodes[name] = node
        self._static_graph.add_node(name)
        for dep in deps or []:
            self._static_graph.add_edge(dep, name)

    def _run_frame_stage(self, shared_data: Dict[str, Any]) -> None:
        """Run the frame DAG for each selected trajectory frame and reduce outputs."""
        u = shared_data["reduced_universe"]
        start, end, step = shared_data["start"], shared_data["end"], shared_data["step"]

        for ts in u.trajectory[start:end:step]:
            frame_index = ts.frame
            frame_out = self._frame_dag.execute_frame(shared_data, frame_index)
            self._reduce_one_frame(shared_data, frame_out)

    @staticmethod
    def _incremental_mean(old: Any, new: Any, n: int) -> Any:
        """Compute an incremental mean.

        Args:
            old: Previous running mean (or None for first sample).
            new: New sample to incorporate.
            n: 1-based sample count after adding `new`.

        Returns:
            Updated running mean.
        """
        if old is None:
            return new.copy() if hasattr(new, "copy") else new
        return old + (new - old) / float(n)

    def _reduce_one_frame(
        self, shared_data: Dict[str, Any], frame_out: Dict[str, Any]
    ) -> None:
        """Reduce one frame's covariance outputs into shared running means.

        Args:
            shared_data: Shared workflow data dict containing accumulators.
            frame_out: Frame-local covariance outputs produced by FrameGraph.
        """
        self._reduce_force_and_torque(shared_data, frame_out)
        self._reduce_forcetorque(shared_data, frame_out)

    def _reduce_force_and_torque(
        self, shared_data: Dict[str, Any], frame_out: Dict[str, Any]
    ) -> None:
        """Reduce force/torque covariance outputs into shared accumulators."""
        f_cov = shared_data["force_covariances"]
        t_cov = shared_data["torque_covariances"]
        counts = shared_data["frame_counts"]
        gid2i = shared_data["group_id_to_index"]

        f_frame = frame_out["force"]
        t_frame = frame_out["torque"]

        for key, F in f_frame["ua"].items():
            counts["ua"][key] = counts["ua"].get(key, 0) + 1
            n = counts["ua"][key]
            f_cov["ua"][key] = self._incremental_mean(f_cov["ua"].get(key), F, n)

        for key, T in t_frame["ua"].items():
            if key not in counts["ua"]:
                counts["ua"][key] = counts["ua"].get(key, 0) + 1
            n = counts["ua"][key]
            t_cov["ua"][key] = self._incremental_mean(t_cov["ua"].get(key), T, n)

        for gid, F in f_frame["res"].items():
            gi = gid2i[gid]
            counts["res"][gi] += 1
            n = counts["res"][gi]
            f_cov["res"][gi] = self._incremental_mean(f_cov["res"][gi], F, n)

        for gid, T in t_frame["res"].items():
            gi = gid2i[gid]
            if counts["res"][gi] == 0:
                counts["res"][gi] += 1
            n = counts["res"][gi]
            t_cov["res"][gi] = self._incremental_mean(t_cov["res"][gi], T, n)

        for gid, F in f_frame["poly"].items():
            gi = gid2i[gid]
            counts["poly"][gi] += 1
            n = counts["poly"][gi]
            f_cov["poly"][gi] = self._incremental_mean(f_cov["poly"][gi], F, n)

        for gid, T in t_frame["poly"].items():
            gi = gid2i[gid]
            if counts["poly"][gi] == 0:
                counts["poly"][gi] += 1
            n = counts["poly"][gi]
            t_cov["poly"][gi] = self._incremental_mean(t_cov["poly"][gi], T, n)

    def _reduce_forcetorque(
        self, shared_data: Dict[str, Any], frame_out: Dict[str, Any]
    ) -> None:
        """Reduce combined force-torque covariance outputs into shared accumulators."""
        if "forcetorque" not in frame_out:
            return

        ft_cov = shared_data["forcetorque_covariances"]
        ft_counts = shared_data["forcetorque_counts"]
        gid2i = shared_data["group_id_to_index"]
        ft_frame = frame_out["forcetorque"]

        for gid, M in ft_frame.get("res", {}).items():
            gi = gid2i[gid]
            ft_counts["res"][gi] += 1
            n = ft_counts["res"][gi]
            ft_cov["res"][gi] = self._incremental_mean(ft_cov["res"][gi], M, n)

        for gid, M in ft_frame.get("poly", {}).items():
            gi = gid2i[gid]
            ft_counts["poly"][gi] += 1
            n = ft_counts["poly"][gi]
            ft_cov["poly"][gi] = self._incremental_mean(ft_cov["poly"][gi], M, n)
