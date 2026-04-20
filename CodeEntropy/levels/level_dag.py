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
from typing import Any

import networkx as nx
from rich.progress import TaskID

from CodeEntropy.levels.axes import AxesCalculator
from CodeEntropy.levels.frame_dag import FrameGraph
from CodeEntropy.levels.nodes.accumulators import InitCovarianceAccumulatorsNode
from CodeEntropy.levels.nodes.beads import BuildBeadsNode
from CodeEntropy.levels.nodes.conformations import ComputeConformationalStatesNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode
from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode
from CodeEntropy.levels.nodes.find_neighbors import ComputeNeighborsNode
from CodeEntropy.results.reporter import _RichProgressSink

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

    def __init__(self, universe_operations: Any | None = None) -> None:
        """Initialise a LevelDAG.

        Args:
            universe_operations: Optional adapter providing universe operations.
                Passed to the FrameGraph and the conformational-state node.
        """
        self._universe_operations = universe_operations

        self._static_graph = nx.DiGraph()
        self._static_nodes: dict[str, Any] = {}

        self._frame_dag = FrameGraph(universe_operations=universe_operations)

    def build(self) -> LevelDAG:
        """Build the static and frame DAG topology.

        This registers all static nodes and their dependencies, and builds the
        internal FrameGraph used for per-frame execution.

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
        self._add_static(
            "find_neighbors", ComputeNeighborsNode(), deps=["detect_levels"]
        )

        self._frame_dag.build()
        return self

    def execute(
        self, shared_data: dict[str, Any], *, progress: _RichProgressSink | None = None
    ) -> dict[str, Any]:
        """Execute the full hierarchy workflow and mutate shared_data.

        This method ensures required shared components exist, runs the static stage
        once, then iterates through trajectory frames to run the per-frame stage and
        reduce outputs into running means.

        Args:
            shared_data: Shared workflow data dict. This mapping is mutated in-place
                by both static and frame stages.
            progress: Optional progress sink passed through to nodes and used for
                per-frame progress reporting when supported.

        Returns:
            The same shared_data mapping passed in, after mutation.
        """
        shared_data.setdefault("axes_manager", AxesCalculator())
        self._run_static_stage(shared_data, progress=progress)
        self._run_frame_stage(shared_data, progress=progress)
        return shared_data

    def _run_static_stage(
        self, shared_data: dict[str, Any], *, progress: _RichProgressSink | None = None
    ) -> None:
        """Run all static nodes in dependency order.

        Nodes are executed in topological order of the static DAG. If a progress
        object is provided, it is passed to node.run when the node accepts it.

        Args:
            shared_data: Shared workflow data dict to be mutated by static nodes.
            progress: Optional progress sink to pass to nodes that support it.
        """
        for node_name in nx.topological_sort(self._static_graph):
            node = self._static_nodes[node_name]
            if progress is not None:
                try:
                    node.run(shared_data, progress=progress)
                    continue
                except TypeError:
                    pass
            node.run(shared_data)

    def _add_static(self, name: str, node: Any, deps: list[str] | None = None) -> None:
        """Register a static node and its dependencies in the static DAG.

        Args:
            name: Unique node name used in the static DAG.
            node: Node object exposing a run(shared_data, **kwargs) method.
            deps: Optional list of upstream node names that must run before this node.

        Returns:
            None. Mutates the internal static graph and node registry.
        """
        self._static_nodes[name] = node
        self._static_graph.add_node(name)
        for dep in deps or []:
            self._static_graph.add_edge(dep, name)

    def _run_frame_stage(
        self, shared_data: dict[str, Any], *, progress: _RichProgressSink | None = None
    ) -> None:
        """Execute the per-frame DAG stage and reduce frame outputs.

        This method iterates over the selected trajectory frames, executes the
        frame-local DAG for each frame, and reduces the resulting outputs into the
        shared accumulators stored in `shared_data`.

        Progress reporting is optional. If a progress sink is provided, a task is
        always created. When the total number of frames cannot be determined, the
        task is created with total=None (indeterminate).

        Args:
            shared_data: Shared data dictionary. Must contain:
                - "reduced_universe": MDAnalysis Universe providing the trajectory.
                - "start", "end", "step": frame slicing parameters.
                - any additional keys required by the frame DAG and reducer.
            progress: Optional progress sink (e.g., from ResultsReporter.progress()).
                Must expose add_task(), update(), and advance().

        Returns:
            None. Mutates `shared_data` in-place via reduction.

        Notes:
            The task title shows the current frame index being processed.
        """
        n_frames = shared_data["n_frames"]

        task: TaskID | None = None
        total_frames: int | None = None

        if progress is not None:
            try:
                total_frames = n_frames
            except Exception:
                total_frames = None

            task = progress.add_task(
                "[green]Frame processing",
                total=total_frames,
                title="Initializing",
            )

        for frame_index in range(n_frames):
            if progress is not None and task is not None:
                progress.update(task, title=f"Frame {frame_index}")

            frame_out = self._frame_dag.execute_frame(shared_data, frame_index)
            self._reduce_one_frame(shared_data, frame_out)

            if progress is not None and task is not None:
                progress.advance(task)

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
        self, shared_data: dict[str, Any], frame_out: dict[str, Any]
    ) -> None:
        """Reduce one frame's covariance outputs into shared running means.

        Args:
            shared_data: Shared workflow data dict containing accumulators.
            frame_out: Frame-local covariance outputs produced by FrameGraph.
        """
        self._reduce_force_and_torque(shared_data, frame_out)
        self._reduce_forcetorque(shared_data, frame_out)

    def _reduce_force_and_torque(
        self, shared_data: dict[str, Any], frame_out: dict[str, Any]
    ) -> None:
        """Reduce force/torque covariance outputs into shared accumulators.

        Args:
            shared_data: Shared workflow data dict containing:
                - "force_covariances", "torque_covariances": accumulator structures.
                - "frame_counts": running sample counts for each accumulator slot.
                - "group_id_to_index": mapping from group id to accumulator index.
            frame_out: Frame-local outputs containing "force" and "torque" sections.

        Returns:
            None. Mutates accumulator values and counts in shared_data in-place.
        """
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
        self, shared_data: dict[str, Any], frame_out: dict[str, Any]
    ) -> None:
        """Reduce combined force-torque covariance outputs into shared accumulators.

        Args:
            shared_data: Shared workflow data dict containing:
                - "forcetorque_covariances": accumulator structures.
                - "forcetorque_counts": running sample counts for each accumulator slot.
                - "group_id_to_index": mapping from group id to accumulator index.
            frame_out: Frame-local outputs that may include a "forcetorque" section.

        Returns:
            None. Mutates accumulator values and counts in shared_data in-place.
        """
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
