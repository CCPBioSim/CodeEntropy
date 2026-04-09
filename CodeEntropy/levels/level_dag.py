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
   - Reduce frame-local outputs into deterministic sums and counts.
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
      - Reducing frame-local outputs into deterministic sums and counts.

    The reduction performed here is order-independent: frame-local sums and
    counts are accumulated across frames and final means are computed once after
    all frames have been processed.
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
        reduce outputs into deterministic sums and counts.

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
        self._finalize_means(shared_data)
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
        u = shared_data["reduced_universe"]
        start, end, step = shared_data["start"], shared_data["end"], shared_data["step"]

        task: TaskID | None = None
        total_frames: int | None = None

        if progress is not None:
            try:
                n_frames = len(u.trajectory)

                s = 0 if start is None else int(start)
                e = n_frames if end is None else int(end)

                if e < 0:
                    e = n_frames + e

                e = max(0, min(e, n_frames))
                s = max(0, min(s, e))

                st = 1 if step is None else int(step)
                if st > 0:
                    total_frames = max(0, (e - s + st - 1) // st)
            except Exception:
                total_frames = None

            task = progress.add_task(
                "[green]Frame processing",
                total=total_frames,
                title="Initializing",
            )

        for ts in u.trajectory[start:end:step]:
            if progress is not None and task is not None:
                progress.update(task, title=f"Frame {ts.frame}")

            frame_out = self._frame_dag.execute_frame(shared_data, ts.frame)
            self._reduce_one_frame(shared_data, frame_out)

            if progress is not None and task is not None:
                progress.advance(task)

    def _reduce_one_frame(
        self, shared_data: dict[str, Any], frame_out: dict[str, Any]
    ) -> None:
        """Reduce one frame's covariance outputs into shared sum accumulators.

        Args:
            shared_data: Shared workflow data dict containing accumulators.
            frame_out: Frame-local covariance outputs produced by FrameGraph.
        """
        self._reduce_force_and_torque(shared_data, frame_out)
        self._reduce_forcetorque(shared_data, frame_out)

    def _reduce_force_and_torque(
        self, shared_data: dict[str, Any], frame_out: dict[str, Any]
    ) -> None:
        """Reduce force/torque frame-local sums into shared accumulators.

        Args:
            shared_data: Shared workflow data dict containing:
                - "force_sums", "torque_sums": running sum accumulators.
                - "force_counts", "torque_counts": running sample counts.
                - "group_id_to_index": mapping from group id to accumulator index.
            frame_out: Frame-local outputs containing "force", "torque",
                "force_counts", and "torque_counts" sections.

        Returns:
            None. Mutates shared accumulators and counts in-place.
        """
        f_sums = shared_data["force_sums"]
        t_sums = shared_data["torque_sums"]
        f_counts = shared_data["force_counts"]
        t_counts = shared_data["torque_counts"]
        gid2i = shared_data["group_id_to_index"]

        f_frame = frame_out["force"]
        t_frame = frame_out["torque"]
        f_frame_counts = frame_out["force_counts"]
        t_frame_counts = frame_out["torque_counts"]

        for key in sorted(f_frame["ua"].keys()):
            F = f_frame["ua"][key]
            c = int(f_frame_counts["ua"].get(key, 0))
            if c <= 0:
                continue
            prev = f_sums["ua"].get(key)
            f_sums["ua"][key] = F.copy() if prev is None else prev + F
            f_counts["ua"][key] = f_counts["ua"].get(key, 0) + c

        for key in sorted(t_frame["ua"].keys()):
            T = t_frame["ua"][key]
            c = int(t_frame_counts["ua"].get(key, 0))
            if c <= 0:
                continue
            prev = t_sums["ua"].get(key)
            t_sums["ua"][key] = T.copy() if prev is None else prev + T
            t_counts["ua"][key] = t_counts["ua"].get(key, 0) + c

        for gid in sorted(f_frame["res"].keys()):
            F = f_frame["res"][gid]
            gi = gid2i[gid]
            c = int(f_frame_counts["res"].get(gid, 0))
            if c <= 0:
                continue
            prev = f_sums["res"][gi]
            f_sums["res"][gi] = F.copy() if prev is None else prev + F
            f_counts["res"][gi] += c

        for gid in sorted(t_frame["res"].keys()):
            T = t_frame["res"][gid]
            gi = gid2i[gid]
            c = int(t_frame_counts["res"].get(gid, 0))
            if c <= 0:
                continue
            prev = t_sums["res"][gi]
            t_sums["res"][gi] = T.copy() if prev is None else prev + T
            t_counts["res"][gi] += c

        for gid in sorted(f_frame["poly"].keys()):
            F = f_frame["poly"][gid]
            gi = gid2i[gid]
            c = int(f_frame_counts["poly"].get(gid, 0))
            if c <= 0:
                continue
            prev = f_sums["poly"][gi]
            f_sums["poly"][gi] = F.copy() if prev is None else prev + F
            f_counts["poly"][gi] += c

        for gid in sorted(t_frame["poly"].keys()):
            T = t_frame["poly"][gid]
            gi = gid2i[gid]
            c = int(t_frame_counts["poly"].get(gid, 0))
            if c <= 0:
                continue
            prev = t_sums["poly"][gi]
            t_sums["poly"][gi] = T.copy() if prev is None else prev + T
            t_counts["poly"][gi] += c

    def _reduce_forcetorque(
        self, shared_data: dict[str, Any], frame_out: dict[str, Any]
    ) -> None:
        """Reduce combined force-torque frame-local sums into shared accumulators.

        Args:
            shared_data: Shared workflow data dict containing:
                - "forcetorque_sums": running sum accumulators.
                - "forcetorque_counts": running sample counts.
                - "group_id_to_index": mapping from group id to accumulator index.
            frame_out: Frame-local outputs that may include "forcetorque" and
                "forcetorque_counts" sections.

        Returns:
            None. Mutates shared accumulators and counts in-place.
        """
        if "forcetorque" not in frame_out:
            return

        ft_sums = shared_data["forcetorque_sums"]
        ft_counts = shared_data["forcetorque_counts"]
        gid2i = shared_data["group_id_to_index"]

        ft_frame = frame_out["forcetorque"]
        ft_frame_counts = frame_out.get("forcetorque_counts", {"res": {}, "poly": {}})

        for gid in sorted(ft_frame.get("res", {}).keys()):
            M = ft_frame["res"][gid]
            gi = gid2i[gid]
            c = int(ft_frame_counts.get("res", {}).get(gid, 0))
            if c <= 0:
                continue
            prev = ft_sums["res"][gi]
            ft_sums["res"][gi] = M.copy() if prev is None else prev + M
            ft_counts["res"][gi] += c

        for gid in sorted(ft_frame.get("poly", {}).keys()):
            M = ft_frame["poly"][gid]
            gi = gid2i[gid]
            c = int(ft_frame_counts.get("poly", {}).get(gid, 0))
            if c <= 0:
                continue
            prev = ft_sums["poly"][gi]
            ft_sums["poly"][gi] = M.copy() if prev is None else prev + M
            ft_counts["poly"][gi] += c

    def _finalize_means(self, shared_data: dict[str, Any]) -> None:
        """Compute finalized mean matrices from accumulated sums and counts.

        Args:
            shared_data: Shared workflow data dict containing running sums and counts.

        Returns:
            None. Writes finalized mean matrices back into shared_data.
        """

        def _compute_means(
            sums: dict[str, Any],
            counts: dict[str, Any],
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}

            for domain in sorted(sums.keys()):
                domain_sums = sums[domain]
                domain_counts = counts[domain]

                if isinstance(domain_sums, dict):
                    out[domain] = {}
                    for key in sorted(domain_sums.keys()):
                        total = domain_sums[key]
                        count = int(domain_counts.get(key, 0))
                        out[domain][key] = total / float(count) if count > 0 else None
                    continue

                mean_list: list[Any] = [None] * len(domain_sums)
                for idx, total in enumerate(domain_sums):
                    if total is None:
                        continue
                    count = int(domain_counts[idx])
                    mean_list[idx] = total / float(count) if count > 0 else None
                out[domain] = mean_list

            return out

        shared_data["force_covariances"] = _compute_means(
            shared_data["force_sums"],
            shared_data["force_counts"],
        )
        shared_data["torque_covariances"] = _compute_means(
            shared_data["torque_sums"],
            shared_data["torque_counts"],
        )
        shared_data["forcetorque_covariances"] = _compute_means(
            shared_data["forcetorque_sums"],
            shared_data["forcetorque_counts"],
        )

        shared_data["frame_counts"] = shared_data["force_counts"]
        shared_data["force_torque_stats"] = {
            "res": list(shared_data["forcetorque_covariances"]["res"]),
            "poly": list(shared_data["forcetorque_covariances"]["poly"]),
        }
        shared_data["force_torque_counts"] = {
            "res": shared_data["forcetorque_counts"]["res"].copy(),
            "poly": shared_data["forcetorque_counts"]["poly"].copy(),
        }
