"""Hierarchy-level DAG orchestration.

LevelDAG owns hierarchy-level workflow order. Static setup nodes prepare
structural and conformational data, then frame-local covariance and neighbour
observables are executed through deterministic frame map-reduce.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from CodeEntropy.levels.axes import AxesCalculator
from CodeEntropy.levels.execution.policy import ExecutionPolicy
from CodeEntropy.levels.execution.reducers import NeighborReducer
from CodeEntropy.levels.execution.scheduler import FrameScheduler
from CodeEntropy.levels.frame_dag import FrameGraph
from CodeEntropy.levels.neighbors import Neighbors
from CodeEntropy.levels.nodes.accumulators import InitCovarianceAccumulatorsNode
from CodeEntropy.levels.nodes.beads import BuildBeadsNode
from CodeEntropy.levels.nodes.conformations import ComputeConformationalStatesNode
from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode
from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode
from CodeEntropy.results.reporter import _RichProgressSink


class LevelDAG:
    """Execute static setup and deterministic frame map-reduce execution."""

    def __init__(self, universe_operations: Any | None = None) -> None:
        """Initialise the hierarchy-level DAG.

        Args:
            universe_operations: Optional universe-operation adapter passed to static
                conformational-state setup and frame-local execution.
        """
        self._universe_operations = universe_operations
        self._static_graph = nx.DiGraph()
        self._static_nodes: dict[str, Any] = {}
        self._frame_dag = FrameGraph(universe_operations=universe_operations)
        self._policy = ExecutionPolicy()

    def build(self) -> LevelDAG:
        """Build static and frame-level DAG topology.

        Returns:
            The current ``LevelDAG`` instance for fluent construction.
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

    def execute(
        self,
        shared_data: dict[str, Any],
        *,
        progress: _RichProgressSink | None = None,
    ) -> dict[str, Any]:
        """Execute the hierarchy workflow.

        Args:
            shared_data: Shared workflow data mutated by static setup, frame execution,
                and parent-side reductions.
            progress: Optional progress sink passed to supported static nodes and frame
                scheduling.

        Returns:
            The same ``shared_data`` mapping after workflow execution.

        Raises:
            KeyError: If required shared workflow keys are missing.
        """
        shared_data.setdefault("axes_manager", AxesCalculator())

        self._run_static_stage(shared_data, progress=progress)
        self._initialise_neighbor_metadata(shared_data)
        NeighborReducer.initialise(shared_data)
        self._run_frame_stage(shared_data, progress=progress)
        NeighborReducer.finalise(shared_data)

        return shared_data

    def _run_static_stage(
        self,
        shared_data: dict[str, Any],
        *,
        progress: _RichProgressSink | None = None,
    ) -> None:
        """Run static setup nodes in dependency order.

        Args:
            shared_data: Shared workflow data mutated by each static node.
            progress: Optional progress sink passed to nodes that accept it.
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

    def _add_static(
        self,
        name: str,
        node: Any,
        deps: list[str] | None = None,
    ) -> None:
        """Register a static node in the hierarchy DAG.

        Args:
            name: Unique node name in the static DAG.
            node: Node object exposing a ``run`` method.
            deps: Optional upstream node names that must execute before ``name``.
        """
        self._static_nodes[name] = node
        self._static_graph.add_node(name)

        for dep in deps or []:
            self._static_graph.add_edge(dep, name)

    def _run_frame_stage(
        self,
        shared_data: dict[str, Any],
        *,
        progress: _RichProgressSink | None = None,
    ) -> None:
        """Execute frame map-reduce work through the frame scheduler.

        Args:
            shared_data: Shared workflow data containing ``frame_source`` and
                frame-stage inputs. The method writes ``n_frames``.
            progress: Optional progress sink forwarded to the frame scheduler.

        Raises:
            KeyError: If ``frame_source`` is missing from ``shared_data``.
        """
        frame_source = shared_data["frame_source"]
        frame_indices = [
            int(frame_index) for frame_index in frame_source.iter_indices()
        ]
        shared_data["n_frames"] = len(frame_indices)

        scheduler = FrameScheduler(
            frame_dag=self._frame_dag,
            policy=self._policy,
            universe_operations=self._universe_operations,
        )
        scheduler.execute(
            shared_data,
            frame_indices=frame_indices,
            progress=progress,
        )

    @staticmethod
    def _initialise_neighbor_metadata(shared_data: dict[str, Any]) -> None:
        """Compute frame-invariant neighbour metadata.

        Args:
            shared_data: Shared workflow data containing ``groups`` and either
                ``reduced_universe`` or ``universe``. The method writes
                ``symmetry_number`` and ``linear``.

        Raises:
            KeyError: If ``groups`` is missing from ``shared_data``.
        """
        helper = Neighbors()
        universe = shared_data.get("reduced_universe", shared_data.get("universe"))

        symmetry_number, linear = helper.get_symmetry(
            universe=universe,
            groups=shared_data["groups"],
        )

        shared_data["symmetry_number"] = symmetry_number
        shared_data["linear"] = linear
