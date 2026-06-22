"""Unit tests for hierarchy-level DAG orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from CodeEntropy.levels.level_dag import LevelDAG


def test_build_registers_static_nodes_and_builds_stage_dags():
    with (
        patch("CodeEntropy.levels.level_dag.DetectMoleculesNode"),
        patch("CodeEntropy.levels.level_dag.DetectLevelsNode"),
        patch("CodeEntropy.levels.level_dag.BuildBeadsNode"),
        patch("CodeEntropy.levels.level_dag.InitCovarianceAccumulatorsNode"),
        patch("CodeEntropy.levels.level_dag.ConformationDAG"),
    ):
        universe_operations = MagicMock()
        dag = LevelDAG(universe_operations=universe_operations)
        dag._conformation_dag.build = MagicMock()
        dag._frame_dag.build = MagicMock()

        out = dag.build()

    assert out is dag
    assert set(dag._static_nodes) == {
        "detect_molecules",
        "detect_levels",
        "build_beads",
        "init_covariance_accumulators",
    }
    assert "find_neighbors" not in dag._static_nodes
    assert "compute_conformational_states" not in dag._static_nodes

    assert ("detect_molecules", "detect_levels") in dag._static_graph.edges
    assert ("detect_levels", "build_beads") in dag._static_graph.edges
    assert ("detect_levels", "init_covariance_accumulators") in dag._static_graph.edges
    assert (
        "detect_levels",
        "compute_conformational_states",
    ) not in dag._static_graph.edges

    dag._conformation_dag.build.assert_called_once()
    dag._frame_dag.build.assert_called_once()


def test_execute_sets_default_axes_manager_and_runs_workflow_stages():
    dag = LevelDAG()

    shared_data = {"groups": {0: [0]}}
    progress = MagicMock()

    dag._run_static_stage = MagicMock()
    dag._run_conformation_stage = MagicMock()
    dag._initialise_neighbor_metadata = MagicMock()
    dag._run_frame_stage = MagicMock()

    with (
        patch("CodeEntropy.levels.level_dag.NeighborReducer.initialise") as initialise,
        patch("CodeEntropy.levels.level_dag.NeighborReducer.finalise") as finalise,
    ):
        out = dag.execute(shared_data, progress=progress)

    assert out is shared_data
    assert "axes_manager" in shared_data

    dag._run_static_stage.assert_called_once_with(shared_data, progress=progress)
    dag._run_conformation_stage.assert_called_once_with(
        shared_data,
        progress=progress,
    )
    dag._initialise_neighbor_metadata.assert_called_once_with(shared_data)
    initialise.assert_called_once_with(shared_data)
    dag._run_frame_stage.assert_called_once_with(shared_data, progress=progress)
    finalise.assert_called_once_with(shared_data)


def test_add_static_adds_dependency_edges():
    dag = LevelDAG()

    node_a = MagicMock()
    node_b = MagicMock()

    dag._add_static("A", node_a)
    dag._add_static("B", node_b, deps=["A"])

    assert dag._static_nodes["A"] is node_a
    assert dag._static_nodes["B"] is node_b
    assert ("A", "B") in dag._static_graph.edges


def test_run_static_stage_calls_nodes_in_topological_order():
    dag = LevelDAG()
    node_a = MagicMock()
    node_b = MagicMock()

    dag._add_static("a", node_a)
    dag._add_static("b", node_b, deps=["a"])

    shared_data = {}

    dag._run_static_stage(shared_data)

    node_a.run.assert_called_once_with(shared_data)
    node_b.run.assert_called_once_with(shared_data)


def test_run_static_stage_forwards_progress_when_node_accepts_it():
    dag = LevelDAG()
    node = MagicMock()
    progress = MagicMock()
    shared_data = {}

    dag._add_static("node", node)

    dag._run_static_stage(shared_data, progress=progress)

    node.run.assert_called_once_with(shared_data, progress=progress)


def test_run_static_stage_falls_back_when_node_does_not_accept_progress():
    dag = LevelDAG()
    node = MagicMock()
    node.run.side_effect = [TypeError("unexpected keyword argument progress"), None]
    progress = MagicMock()
    shared_data = {}

    dag._add_static("node", node)

    dag._run_static_stage(shared_data, progress=progress)

    assert node.run.call_count == 2
    assert node.run.call_args_list == [
        call(shared_data, progress=progress),
        call(shared_data),
    ]


def test_run_conformation_stage_delegates_to_conformation_dag():
    dag = LevelDAG()
    shared_data = {}
    progress = MagicMock()

    dag._conformation_dag.execute = MagicMock()

    dag._run_conformation_stage(shared_data, progress=progress)

    dag._conformation_dag.execute.assert_called_once_with(
        shared_data,
        progress=progress,
    )


def test_run_frame_stage_collects_frame_indices_and_delegates_to_scheduler():
    universe_operations = MagicMock()
    dag = LevelDAG(universe_operations=universe_operations)
    progress = MagicMock()

    frame_source = MagicMock()
    frame_source.iter_indices.return_value = ["2", 4]

    shared_data = {"frame_source": frame_source}

    with patch("CodeEntropy.levels.level_dag.FrameScheduler") as Scheduler:
        scheduler = Scheduler.return_value

        dag._run_frame_stage(shared_data, progress=progress)

    assert shared_data["n_frames"] == 2
    frame_source.iter_indices.assert_called_once()

    Scheduler.assert_called_once_with(
        frame_dag=dag._frame_dag,
        policy=dag._policy,
        universe_operations=universe_operations,
    )
    scheduler.execute.assert_called_once_with(
        shared_data,
        frame_indices=[2, 4],
        progress=progress,
    )


def test_initialise_neighbor_metadata_writes_symmetry_and_linearity():
    universe = object()
    groups = {0: [0], 1: [1]}
    shared_data = {"reduced_universe": universe, "groups": groups}

    with patch("CodeEntropy.levels.level_dag.Neighbors") as Neighbors:
        helper = Neighbors.return_value
        helper.get_symmetry.return_value = ({0: 12, 1: 2}, {0: False, 1: True})

        LevelDAG._initialise_neighbor_metadata(shared_data)

    helper.get_symmetry.assert_called_once_with(universe=universe, groups=groups)
    assert shared_data["symmetry_number"] == {0: 12, 1: 2}
    assert shared_data["linear"] == {0: False, 1: True}


def test_initialise_neighbor_metadata_falls_back_to_universe_key():
    universe = object()
    shared_data = {"universe": universe, "groups": {0: [0]}}

    with patch("CodeEntropy.levels.level_dag.Neighbors") as Neighbors:
        helper = Neighbors.return_value
        helper.get_symmetry.return_value = ({0: 1}, {0: False})

        LevelDAG._initialise_neighbor_metadata(shared_data)

    helper.get_symmetry.assert_called_once_with(universe=universe, groups={0: [0]})


def test_level_dag_runs_static_conformation_then_frame(monkeypatch):
    dag = LevelDAG(universe_operations=object())
    calls = []

    monkeypatch.setattr(
        dag,
        "_run_static_stage",
        lambda shared_data, progress=None: calls.append("static"),
    )
    monkeypatch.setattr(
        dag,
        "_run_conformation_stage",
        lambda shared_data, progress=None: calls.append("conformation"),
    )
    monkeypatch.setattr(
        dag,
        "_initialise_neighbor_metadata",
        lambda shared_data: calls.append("neighbor_metadata"),
    )
    monkeypatch.setattr(
        dag,
        "_run_frame_stage",
        lambda shared_data, progress=None: calls.append("frame"),
    )

    monkeypatch.setattr(
        "CodeEntropy.levels.level_dag.NeighborReducer.initialise",
        lambda shared_data: calls.append("neighbor_initialise"),
    )
    monkeypatch.setattr(
        "CodeEntropy.levels.level_dag.NeighborReducer.finalise",
        lambda shared_data: calls.append("neighbor_finalise"),
    )

    dag.execute({})

    assert calls == [
        "static",
        "conformation",
        "neighbor_metadata",
        "neighbor_initialise",
        "frame",
        "neighbor_finalise",
    ]
