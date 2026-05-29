import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from CodeEntropy.entropy.workflow import EntropyWorkflow
from CodeEntropy.trajectory.frames import FrameSelection


def _make_wf(args):
    return EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=MagicMock(),
        reporter=MagicMock(molecule_data=[], residue_data=[]),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )


def _make_frame_selection(
    start: int = 0,
    end: int = 5,
    step: int = 1,
) -> FrameSelection:
    """Build a FrameSelection for workflow unit tests."""
    return FrameSelection.from_bounds(start=start, stop=end, step=step)


def test_execute_calls_level_dag_and_entropy_graph_and_logs_tables():
    args = SimpleNamespace(
        start=0,
        end=-1,
        step=1,
        grouping="molecules",
        water_entropy=False,
        selection_string="all",
    )

    universe = MagicMock()
    universe.trajectory = list(range(5))

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=universe,
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    wf._build_reduced_universe = MagicMock(return_value=MagicMock())
    wf._detect_levels = MagicMock(return_value={0: ["united_atom"]})
    wf._split_water_groups = MagicMock(return_value=({0: [0]}, {}))
    wf._finalize_molecule_results = MagicMock()

    wf._group_molecules.grouping_molecules.return_value = {0: [0]}

    with (
        patch("CodeEntropy.entropy.workflow.LevelDAG") as LevelDAGCls,
        patch("CodeEntropy.entropy.workflow.EntropyGraph") as GraphCls,
    ):
        LevelDAGCls.return_value.build.return_value.execute.return_value = None
        GraphCls.return_value.build.return_value.execute.return_value = {"x": 1}

        wf.execute()

    wf._reporter.log_tables.assert_called_once()


def test_execute_water_entropy_branch_calls_water_entropy_solver():
    args = SimpleNamespace(
        start=0,
        end=-1,
        step=1,
        grouping="molecules",
        water_entropy=True,
        selection_string="all",
        output_file="out.json",
    )

    universe = MagicMock()
    universe.trajectory = list(range(5))

    reporter = MagicMock()
    reporter.molecule_data = []
    reporter.residue_data = []
    reporter.save_dataframes_as_json = MagicMock()

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=universe,
        reporter=reporter,
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    wf._build_reduced_universe = MagicMock(return_value=MagicMock())
    wf._detect_levels = MagicMock(return_value={0: ["united_atom"]})

    wf._split_water_groups = MagicMock(return_value=({0: [0]}, {9: [1, 2]}))
    wf._finalize_molecule_results = MagicMock()

    wf._group_molecules.grouping_molecules.return_value = {0: [0], 9: [1, 2]}

    with (
        patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls,
        patch("CodeEntropy.entropy.workflow.LevelDAG") as LevelDAGCls,
        patch("CodeEntropy.entropy.workflow.EntropyGraph") as GraphCls,
    ):
        water_instance = WaterCls.return_value
        water_instance.calculate_and_log = MagicMock()

        LevelDAGCls.return_value.build.return_value.execute.return_value = None
        GraphCls.return_value.build.return_value.execute.return_value = {}

        wf.execute()

    water_instance.calculate_and_log.assert_called_once()
    _, kwargs = water_instance.calculate_and_log.call_args
    assert kwargs["universe"] is universe
    assert kwargs["start"] == 0
    assert kwargs["end"] == 5
    assert kwargs["step"] == 1
    assert kwargs["group_id"] == 9


def test_get_trajectory_bounds_end_minus_one_uses_trajectory_length():
    args = SimpleNamespace(
        start=0,
        end=-1,
        step=2,
        grouping="molecules",
        water_entropy=False,
        selection_string="all",
    )
    universe = SimpleNamespace(trajectory=list(range(10)))

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=universe,
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    start, end, step = wf._get_trajectory_bounds()
    assert (start, end, step) == (0, 10, 2)


def test_frame_selection_matches_python_slice_math():
    """FrameSelection uses Python range semantics for selected frames."""
    selection_unit_step = _make_frame_selection(start=0, end=10, step=1)
    selection_stride_two = _make_frame_selection(start=0, end=10, step=2)

    assert selection_unit_step.n_frames == 10
    assert selection_unit_step.indices == tuple(range(0, 10, 1))

    assert selection_stride_two.n_frames == 5
    assert selection_stride_two.indices == tuple(range(0, 10, 2))


def test_finalize_results_called_even_if_empty():
    args = SimpleNamespace(output_file="out.json")
    reporter = MagicMock()
    reporter.molecule_data = []
    reporter.residue_data = []
    reporter.save_dataframes_as_json = MagicMock()

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=MagicMock(),
        reporter=reporter,
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    wf._finalize_molecule_results()

    reporter.save_dataframes_as_json.assert_called_once()


def test_split_water_groups_returns_empty_when_none():
    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=MagicMock(water_entropy=False),
        universe=MagicMock(),
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    groups, water = wf._split_water_groups(wf._universe, {0: [1, 2]})

    assert water == {}


def test_build_reduced_universe_non_all_selects_atoms_and_writes_universe():
    args = SimpleNamespace(
        selection_string="protein",
        grouping="molecules",
        start=0,
        end=-1,
        step=1,
        water_entropy=False,
        output_file="out.json",
    )
    universe = MagicMock()
    universe.trajectory = list(range(3))

    reduced = MagicMock()
    reduced.trajectory = list(range(3))

    uops = MagicMock()
    uops.select_atoms.return_value = reduced

    run_manager = MagicMock()
    reporter = MagicMock()

    wf = EntropyWorkflow(
        run_manager=run_manager,
        args=args,
        universe=universe,
        reporter=reporter,
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=uops,
    )

    frame_selection = _make_frame_selection(start=0, end=3, step=1)

    out = wf._build_reduced_universe(frame_selection)

    assert out is reduced
    uops.select_atoms.assert_called_once_with(universe, "protein")
    uops.select_frames.assert_not_called()
    run_manager.write_universe.assert_called_once_with(
        reduced,
        f"{len(reduced.trajectory)}_frame_dump_atom_selection",
    )


def test_build_reduced_universe_raises_when_frame_selection_empty():
    args = SimpleNamespace(
        selection_string="all",
        grouping="molecules",
        start=0,
        end=0,
        step=1,
        water_entropy=False,
        output_file="out.json",
    )

    universe = MagicMock()
    uops = MagicMock()
    run_manager = MagicMock()

    wf = EntropyWorkflow(
        run_manager=run_manager,
        args=args,
        universe=universe,
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=uops,
    )

    frame_selection = FrameSelection(indices=())

    with pytest.raises(ValueError, match="Frame selection is empty"):
        wf._build_reduced_universe(frame_selection)

    uops.select_atoms.assert_not_called()
    uops.select_frames.assert_not_called()
    run_manager.write_universe.assert_not_called()


def test_compute_water_entropy_updates_selection_string_and_calls_internal_method():
    args = SimpleNamespace(
        selection_string="all",
        water_entropy=True,
        temperature=298.0,
    )
    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=MagicMock(),
        reporter=MagicMock(molecule_data=[], residue_data=[]),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    frame_selection = _make_frame_selection(start=0, end=5, step=1)
    water_groups = {9: [1, 2]}

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        inst = WaterCls.return_value
        inst.calculate_and_log = MagicMock()

        wf._compute_water_entropy(frame_selection, water_groups)

    inst.calculate_and_log.assert_called_once_with(
        universe=wf._universe,
        start=0,
        end=5,
        step=1,
        group_id=9,
    )
    assert wf._args.selection_string == "not water"


def test_finalize_molecule_results_skips_invalid_entries_with_warning(caplog):
    args = SimpleNamespace(output_file="out.json")
    reporter = MagicMock()

    reporter.molecule_data = [(1, "united_atom", "Trans", "not-a-number")]
    reporter.residue_data = []
    reporter.save_dataframes_as_json = MagicMock()

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=MagicMock(),
        reporter=reporter,
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    caplog.set_level(logging.WARNING)
    wf._finalize_molecule_results()

    assert any("Skipping invalid entry" in r.message for r in caplog.records)
    reporter.save_dataframes_as_json.assert_called_once()


def test_build_reduced_universe_all_returns_original_universe():
    args = SimpleNamespace(
        selection_string="all",
        start=0,
        end=-1,
        step=1,
        grouping="molecules",
        water_entropy=False,
        output_file="out.json",
    )
    universe = MagicMock()
    universe.trajectory = list(range(2))

    uops = MagicMock()
    run_manager = MagicMock()

    wf = EntropyWorkflow(
        run_manager=run_manager,
        args=args,
        universe=universe,
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=uops,
    )

    frame_selection = _make_frame_selection(start=0, end=2, step=1)

    out = wf._build_reduced_universe(frame_selection)

    assert out is universe
    uops.select_atoms.assert_not_called()
    uops.select_frames.assert_not_called()
    run_manager.write_universe.assert_not_called()


def test_split_water_groups_partitions_correctly():
    args = SimpleNamespace(
        start=0,
        end=-1,
        step=1,
        grouping="molecules",
        water_entropy=False,
        selection_string="all",
        output_file="out.json",
    )
    universe = MagicMock()

    water_res = MagicMock()
    water_res.resid = 10
    water_atoms = MagicMock()
    water_atoms.residues = [water_res]
    universe.select_atoms.return_value = water_atoms

    frag0 = MagicMock()
    r0 = MagicMock()
    r0.resid = 10
    frag0.residues = [r0]

    frag1 = MagicMock()
    r1 = MagicMock()
    r1.resid = 99
    frag1.residues = [r1]

    universe.atoms.fragments = [frag0, frag1]

    wf = EntropyWorkflow(
        MagicMock(), args, universe, MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    groups = {0: [0], 1: [1]}
    nonwater, water = wf._split_water_groups(universe, groups)

    assert 0 in water
    assert 1 in nonwater


def test_compute_water_entropy_instantiates_waterentropy_and_updates_selection_string():
    args = SimpleNamespace(
        selection_string="all",
        water_entropy=True,
        temperature=298.0,
    )
    universe = MagicMock()
    reporter = MagicMock()

    wf = EntropyWorkflow(
        MagicMock(),
        args,
        universe,
        reporter,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    frame_selection = _make_frame_selection(start=0, end=5, step=1)
    water_groups = {9: [0]}

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        inst = WaterCls.return_value
        inst.calculate_and_log = MagicMock()

        wf._compute_water_entropy(frame_selection, water_groups)

    WaterCls.assert_called_once_with(args, reporter)
    inst.calculate_and_log.assert_called_once_with(
        universe=universe,
        start=0,
        end=5,
        step=1,
        group_id=9,
    )
    assert args.selection_string == "not water"


def test_compute_water_entropy_returns_when_no_water_groups():
    args = SimpleNamespace(
        selection_string="all",
        water_entropy=True,
        temperature=298.0,
        output_file="out.json",
    )
    wf = _make_wf(args)
    frame_selection = FrameSelection.from_bounds(start=0, stop=5, step=1)

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        wf._compute_water_entropy(frame_selection, water_groups={})

    WaterCls.assert_not_called()
    assert args.selection_string == "all"


def test_compute_water_entropy_returns_when_water_entropy_disabled():
    args = SimpleNamespace(
        selection_string="all",
        water_entropy=False,
        temperature=298.0,
        output_file="out.json",
    )
    wf = _make_wf(args)
    frame_selection = FrameSelection.from_bounds(start=0, stop=5, step=1)

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        wf._compute_water_entropy(frame_selection, water_groups={9: [0]})

    WaterCls.assert_not_called()
    assert args.selection_string == "all"


def test_compute_water_entropy_returns_when_frame_selection_empty():
    args = SimpleNamespace(
        selection_string="all",
        water_entropy=True,
        temperature=298.0,
        output_file="out.json",
    )
    wf = _make_wf(args)
    frame_selection = FrameSelection(indices=())

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        wf._compute_water_entropy(frame_selection, water_groups={9: [0]})

    WaterCls.assert_not_called()
    assert args.selection_string == "all"


def test_detect_levels_calls_hierarchy_builder():
    args = SimpleNamespace(
        selection_string="all", water_entropy=False, output_file="out.json"
    )
    wf = _make_wf(args)

    with patch("CodeEntropy.entropy.workflow.HierarchyBuilder") as HB:
        HB.return_value.select_levels.return_value = (123, {"levels": "ok"})

        out = wf._detect_levels(reduced_universe=MagicMock())

    assert out == {"levels": "ok"}
    HB.return_value.select_levels.assert_called_once()


def test_compute_water_entropy_returns_early_when_disabled_or_empty_groups():
    args = SimpleNamespace(
        selection_string="all",
        water_entropy=False,
        temperature=298.0,
        output_file="out.json",
    )
    wf = _make_wf(args)

    frame_selection = _make_frame_selection(start=0, end=5, step=1)

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        wf._compute_water_entropy(frame_selection, water_groups={})

    WaterCls.assert_not_called()
    assert args.selection_string == "all"


def test_finalize_molecule_results_skips_group_total_rows():
    args = SimpleNamespace(
        output_file="out.json", selection_string="all", water_entropy=False
    )
    wf = _make_wf(args)

    wf._reporter.molecule_data = [
        (1, "Group Total", "Group Total Entropy", 999.0),  # should be skipped
        (1, "united_atom", "Transvibrational", 1.5),  # should count
    ]
    wf._reporter.residue_data = []

    wf._finalize_molecule_results()

    # should append a new "Group Total" row based only on the non-total entries
    assert any(
        row[1] == "Group Total" and row[3] == 1.5 for row in wf._reporter.molecule_data
    )


def test_configure_parallel_frame_execution_returns_when_disabled():
    args = SimpleNamespace(
        parallel_frames=False,
        use_dask=False,
        output_file="out.json",
    )
    wf = _make_wf(args)
    shared_data = {}

    wf._configure_parallel_frame_execution(shared_data)

    assert shared_data == {}


def test_configure_parallel_frame_execution_reuses_existing_client():
    args = SimpleNamespace(
        parallel_frames=True,
        use_dask=False,
        output_file="out.json",
    )
    wf = _make_wf(args)

    client = MagicMock()
    shared_data = {"dask_client": client}

    wf._configure_parallel_frame_execution(shared_data)

    assert shared_data["dask_client"] is client
    assert shared_data["parallel_frames"] is True


def test_configure_parallel_frame_execution_creates_local_dask_client():
    args = SimpleNamespace(
        parallel_frames=True,
        use_dask=False,
        hpc=False,
        dask_workers=3,
        dask_threads_per_worker=1,
        output_file="out.json",
    )
    wf = _make_wf(args)

    fake_client_instance = MagicMock()
    fake_client_cls = MagicMock(return_value=fake_client_instance)

    fake_dask = types.ModuleType("dask")
    fake_distributed = types.ModuleType("dask.distributed")
    fake_distributed.Client = fake_client_cls

    shared_data = {}

    with patch.dict(
        sys.modules,
        {
            "dask": fake_dask,
            "dask.distributed": fake_distributed,
        },
    ):
        wf._configure_parallel_frame_execution(shared_data)

    fake_client_cls.assert_called_once_with(
        processes=True,
        n_workers=3,
        threads_per_worker=1,
    )
    assert shared_data["dask_client"] is fake_client_instance
    assert shared_data["parallel_frames"] is True


def test_configure_parallel_frame_execution_raises_when_dask_missing():
    args = SimpleNamespace(
        parallel_frames=True,
        use_dask=False,
        hpc=False,
        dask_workers=2,
        dask_threads_per_worker=1,
        output_file="out.json",
    )
    wf = _make_wf(args)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dask.distributed":
            raise ImportError("No module named dask.distributed")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(
            RuntimeError, match="Parallel frame execution was requested"
        ):
            wf._configure_parallel_frame_execution({})


def test_build_shared_data_contains_frame_source_and_frame_indices():
    args = SimpleNamespace(
        selection_string="all",
        water_entropy=False,
        output_file="out.json",
    )
    wf = _make_wf(args)

    reduced_universe = MagicMock()
    levels = {0: ["united_atom"]}
    groups = {2: [0, 1]}
    frame_selection = FrameSelection.from_bounds(start=2, stop=8, step=2)

    with patch("CodeEntropy.entropy.workflow.FrameSource") as FrameSourceCls:
        frame_source = MagicMock()
        FrameSourceCls.return_value = frame_source

        shared_data = wf._build_shared_data(
            reduced_universe=reduced_universe,
            levels=levels,
            groups=groups,
            frame_selection=frame_selection,
        )

    FrameSourceCls.assert_called_once_with(
        universe=reduced_universe,
        selection=frame_selection,
    )

    assert shared_data["entropy_manager"] is wf
    assert shared_data["run_manager"] is wf._run_manager
    assert shared_data["reporter"] is wf._reporter
    assert shared_data["args"] is args
    assert shared_data["universe"] is wf._universe
    assert shared_data["reduced_universe"] is reduced_universe
    assert shared_data["levels"] is levels
    assert shared_data["groups"] == groups
    assert shared_data["start"] == frame_selection.source_start
    assert shared_data["end"] == frame_selection.source_stop_exclusive
    assert shared_data["step"] == frame_selection.infer_source_step()
    assert shared_data["n_frames"] == frame_selection.n_frames
    assert shared_data["frame_selection"] is frame_selection
    assert shared_data["frame_source"] is frame_source
    assert shared_data["frame_indices"] == [2, 4, 6]
    assert shared_data["source_frame_indices"] == [2, 4, 6]


def test_run_level_dag_builds_and_executes_level_dag():
    args = SimpleNamespace(output_file="out.json")
    wf = _make_wf(args)
    shared_data = {"x": 1}
    progress = MagicMock()

    with patch("CodeEntropy.entropy.workflow.LevelDAG") as LevelDAGCls:
        level_dag = LevelDAGCls.return_value
        built_dag = level_dag.build.return_value

        wf._run_level_dag(shared_data, progress=progress)

    LevelDAGCls.assert_called_once_with(wf._universe_operations)
    level_dag.build.assert_called_once()
    built_dag.execute.assert_called_once_with(shared_data, progress=progress)


def test_run_entropy_graph_executes_and_updates_shared_data():
    args = SimpleNamespace(output_file="out.json")
    wf = _make_wf(args)
    shared_data = {"existing": "value"}
    progress = MagicMock()

    with patch("CodeEntropy.entropy.workflow.EntropyGraph") as GraphCls:
        graph = GraphCls.return_value
        built_graph = graph.build.return_value
        built_graph.execute.return_value = {"entropy_results": {"ok": True}}

        wf._run_entropy_graph(shared_data, progress=progress)

    GraphCls.assert_called_once()
    graph.build.assert_called_once()
    built_graph.execute.assert_called_once_with(shared_data, progress=progress)
    assert shared_data["existing"] == "value"
    assert shared_data["entropy_results"] == {"ok": True}


def test_get_trajectory_bounds_none_values_use_defaults():
    args = SimpleNamespace(
        start=None,
        end=None,
        step=None,
        output_file="out.json",
    )
    universe = SimpleNamespace(trajectory=list(range(7)))

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=universe,
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    assert wf._get_trajectory_bounds() == (0, 7, 1)


def test_compute_water_entropy_appends_not_water_to_existing_selection():
    args = SimpleNamespace(
        selection_string="protein",
        water_entropy=True,
        temperature=298.0,
        output_file="out.json",
    )
    wf = _make_wf(args)

    frame_selection = FrameSelection.from_bounds(start=0, stop=5, step=1)

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        inst = WaterCls.return_value
        inst.calculate_and_log = MagicMock()

        wf._compute_water_entropy(frame_selection, water_groups={9: [0]})

    inst.calculate_and_log.assert_called_once_with(
        universe=wf._universe,
        start=0,
        end=5,
        step=1,
        group_id=9,
    )
    assert args.selection_string == "protein and not water"


def test_execute_closes_dask_client_in_finally():
    args = SimpleNamespace(
        start=0,
        end=-1,
        step=1,
        grouping="molecules",
        water_entropy=False,
        selection_string="all",
        output_file="out.json",
    )

    universe = MagicMock()
    universe.trajectory = list(range(5))

    reporter = MagicMock()
    reporter.molecule_data = []
    reporter.residue_data = []

    progress_cm = MagicMock()
    progress_cm.__enter__.return_value = MagicMock()
    progress_cm.__exit__.return_value = False
    reporter.progress.return_value = progress_cm

    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=args,
        universe=universe,
        reporter=reporter,
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )

    client = MagicMock()

    wf._build_reduced_universe = MagicMock(return_value=MagicMock())
    wf._detect_levels = MagicMock(return_value={0: ["united_atom"]})
    wf._split_water_groups = MagicMock(return_value=({0: [0]}, {}))
    wf._build_shared_data = MagicMock(return_value={"dask_client": client})
    wf._configure_parallel_frame_execution = MagicMock()
    wf._run_level_dag = MagicMock()
    wf._run_entropy_graph = MagicMock()
    wf._finalize_molecule_results = MagicMock()
    wf._group_molecules.grouping_molecules.return_value = {0: [0]}

    wf.execute()

    client.close.assert_called_once()


def test_configure_parallel_frame_execution_uses_hpc_dask_manager():
    args = SimpleNamespace(
        parallel_frames=False,
        use_dask=False,
        hpc=True,
        dask_workers=None,
        dask_threads_per_worker=1,
        output_file="out.json",
    )
    wf = _make_wf(args)

    shared_data = {}
    client = MagicMock()

    with patch("CodeEntropy.entropy.workflow.HPCDaskManager") as HPCDaskManagerCls:
        HPCDaskManagerCls.return_value.configure_cluster.return_value = client

        wf._configure_parallel_frame_execution(shared_data)

    HPCDaskManagerCls.assert_called_once_with(args)
    HPCDaskManagerCls.return_value.configure_cluster.assert_called_once()

    assert shared_data["dask_client"] is client
    assert shared_data["parallel_frames"] is True
