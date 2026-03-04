import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from CodeEntropy.entropy.workflow import EntropyWorkflow


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


def test_get_number_frames_matches_python_slice_math():
    wf = EntropyWorkflow(
        run_manager=MagicMock(),
        args=MagicMock(),
        universe=MagicMock(),
        reporter=MagicMock(),
        group_molecules=MagicMock(),
        dihedral_analysis=MagicMock(),
        universe_operations=MagicMock(),
    )
    assert wf._get_number_frames(0, 10, 1) == 10
    assert wf._get_number_frames(0, 10, 2) == 5


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


def test_build_reduced_universe_non_all_selects_and_writes_universe():
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
    reduced.trajectory = list(range(2))

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

    out = wf._build_reduced_universe()

    assert out is reduced
    uops.select_atoms.assert_called_once_with(universe, "protein")
    run_manager.write_universe.assert_called_once()


def test_compute_water_entropy_updates_selection_string_and_calls_internal_method():
    args = SimpleNamespace(
        selection_string="all", water_entropy=True, temperature=298.0
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

    traj = SimpleNamespace(start=0, end=5, step=1)
    water_groups = {9: [1, 2]}

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        inst = WaterCls.return_value
        inst.calculate_and_log = MagicMock()

        wf._compute_water_entropy(traj, water_groups)

    inst.calculate_and_log.assert_called_once_with(
        universe=wf._universe,
        start=traj.start,
        end=traj.end,
        step=traj.step,
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
    uops = MagicMock()
    run_manager = MagicMock()
    wf = EntropyWorkflow(
        run_manager, args, universe, MagicMock(), MagicMock(), MagicMock(), uops
    )

    out = wf._build_reduced_universe()

    assert out is universe
    uops.select_atoms.assert_not_called()
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
        selection_string="all", water_entropy=True, temperature=298.0
    )
    universe = MagicMock()
    reporter = MagicMock()
    wf = EntropyWorkflow(
        MagicMock(), args, universe, reporter, MagicMock(), MagicMock(), MagicMock()
    )

    traj = SimpleNamespace(start=0, end=5, step=1, n_frames=5)
    water_groups = {9: [0]}

    with patch("CodeEntropy.entropy.workflow.WaterEntropy") as WaterCls:
        inst = WaterCls.return_value
        inst.calculate_and_log = MagicMock()

        wf._compute_water_entropy(traj, water_groups)

    WaterCls.assert_called_once_with(args, reporter)
    inst.calculate_and_log.assert_called_once_with(
        universe=universe,
        start=traj.start,
        end=traj.end,
        step=traj.step,
        group_id=9,
    )
    assert args.selection_string == "not water"


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

    traj = SimpleNamespace(start=0, end=5, step=1, n_frames=5)

    # empty water groups OR water_entropy disabled -> early return
    wf._compute_water_entropy(traj, water_groups={})
    # no exception and no side effects expected


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
