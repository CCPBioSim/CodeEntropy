import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from CodeEntropy.entropy.workflow import EntropyWorkflow


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
        inst._calculate_water_entropy = MagicMock()

        wf._compute_water_entropy(traj, water_groups)

    inst._calculate_water_entropy.assert_called_once()
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
    nonwater, water = wf._split_water_groups(groups)

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
        inst._calculate_water_entropy = MagicMock()

        wf._compute_water_entropy(traj, water_groups)

    WaterCls.assert_called_once_with(args)
    inst._calculate_water_entropy.assert_called_once()
    assert wf._args.selection_string == "not water"
