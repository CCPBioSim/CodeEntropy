from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.entropy.workflow import EntropyWorkflow


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

    groups, water = wf._split_water_groups({0: [1, 2]})

    assert water == {}
