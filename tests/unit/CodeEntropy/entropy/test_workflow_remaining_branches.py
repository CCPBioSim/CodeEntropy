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
