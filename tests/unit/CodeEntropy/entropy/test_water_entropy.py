from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.entropy.water import WaterEntropy


def _make_fake_universe_with_water():
    universe = MagicMock()

    water_selection = MagicMock()

    water_residues = MagicMock()
    water_residues.resnames = ["WAT"]
    water_residues.__len__.return_value = 1

    water_selection.residues = water_residues
    water_selection.atoms = [1, 2, 3]

    universe.select_atoms.return_value = water_selection
    return universe


def test_water_entropy_calls_solver_and_logs_components():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    Sorient_dict = {1: {"WAT": [3.0, 5]}}

    covariances = SimpleNamespace(counts={(7, "WAT"): 2})

    vibrations = SimpleNamespace(
        translational_S={(7, "WAT"): [1.0, 2.0]},
        rotational_S={(7, "WAT"): 4.0},
    )

    solver = MagicMock(return_value=(Sorient_dict, covariances, vibrations, None, 123))

    we = WaterEntropy(args=args, reporter=reporter, solver=solver)

    we._solute_id_to_resname = MagicMock(return_value="SOL")

    universe = _make_fake_universe_with_water()

    we.calculate_and_log(universe=universe, start=0, end=10, step=1, group_id=9)

    solver.assert_called_once()

    reporter.add_residue_data.assert_any_call(
        9, "WAT", "Water", "Orientational", 5, 3.0
    )

    reporter.add_residue_data.assert_any_call(
        9, "SOL", "Water", "Transvibrational", 2, 3.0
    )

    reporter.add_residue_data.assert_any_call(
        9, "SOL", "Water", "Rovibrational", 2, 4.0
    )

    reporter.add_group_label.assert_called_once()


def test_water_entropy_handles_empty_solver_results_gracefully():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    solver = MagicMock(
        return_value=(
            {},
            SimpleNamespace(counts={}),
            SimpleNamespace(translational_S={}, rotational_S={}),
            None,
            0,
        )
    )

    we = WaterEntropy(args=args, reporter=reporter, solver=solver)
    we._solute_id_to_resname = MagicMock(return_value="SOL")

    universe = _make_fake_universe_with_water()

    we.calculate_and_log(universe=universe, start=0, end=10, step=1, group_id=1)

    reporter.add_residue_data.assert_not_called()
    reporter.add_group_label.assert_called_once()


def test_water_group_label_handles_multiple_water_resnames():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    solver = MagicMock(
        return_value=(
            {},
            SimpleNamespace(counts={}),
            SimpleNamespace(translational_S={}, rotational_S={}),
            None,
            0,
        )
    )
    we = WaterEntropy(args=args, reporter=reporter, solver=solver)
    we._solute_id_to_resname = MagicMock(return_value="SOL")

    universe = MagicMock()
    water_selection = MagicMock()

    water_residues = MagicMock()
    water_residues.resnames = ["WAT", "TIP3"]
    water_residues.__len__.return_value = 2

    water_selection.residues = water_residues
    water_selection.atoms = [1, 2, 3, 4]

    universe.select_atoms.return_value = water_selection

    we.calculate_and_log(universe=universe, start=0, end=10, step=1, group_id=1)

    reporter.add_group_label.assert_called_once()


def test_log_group_label_defaults_to_WAT_when_no_residue_names_match():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()

    solver = MagicMock(
        return_value=(
            {},
            SimpleNamespace(counts={}),
            SimpleNamespace(translational_S={}, rotational_S={}),
            None,
            0,
        )
    )
    we = WaterEntropy(args=args, reporter=reporter, solver=solver)

    universe = MagicMock()

    water_selection = MagicMock()
    water_residues = MagicMock()
    water_residues.resnames = ["WAT"]
    water_residues.__len__.return_value = 1

    water_selection.residues = water_residues
    water_selection.atoms = [1, 2, 3]
    universe.select_atoms.return_value = water_selection

    Sorient_dict = {1: {"TIP3": [1.0, 1]}}

    we._log_group_label(universe, Sorient_dict, group_id=7)

    reporter.add_group_label.assert_called_once()
    _, residue_group, *_ = reporter.add_group_label.call_args.args
    assert residue_group == "WAT"


def test_log_group_label_defaults_to_WAT_when_no_names_match():
    args = SimpleNamespace(temperature=298.0)
    reporter = MagicMock()
    we = WaterEntropy(args=args, reporter=reporter, solver=MagicMock())

    universe = MagicMock()
    water_selection = MagicMock()

    residues = MagicMock()
    residues.resnames = ["WAT"]
    residues.__len__.return_value = 2

    water_selection.residues = residues
    water_selection.atoms = [1, 2, 3, 4]
    universe.select_atoms.return_value = water_selection

    Sorient_dict = {1: {"TIP3": [1.0, 2]}}

    we._log_group_label(universe, Sorient_dict, group_id=7)

    reporter.add_group_label.assert_called_once()

    assert reporter.add_group_label.call_args.args[1] == "WAT"


def test_solute_id_to_resname_strips_suffix_after_last_underscore():
    assert WaterEntropy._solute_id_to_resname("ALA_0") == "ALA"
    assert WaterEntropy._solute_id_to_resname("ALA_BLA_12") == "ALA_BLA"


def test_solute_id_to_resname_returns_string_when_no_underscore():
    assert WaterEntropy._solute_id_to_resname("WAT") == "WAT"
    assert WaterEntropy._solute_id_to_resname(123) == "123"
