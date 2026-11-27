import pytest

from CodeEntropy import __version__
from CodeEntropy.config.argparse import ConfigResolver


def test_build_parser_parses_selection_string():
    resolver = ConfigResolver()
    parser = resolver.build_parser()

    args = parser.parse_args(["--selection_string", "protein"])

    assert args.selection_string == "protein"


def test_build_parser_parses_bool_with_str2bool():
    resolver = ConfigResolver()
    parser = resolver.build_parser()

    args = parser.parse_args(["--kcal_force_units", "true"])

    assert args.kcal_force_units is True


def test_build_parser_store_true_flag_verbose_defaults_false_and_sets_true():
    resolver = ConfigResolver()
    parser = resolver.build_parser()

    args_default = parser.parse_args([])
    assert args_default.verbose is False

    args_verbose = parser.parse_args(["--verbose"])
    assert args_verbose.verbose is True


def test_build_parser_nargs_plus_parses_top_traj_file_list():
    resolver = ConfigResolver()
    parser = resolver.build_parser()

    args = parser.parse_args(["--top_traj_file", "a.tpr", "b.trr"])

    assert args.top_traj_file == ["a.tpr", "b.trr"]


def test_version_flag_prints_installed_version_and_exits(capsys):
    parser = ConfigResolver().build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])

    assert exc.value.code == 0
    assert f"CodeEntropy {__version__}" in capsys.readouterr().out


def test_build_parser_symmetry_number_defaults_to_none():
    parser = ConfigResolver().build_parser()

    args = parser.parse_args([])

    assert args.symmetry_number is None


def test_build_parser_parses_single_symmetry_number():
    parser = ConfigResolver().build_parser()

    args = parser.parse_args(["--symmetry_number", "12"])

    assert args.symmetry_number == ["12"]


def test_build_parser_parses_per_residue_symmetry_numbers():
    parser = ConfigResolver().build_parser()

    args = parser.parse_args(["--symmetry_number", "MET:12", "NH3:3"])

    assert args.symmetry_number == ["MET:12", "NH3:3"]


def test_build_parser_symmetry_number_requires_a_value():
    parser = ConfigResolver().build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--symmetry_number"])
