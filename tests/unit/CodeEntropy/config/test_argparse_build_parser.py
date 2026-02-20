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
