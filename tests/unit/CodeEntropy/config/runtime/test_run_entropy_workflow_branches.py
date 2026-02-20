from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_run_entropy_workflow_warns_and_skips_non_dict_run_config(runner):
    # Arrange: mock all collaborators so the method stays unit-level
    runner._logging_config = MagicMock()
    runner._config_manager = MagicMock()
    runner._reporter = MagicMock()

    run_logger = MagicMock()
    runner._logging_config.configure.return_value = run_logger

    runner._config_manager.load_config.return_value = {"bad_run": "not_a_dict"}

    parser = MagicMock()
    parser.parse_known_args.return_value = (SimpleNamespace(output_file="out.json"), [])
    runner._config_manager.build_parser.return_value = parser

    runner.show_splash = MagicMock()

    # Act
    runner.run_entropy_workflow()

    # Assert: one behavior only — warning + skip
    run_logger.warning.assert_called_once()
    runner._config_manager.resolve.assert_not_called()


def test_run_entropy_workflow_raises_when_required_args_missing(runner):
    runner._logging_config = MagicMock()
    runner._config_manager = MagicMock()
    runner._reporter = MagicMock()

    runner._logging_config.configure.return_value = MagicMock()
    runner.show_splash = MagicMock()

    # config contains a valid dict run so it will try to process it
    runner._config_manager.load_config.return_value = {"run1": {}}

    # parser returns args with missing top_traj_file/selection_string
    parser = MagicMock()
    args = SimpleNamespace(
        output_file="out.json",
        verbose=False,
        top_traj_file=None,
        selection_string=None,
        force_file=None,
        file_format=None,
        kcal_force_units=False,
    )
    parser.parse_known_args.return_value = (args, [])
    runner._config_manager.build_parser.return_value = parser

    # resolve returns same args (still missing required)
    runner._config_manager.resolve.return_value = args

    with pytest.raises(ValueError):
        runner.run_entropy_workflow()


def test_run_entropy_workflow_happy_path_calls_execute_once(runner):
    # Mock collaborators
    runner._logging_config = MagicMock()
    runner._config_manager = MagicMock()
    runner._reporter = MagicMock()
    runner.show_splash = MagicMock()
    runner.print_args_table = MagicMock()

    run_logger = MagicMock()
    runner._logging_config.configure.return_value = run_logger

    # One valid run config dict (so it doesn't hit the "warning+continue" branch)
    runner._config_manager.load_config.return_value = {"run1": {}}

    # CLI args (must satisfy required args)
    args = SimpleNamespace(
        output_file="out.json",
        verbose=False,
        top_traj_file=["top.tpr", "traj.trr"],
        selection_string="all",
        force_file=None,
        file_format=None,
        kcal_force_units=False,
    )
    parser = MagicMock()
    parser.parse_known_args.return_value = (args, [])
    runner._config_manager.build_parser.return_value = parser

    # resolve returns the args
    runner._config_manager.resolve.return_value = args

    # Avoid MDAnalysis/real work: stub universe creation + validation
    runner._build_universe = MagicMock(return_value="U")
    runner._config_manager.validate_inputs = MagicMock()

    # Patch constructors used in the happy path
    with (
        patch("CodeEntropy.config.runtime.UniverseOperations") as _,
        patch("CodeEntropy.config.runtime.MoleculeGrouper") as _,
        patch("CodeEntropy.config.runtime.ConformationStateBuilder") as _,
        patch("CodeEntropy.config.runtime.EntropyWorkflow") as EWCls,
    ):
        entropy_instance = MagicMock()
        EWCls.return_value = entropy_instance

        runner.run_entropy_workflow()

    runner.print_args_table.assert_called_once_with(args)
    runner._build_universe.assert_called_once()
    runner._config_manager.validate_inputs.assert_called_once_with("U", args)
    EWCls.assert_called_once()
    entropy_instance.execute.assert_called_once()
    runner._logging_config.export_console.assert_called_once()
