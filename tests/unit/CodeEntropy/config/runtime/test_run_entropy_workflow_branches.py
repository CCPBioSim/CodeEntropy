from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import CodeEntropy.config.runtime as runtime_mod


def test_run_entropy_workflow_warns_and_skips_non_dict_run_config(runner):
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

    runner.run_entropy_workflow()

    run_logger.warning.assert_called_once()
    runner._config_manager.resolve.assert_not_called()


def test_run_entropy_workflow_raises_when_required_args_missing(runner):
    runner._logging_config = MagicMock()
    runner._config_manager = MagicMock()
    runner._reporter = MagicMock()

    runner._logging_config.configure.return_value = MagicMock()
    runner.show_splash = MagicMock()

    runner._config_manager.load_config.return_value = {"run1": {}}

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
    runner._config_manager.resolve.return_value = args

    with pytest.raises(RuntimeError) as exc:
        runner.run_entropy_workflow()

    assert str(exc.value) == "CodeEntropyRunner encountered an error"
    assert isinstance(exc.value.__cause__, ValueError)
    assert "Missing 'top_traj_file' argument." in str(exc.value.__cause__)


def test_run_entropy_workflow_happy_path_calls_execute_once(runner):
    runner._logging_config = MagicMock()
    runner._config_manager = MagicMock()
    runner._reporter = MagicMock()
    runner.show_splash = MagicMock()
    runner.print_args_table = MagicMock()

    run_logger = MagicMock()
    runner._logging_config.configure.return_value = run_logger

    runner._config_manager.load_config.return_value = {"run1": {}}

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

    runner._config_manager.resolve.return_value = args

    runner._build_universe = MagicMock(return_value="U")
    runner._config_manager.validate_inputs = MagicMock()

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


def test_run_entropy_workflow_logs_when_args_cannot_be_serialized(runner, monkeypatch):
    runner._logging_config = MagicMock()
    runner._config_manager = MagicMock()
    runner._reporter = MagicMock()

    runner._logging_config.configure.return_value = MagicMock()
    runner.show_splash = MagicMock()
    runner._config_manager.load_config.return_value = {"run1": {}}

    class BadArgs:
        __slots__ = (
            "output_file",
            "verbose",
            "top_traj_file",
            "selection_string",
            "force_file",
            "file_format",
            "kcal_force_units",
        )

    args = BadArgs()
    args.output_file = "out.json"
    args.verbose = False
    args.top_traj_file = None
    args.selection_string = None
    args.force_file = None
    args.file_format = None
    args.kcal_force_units = False

    parser = MagicMock()
    parser.parse_known_args.return_value = (args, [])
    runner._config_manager.build_parser.return_value = parser
    runner._config_manager.resolve.return_value = args

    error_spy = MagicMock()
    monkeypatch.setattr(runtime_mod.logger, "error", error_spy)

    with pytest.raises(RuntimeError) as exc:
        runner.run_entropy_workflow()

    assert str(exc.value) == "CodeEntropyRunner encountered an error"
    assert isinstance(exc.value.__cause__, ValueError)

    assert any(
        "Run arguments at failure could not be serialized" in str(call.args[0])
        for call in error_spy.call_args_list
    )
