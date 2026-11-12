import logging
from unittest import mock

import pytest

from CodeEntropy.config.argparse import ConfigResolver, logger


def test_resolve_run_config_wrong_type_raises_type_error(resolver, empty_cli_args):
    with pytest.raises(TypeError):
        resolver.resolve(empty_cli_args, run_config="not-a-dict")


def test_resolve_none_run_config_treated_as_empty(resolver, empty_cli_args):
    resolved = resolver.resolve(empty_cli_args, None)
    # should still have defaults applied
    assert resolved.selection_string is not None


def test_resolve_yaml_applied_when_cli_not_provided(resolver, empty_cli_args):
    run_config = {"selection_string": "yaml_value"}

    resolved = resolver.resolve(empty_cli_args, run_config)

    assert resolved.selection_string == "yaml_value"


def test_resolve_cli_value_overrides_yaml(resolver):
    parser = resolver.build_parser()
    args = parser.parse_args(["--selection_string", "cli_value"])
    run_config = {"selection_string": "yaml_value"}

    resolved = resolver.resolve(args, run_config)

    assert resolved.selection_string == "cli_value"


def test_resolve_does_not_apply_yaml_key_not_in_arg_specs(resolver, empty_cli_args):
    run_config = {"not_a_real_arg": 123}

    resolved = resolver.resolve(empty_cli_args, run_config)

    assert not hasattr(resolved, "not_a_real_arg")


def test_resolve_ensure_defaults_sets_none_values(resolver):
    # If a known arg is None, _ensure_defaults should fill it.
    parser = resolver.build_parser()
    args = parser.parse_args([])

    # force a known arg to None to simulate partial/mutated namespace
    args.selection_string = None

    resolved = resolver.resolve(args, {})

    assert resolved.selection_string == "all"


def test_resolve_verbose_sets_logger_debug_level(resolver):
    parser = resolver.build_parser()
    args = parser.parse_args(["--verbose"])

    resolver.resolve(args, {})

    assert logger.level == logging.DEBUG


def test_apply_logging_level_updates_handler_level():
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)

    try:
        ConfigResolver._apply_logging_level(verbose=True)
        assert logger.level == logging.DEBUG
        assert handler.level == logging.DEBUG
    finally:
        logger.removeHandler(handler)


@mock.patch("CodeEntropy.config.argparse.HPCDaskManager")
def test_apply_hpc_conda_auto_detection_uses_manager_when_hpc_enabled(
    hpc_dask_manager,
    make_args,
):
    args = make_args(hpc=True, submit=False)

    manager_instance = mock.MagicMock()
    hpc_dask_manager.return_value = manager_instance

    ConfigResolver._apply_hpc_conda_auto_detection(args)

    hpc_dask_manager.assert_called_once_with(args)
    manager_instance.resolve_conda_settings.assert_called_once_with()


@mock.patch("CodeEntropy.config.argparse.HPCDaskManager")
def test_apply_hpc_conda_auto_detection_uses_manager_when_submit_enabled(
    hpc_dask_manager,
    make_args,
):
    args = make_args(hpc=False, submit=True)

    manager_instance = mock.MagicMock()
    hpc_dask_manager.return_value = manager_instance

    ConfigResolver._apply_hpc_conda_auto_detection(args)

    hpc_dask_manager.assert_called_once_with(args)
    manager_instance.resolve_conda_settings.assert_called_once_with()


@mock.patch("CodeEntropy.config.argparse.HPCDaskManager")
def test_apply_hpc_conda_auto_detection_skips_when_hpc_and_submit_disabled(
    hpc_dask_manager,
    make_args,
):
    args = make_args(hpc=False, submit=False)

    ConfigResolver._apply_hpc_conda_auto_detection(args)

    hpc_dask_manager.assert_not_called()
