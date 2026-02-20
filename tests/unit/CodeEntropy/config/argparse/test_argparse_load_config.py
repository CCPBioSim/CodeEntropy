from unittest.mock import mock_open, patch

from CodeEntropy.config.argparse import ConfigResolver


def test_load_config_valid_yaml_returns_dict():
    yaml_content = """
run1:
  selection_string: protein
"""
    with (
        patch("glob.glob", return_value=["/fake/config.yaml"]),
        patch("builtins.open", mock_open(read_data=yaml_content)),
    ):
        resolver = ConfigResolver()
        config = resolver.load_config("/fake")

    assert "run1" in config
    assert config["run1"]["selection_string"] == "protein"


def test_load_config_no_yaml_files_returns_default():
    with patch("glob.glob", return_value=[]):
        resolver = ConfigResolver()
        config = resolver.load_config("/fake")

    assert config == {"run1": {}}


def test_load_config_yaml_empty_returns_default_run1():
    yaml_content = ""  # yaml.safe_load -> None
    with (
        patch("glob.glob", return_value=["/fake/config.yaml"]),
        patch("builtins.open", mock_open(read_data=yaml_content)),
    ):
        resolver = ConfigResolver()
        config = resolver.load_config("/fake")

    assert config == {"run1": {}}


def test_load_config_open_error_returns_default():
    with (
        patch("glob.glob", return_value=["/fake/config.yaml"]),
        patch("builtins.open", side_effect=OSError("boom")),
    ):
        resolver = ConfigResolver()
        config = resolver.load_config("/fake")

    assert config == {"run1": {}}
