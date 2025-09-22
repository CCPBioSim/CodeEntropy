import argparse
import logging
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import tests.data as data
from CodeEntropy.config.arg_config_manager import ConfigManager
from CodeEntropy.main import main
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestArgConfigManager(BaseTestCase):
    """
    Unit tests for the ConfigManager.
    """

    def setUp(self):
        super().setUp()

        self.test_data_dir = os.path.dirname(data.__file__)
        self.config_file = os.path.join(self.test_dir, "config.yaml")

        # Create a mock config file
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            self.setup_file(mock_file)
            with open(self.config_file, "w") as f:
                f.write(mock_file.return_value.read())

    def list_data_files(self):
        """
        List all files in the test data directory.
        """
        return os.listdir(self.test_data_dir)

    def setup_file(self, mock_file):
        """
        Mock the contents of a configuration file.
        """
        mock_file.return_value = mock_open(
            read_data="--- \n \nrun1:\n  "
            "top_traj_file: ['/path/to/tpr', '/path/to/trr']\n  "
            "selection_string: 'all'\n  "
            "start: 0\n  "
            "end: -1\n  "
            "step: 1\n  "
            "bin_width: 30\n  "
            "tempra: 298.0\n  "
            "verbose: False\n  "
            "thread: 1\n  "
            "output_file: 'output_file.json'\n  "
            "force_partitioning: 0.5\n  "
            "water_entropy: False"
        ).return_value

    @patch("builtins.open")
    @patch("glob.glob", return_value=["config.yaml"])
    def test_load_config(self, mock_glob, mock_file):
        """
        Test loading a valid configuration file.
        """
        # Setup the mock file content
        self.setup_file(mock_file)

        arg_config = ConfigManager()
        config = arg_config.load_config("/some/path")

        self.assertIn("run1", config)
        self.assertEqual(
            config["run1"]["top_traj_file"], ["/path/to/tpr", "/path/to/trr"]
        )
        self.assertEqual(config["run1"]["selection_string"], "all")
        self.assertEqual(config["run1"]["start"], 0)
        self.assertEqual(config["run1"]["end"], -1)
        self.assertEqual(config["run1"]["step"], 1)
        self.assertEqual(config["run1"]["bin_width"], 30)
        self.assertEqual(config["run1"]["tempra"], 298.0)
        self.assertFalse(config["run1"]["verbose"])
        self.assertEqual(config["run1"]["thread"], 1)
        self.assertEqual(config["run1"]["output_file"], "output_file.json")
        self.assertEqual(config["run1"]["force_partitioning"], 0.5)
        self.assertFalse(config["run1"]["water_entropy"])

    @patch("glob.glob", return_value=[])
    def test_load_config_no_yaml_files(self, mock_glob):
        arg_config = ConfigManager()
        config = arg_config.load_config("/some/path")
        self.assertEqual(config, {"run1": {}})

    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("glob.glob", return_value=["config.yaml"])
    def test_load_config_file_not_found(self, mock_glob, mock_open):
        """
        Test loading a configuration file that exists but cannot be opened.
        Should return default config instead of raising an error.
        """
        arg_config = ConfigManager()
        config = arg_config.load_config("/some/path")
        self.assertEqual(config, {"run1": {}})

    @patch.object(ConfigManager, "load_config", return_value=None)
    def test_no_cli_no_yaml(self, mock_load_config):
        """Test behavior when no CLI arguments and no YAML file are provided."""
        with self.assertRaises(SystemExit) as context:
            main()
        self.assertEqual(context.exception.code, 1)

    def test_invalid_run_config_type(self):
        """
        Test that passing an invalid type for run_config raises a TypeError.
        """
        arg_config = ConfigManager()
        args = MagicMock()
        invalid_configs = ["string", 123, 3.14, ["list"], {("tuple_key",): "value"}]

        for invalid in invalid_configs:
            with self.assertRaises(TypeError):
                arg_config.merge_configs(args, invalid)

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            top_traj_file=["/path/to/tpr", "/path/to/trr"],
            selection_string="all",
            start=0,
            end=-1,
            step=1,
            bin_width=30,
            tempra=298.0,
            verbose=False,
            thread=1,
            output_file="output_file.json",
            force_partitioning=0.5,
            water_entropy=False,
        ),
    )
    def test_setup_argparse(self, mock_args):
        """
        Test parsing command-line arguments.
        """
        arg_config = ConfigManager()
        parser = arg_config.setup_argparse()
        args = parser.parse_args()
        self.assertEqual(args.top_traj_file, ["/path/to/tpr", "/path/to/trr"])
        self.assertEqual(args.selection_string, "all")

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            top_traj_file=["/path/to/tpr", "/path/to/trr"],
            start=10,
            water_entropy=False,
        ),
    )
    def test_setup_argparse_false_boolean(self, mock_args):
        """
        Test that non-boolean arguments are parsed correctly.
        """
        arg_config = ConfigManager()
        parser = arg_config.setup_argparse()
        args = parser.parse_args()

        self.assertEqual(args.top_traj_file, ["/path/to/tpr", "/path/to/trr"])
        self.assertEqual(args.start, 10)
        self.assertFalse(args.water_entropy)

    def test_str2bool_true_variants(self):
        """Test that various string representations of True are correctly parsed."""
        arg_config = ConfigManager()

        self.assertTrue(arg_config.str2bool("true"))
        self.assertTrue(arg_config.str2bool("True"))
        self.assertTrue(arg_config.str2bool("t"))
        self.assertTrue(arg_config.str2bool("yes"))
        self.assertTrue(arg_config.str2bool("1"))

    def test_str2bool_false_variants(self):
        """Test that various string representations of False are correctly parsed."""
        arg_config = ConfigManager()

        self.assertFalse(arg_config.str2bool("false"))
        self.assertFalse(arg_config.str2bool("False"))
        self.assertFalse(arg_config.str2bool("f"))
        self.assertFalse(arg_config.str2bool("no"))
        self.assertFalse(arg_config.str2bool("0"))

    def test_str2bool_boolean_passthrough(self):
        """Test that boolean values passed directly are returned unchanged."""
        arg_config = ConfigManager()

        self.assertTrue(arg_config.str2bool(True))
        self.assertFalse(arg_config.str2bool(False))

    def test_str2bool_invalid_input(self):
        """Test that invalid string inputs raise an ArgumentTypeError."""
        arg_config = ConfigManager()

        with self.assertRaises(Exception) as context:
            arg_config.str2bool("maybe")
        self.assertIn("Boolean value expected", str(context.exception))

    def test_str2bool_empty_string(self):
        """Test that an empty string raises an ArgumentTypeError."""
        arg_config = ConfigManager()

        with self.assertRaises(Exception) as context:
            arg_config.str2bool("")
        self.assertIn("Boolean value expected", str(context.exception))

    def test_str2bool_unexpected_number(self):
        """Test that unexpected numeric strings raise an ArgumentTypeError."""
        arg_config = ConfigManager()

        with self.assertRaises(Exception) as context:
            arg_config.str2bool("2")
        self.assertIn("Boolean value expected", str(context.exception))

    def test_cli_overrides_defaults(self):
        """
        Test if CLI parameters override default values.
        """
        arg_config = ConfigManager()
        parser = arg_config.setup_argparse()
        args = parser.parse_args(
            ["--top_traj_file", "/cli/path", "--selection_string", "cli_value"]
        )
        self.assertEqual(args.top_traj_file, ["/cli/path"])
        self.assertEqual(args.selection_string, "cli_value")

    def test_cli_overrides_yaml(self):
        """
        Test if CLI parameters override YAML parameters correctly.
        """
        arg_config = ConfigManager()
        parser = arg_config.setup_argparse()
        args = parser.parse_args(
            ["--top_traj_file", "/cli/path", "--selection_string", "cli_value"]
        )
        run_config = {"top_traj_file": ["/yaml/path"], "selection_string": "yaml_value"}
        merged_args = arg_config.merge_configs(args, run_config)
        self.assertEqual(merged_args.top_traj_file, ["/cli/path"])
        self.assertEqual(merged_args.selection_string, "cli_value")

    def test_cli_overrides_yaml_with_multiple_values(self):
        """
        Ensures that CLI arguments override YAML when multiple values are provided in
        YAML.
        """
        arg_config = ConfigManager()
        yaml_config = {"top_traj_file": ["/yaml/path1", "/yaml/path2"]}
        args = argparse.Namespace(top_traj_file=["/cli/path"])

        merged_args = arg_config.merge_configs(args, yaml_config)

        self.assertEqual(merged_args.top_traj_file, ["/cli/path"])

    def test_yaml_overrides_defaults(self):
        """
        Test if YAML parameters override default values.
        """
        run_config = {"top_traj_file": ["/yaml/path"], "selection_string": "yaml_value"}
        args = argparse.Namespace()
        arg_config = ConfigManager()
        merged_args = arg_config.merge_configs(args, run_config)
        self.assertEqual(merged_args.top_traj_file, ["/yaml/path"])
        self.assertEqual(merged_args.selection_string, "yaml_value")

    def test_yaml_does_not_override_cli_if_set(self):
        """
        Ensure YAML does not override CLI arguments that are set.
        """
        arg_config = ConfigManager()

        yaml_config = {"bin_width": 50}
        args = argparse.Namespace(bin_width=100)

        merged_args = arg_config.merge_configs(args, yaml_config)

        self.assertEqual(merged_args.bin_width, 100)

    def test_yaml_overrides_defaults_when_no_cli(self):
        """
        Test if YAML parameters override default values when no CLI input is given.
        """
        arg_config = ConfigManager()

        yaml_config = {
            "top_traj_file": ["/yaml/path"],
            "bin_width": 50,
        }

        args = argparse.Namespace()

        merged_args = arg_config.merge_configs(args, yaml_config)

        self.assertEqual(merged_args.top_traj_file, ["/yaml/path"])
        self.assertEqual(merged_args.bin_width, 50)

    def test_yaml_none_does_not_override_defaults(self):
        """
        Ensures that YAML values set to `None` do not override existing CLI values.
        """
        arg_config = ConfigManager()
        yaml_config = {"bin_width": None}
        args = argparse.Namespace(bin_width=100)

        merged_args = arg_config.merge_configs(args, yaml_config)

        self.assertEqual(merged_args.bin_width, 100)

    def test_hierarchy_cli_yaml_defaults(self):
        """
        Test if CLI arguments override YAML, and YAML overrides defaults.
        """
        arg_config = ConfigManager()

        yaml_config = {
            "top_traj_file": ["/yaml/path", "/yaml/path"],
            "bin_width": "50",
        }

        args = argparse.Namespace(
            top_traj_file=["/cli/path", "/cli/path"], bin_width=100
        )

        merged_args = arg_config.merge_configs(args, yaml_config)

        self.assertEqual(merged_args.top_traj_file, ["/cli/path", "/cli/path"])
        self.assertEqual(merged_args.bin_width, 100)

    def test_merge_configs(self):
        """
        Test merging default arguments with a run configuration.
        """
        arg_config = ConfigManager()
        args = MagicMock(
            top_traj_file=None,
            selection_string=None,
            start=None,
            end=None,
            step=None,
            bin_width=None,
            tempra=None,
            verbose=None,
            thread=None,
            output_file=None,
            force_partitioning=None,
            water_entropy=None,
        )
        run_config = {
            "top_traj_file": ["/path/to/tpr", "/path/to/trr"],
            "selection_string": "all",
            "start": 0,
            "end": -1,
            "step": 1,
            "bin_width": 30,
            "tempra": 298.0,
            "verbose": False,
            "thread": 1,
            "output_file": "output_file.json",
            "force_partitioning": 0.5,
            "water_entropy": False,
        }
        merged_args = arg_config.merge_configs(args, run_config)
        self.assertEqual(merged_args.top_traj_file, ["/path/to/tpr", "/path/to/trr"])
        self.assertEqual(merged_args.selection_string, "all")

    def test_merge_with_none_yaml(self):
        """
        Ensure merging still works if no YAML config is provided.
        """
        arg_config = ConfigManager()

        args = argparse.Namespace(top_traj_file=["/cli/path"])
        yaml_config = None

        merged_args = arg_config.merge_configs(args, yaml_config)

        self.assertEqual(merged_args.top_traj_file, ["/cli/path"])

    @patch("CodeEntropy.config.arg_config_manager.logger")
    def test_merge_configs_sets_debug_logging(self, mock_logger):
        """
        Ensure logging is set to DEBUG when verbose=True.
        """
        arg_config = ConfigManager()
        args = argparse.Namespace(verbose=True)
        for key in arg_config.arg_map:
            if not hasattr(args, key):
                setattr(args, key, None)

        # Mock logger handlers
        mock_handler = MagicMock()
        mock_logger.handlers = [mock_handler]

        arg_config.merge_configs(args, {})

        mock_logger.setLevel.assert_called_with(logging.DEBUG)
        mock_handler.setLevel.assert_called_with(logging.DEBUG)
        mock_logger.debug.assert_called_with(
            "Verbose mode enabled. Logger set to DEBUG level."
        )

    @patch("CodeEntropy.config.arg_config_manager.logger")
    def test_merge_configs_sets_info_logging(self, mock_logger):
        """
        Ensure logging is set to INFO when verbose=False.
        """
        arg_config = ConfigManager()
        args = argparse.Namespace(verbose=False)
        for key in arg_config.arg_map:
            if not hasattr(args, key):
                setattr(args, key, None)

        # Mock logger handlers
        mock_handler = MagicMock()
        mock_logger.handlers = [mock_handler]

        arg_config.merge_configs(args, {})

        mock_logger.setLevel.assert_called_with(logging.INFO)
        mock_handler.setLevel.assert_called_with(logging.INFO)

    @patch("argparse.ArgumentParser.parse_args")
    def test_default_values(self, mock_parse_args):
        """
        Test if argument parser assigns default values correctly.
        """
        arg_config = ConfigManager()
        mock_parse_args.return_value = MagicMock(
            top_traj_file=["example.top", "example.traj"]
        )
        parser = arg_config.setup_argparse()
        args = parser.parse_args()
        self.assertEqual(args.top_traj_file, ["example.top", "example.traj"])

    def test_fallback_to_defaults(self):
        """
        Ensure arguments fall back to defaults if neither YAML nor CLI provides them.
        """
        arg_config = ConfigManager()

        yaml_config = {}
        args = argparse.Namespace()

        merged_args = arg_config.merge_configs(args, yaml_config)

        self.assertEqual(merged_args.step, 1)
        self.assertEqual(merged_args.end, -1)

    @patch(
        "argparse.ArgumentParser.parse_args", return_value=MagicMock(top_traj_file=None)
    )
    def test_missing_required_arguments(self, mock_args):
        """
        Test behavior when required arguments are missing.
        """
        arg_config = ConfigManager()
        parser = arg_config.setup_argparse()
        args = parser.parse_args()
        with self.assertRaises(ValueError):
            if not args.top_traj_file:
                raise ValueError(
                    "The 'top_traj_file' argument is required but not provided."
                )

    def test_invalid_argument_type(self):
        """
        Test handling of invalid argument types.
        """
        arg_config = ConfigManager()
        parser = arg_config.setup_argparse()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--start", "invalid"])

    @patch(
        "argparse.ArgumentParser.parse_args", return_value=MagicMock(start=-1, end=-10)
    )
    def test_edge_case_argument_values(self, mock_args):
        """
        Test parsing of edge case values.
        """
        arg_config = ConfigManager()
        parser = arg_config.setup_argparse()
        args = parser.parse_args()
        self.assertEqual(args.start, -1)
        self.assertEqual(args.end, -10)

    @patch("builtins.open", new_callable=mock_open, read_data="--- \n")
    @patch("glob.glob", return_value=["config.yaml"])
    def test_empty_yaml_config(self, mock_glob, mock_file):
        """
        Test behavior when an empty YAML file is provided.
        Should return default config {'run1': {}}.
        """
        arg_config = ConfigManager()
        config = arg_config.load_config("/some/path")

        self.assertIsInstance(config, dict)
        self.assertEqual(config, {"run1": {}})

    def test_input_parameters_validation_all_valid(self):
        """Test that input_parameters_validation passes with all valid inputs."""
        manager = ConfigManager()
        u = MagicMock()
        u.trajectory = [0] * 100

        args = MagicMock(
            start=10,
            end=90,
            step=1,
            bin_width=30,
            temperature=298.0,
            force_partitioning=0.5,
        )

        with patch.dict(
            "CodeEntropy.config.arg_config_manager.arg_map",
            {"force_partitioning": {"default": 0.5}},
        ):
            manager.input_parameters_validation(u, args)

    def test_check_input_start_valid(self):
        """Test that a valid 'start' value does not raise an error."""
        args = MagicMock(start=50)
        u = MagicMock()
        u.trajectory = [0] * 100
        ConfigManager()._check_input_start(u, args)

    def test_check_input_start_invalid(self):
        """Test that an invalid 'start' value raises a ValueError."""
        args = MagicMock(start=150)
        u = MagicMock()
        u.trajectory = [0] * 100
        with self.assertRaises(ValueError):
            ConfigManager()._check_input_start(u, args)

    def test_check_input_end_valid(self):
        """Test that a valid 'end' value does not raise an error."""
        args = MagicMock(end=100)
        u = MagicMock()
        u.trajectory = [0] * 100
        ConfigManager()._check_input_end(u, args)

    def test_check_input_end_invalid(self):
        """Test that an 'end' value exceeding trajectory length raises a ValueError."""
        args = MagicMock(end=101)
        u = MagicMock()
        u.trajectory = [0] * 100
        with self.assertRaises(ValueError):
            ConfigManager()._check_input_end(u, args)

    @patch("CodeEntropy.config.arg_config_manager.logger")
    def test_check_input_step_negative(self, mock_logger):
        """Test that a negative 'step' value triggers a warning."""
        args = MagicMock(step=-1)
        ConfigManager()._check_input_step(args)
        mock_logger.warning.assert_called_once()

    def test_check_input_bin_width_valid(self):
        """Test that a valid 'bin_width' value does not raise an error."""
        args = MagicMock(bin_width=180)
        ConfigManager()._check_input_bin_width(args)

    def test_check_input_bin_width_invalid_low(self):
        """Test that a negative 'bin_width' value raises a ValueError."""
        args = MagicMock(bin_width=-10)
        with self.assertRaises(ValueError):
            ConfigManager()._check_input_bin_width(args)

    def test_check_input_bin_width_invalid_high(self):
        """Test that a 'bin_width' value above 360 raises a ValueError."""
        args = MagicMock(bin_width=400)
        with self.assertRaises(ValueError):
            ConfigManager()._check_input_bin_width(args)

    def test_check_input_temperature_valid(self):
        """Test that a valid 'temperature' value does not raise an error."""
        args = MagicMock(temperature=298.0)
        ConfigManager()._check_input_temperature(args)

    def test_check_input_temperature_invalid(self):
        """Test that a negative 'temperature' value raises a ValueError."""
        args = MagicMock(temperature=-5)
        with self.assertRaises(ValueError):
            ConfigManager()._check_input_temperature(args)

    @patch("CodeEntropy.config.arg_config_manager.logger")
    def test_check_input_force_partitioning_warning(self, mock_logger):
        """Test that a non-default 'force_partitioning' value triggers a warning."""
        args = MagicMock(force_partitioning=0.7)
        with patch.dict(
            "CodeEntropy.config.arg_config_manager.arg_map",
            {"force_partitioning": {"default": 0.5}},
        ):
            ConfigManager()._check_input_force_partitioning(args)
            mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
