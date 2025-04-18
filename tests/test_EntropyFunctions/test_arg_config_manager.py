import argparse
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

import tests.data as data
from CodeEntropy.config.arg_config_manager import ConfigManager
from CodeEntropy.main_mcc import main


class test_arg_config_manager(unittest.TestCase):
    """
    Unit tests for the ConfigManager.
    """

    def setUp(self):
        """
        Setup test data and output directories.
        """
        self.test_data_dir = os.path.dirname(data.__file__)
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.config_file = os.path.join(self.test_dir, "config.yaml")

        # Create a mock config file
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            self.setup_file(mock_file)
            with open(self.config_file, "w") as f:
                f.write(mock_file.return_value.read())

        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """
        Clean up after each test.
        """
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

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

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_load_config(self, mock_exists, mock_file):
        """
        Test loading a valid configuration file.
        """
        arg_config = ConfigManager()
        self.setup_file(mock_file)
        config = arg_config.load_config(self.config_file)
        self.assertIn("run1", config)
        self.assertEqual(
            config["run1"]["top_traj_file"], ["/path/to/tpr", "/path/to/trr"]
        )

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_config_file_not_found(self, mock_file):
        """
        Test loading a configuration file that does not exist.
        """
        arg_config = ConfigManager()
        with self.assertRaises(FileNotFoundError):
            arg_config.load_config(self.config_file)

    @patch.object(ConfigManager, "load_config", return_value=None)
    def test_no_cli_no_yaml(self, mock_load_config):
        """Test behavior when no CLI arguments and no YAML file are provided."""
        with self.assertRaises(ValueError) as context:
            main()
        self.assertEqual(
            str(context.exception),
            "No configuration file found, and no CLI arguments were provided.",
        )

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
    @patch("os.path.exists", return_value=True)
    def test_empty_yaml_config(self, mock_exists, mock_file):
        """
        Test behavior when an empty YAML file is provided.
        Should use defaults or raise an appropriate error.
        """

        arg_config = ConfigManager()

        config = arg_config.load_config(self.config_file)

        self.assertIsInstance(config, dict)
        self.assertEqual(config, {})


if __name__ == "__main__":
    unittest.main()
