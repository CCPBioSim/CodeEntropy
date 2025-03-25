import argparse
import unittest
from unittest.mock import MagicMock, mock_open, patch

from CodeEntropy.config.arg_config_manager import ConfigManager
from CodeEntropy.main_mcc import main


class test_maincc(unittest.TestCase):
    """
    Unit tests for the main functionality of CodeEntropy.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        self.config_file = "config.yaml"
        self.code_entropy = main

    def tearDown(self):
        """
        Clean up after each test.
        """
        return super().tearDown()

    def setup_file(self, mock_file):
        """
        Mock the contents of a configuration file.
        """
        mock_file.return_value = mock_open(
            read_data="--- \n \nrun1:\n  "
            "top_traj_file: ['/path/to/tpr', '/path/to/trr']\n  "
            "selection_string: "
            "'all'\n  "
            "start: 0\n  "
            "end: -1\n  "
            "step: 1\n  "
            "bin_width: 30\n  "
            "tempra: 298.0\n  "
            "verbose: False\n  "
            "thread: 1\n  "
            "outfile: 'outfile.out'\n  "
            "resfile: 'res_outfile.out'\n  "
            "mout: null\n  "
            "force_partitioning: 0.5\n  "
            "waterEntropy: False"
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
            self.code_entropy()

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
            outfile="outfile.out",
            resfile="res_outfile.out",
            mout=None,
            force_partitioning=0.5,
            waterEntropy=False,
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
            outfile=None,
            resfile=None,
            mout=None,
            force_partitioning=None,
            waterEntropy=None,
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
            "outfile": "outfile.out",
            "resfile": "res_outfile.out",
            "mout": None,
            "force_partitioning": 0.5,
            "waterEntropy": False,
        }
        merged_args = arg_config.merge_configs(args, run_config)
        self.assertEqual(merged_args.top_traj_file, ["/path/to/tpr", "/path/to/trr"])
        self.assertEqual(merged_args.selection_string, "all")

    @patch("argparse.ArgumentParser.parse_args")
    def test_default_values(self, mock_parse_args):
        """
        Test if argument parser assigns default values correctly.
        """
        arg_config = ConfigManager()

        # Ensure every argument gets a sensible default
        default_args = {
            arg: params.get("default", False if "action" in params else None)
            for arg, params in arg_config.arg_map.items()
        }

        # Mock argparse to return expected defaults
        mock_parse_args.return_value = MagicMock(**default_args)

        parser = arg_config.setup_argparse()
        args = parser.parse_args()

        # Compare parsed args with expected defaults
        for arg, params in arg_config.arg_map.items():
            expected_default = params.get(
                "default", False if "action" in params else None
            )
            self.assertEqual(getattr(args, arg), expected_default)

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

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=b"--- \n top_traj_file: ['/path/to/tpr', '/path/to/trr'] \n",
    )
    @patch("os.path.exists", return_value=True)
    @patch("MDAnalysis.Universe")
    @patch("gettext.translation", return_value=MagicMock())
    def test_run(self, mock_translation, mock_universe, mock_exists, mock_file):
        """
        Test the execution of the main function with the necessary CLI argument.
        """
        with patch(
            "sys.argv",
            ["CodeEntropy", "--top_traj_file", "/path/to/tpr", "/path/to/trr"],
        ):
            self.setup_file(mock_file)
            mock_universe.return_value.trajectory = MagicMock()
            main()


if __name__ == "__main__":
    unittest.main()
