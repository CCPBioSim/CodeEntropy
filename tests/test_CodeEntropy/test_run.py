import os
import unittest
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import requests
import yaml
from rich.console import Console

from CodeEntropy.run import RunManager
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestRunManager(BaseTestCase):
    """
    Unit tests for the RunManager class. These tests verify the
    correct behavior of run manager.
    """

    def setUp(self):
        super().setUp()
        self.config_file = os.path.join(self.test_dir, "CITATION.cff")
        # Create mock config
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            self.setup_citation_file(mock_file)
            with open(self.config_file, "w") as f:
                f.write(mock_file.return_value.read())
        self.run_manager = RunManager(folder=self.test_dir)

    def setup_citation_file(self, mock_file):
        """
        Mock the contents of the CITATION.cff file.
        """
        citation_content = """\
    authors:
    - given-names: Alice
      family-names: Smith
    """

        mock_file.return_value = mock_open(read_data=citation_content).return_value

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_empty_directory(self, mock_listdir, mock_makedirs):
        """
        Test that 'job001' is created when the directory is initially empty.
        """
        mock_listdir.return_value = []
        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job001")
        self.assertEqual(
            os.path.realpath(new_folder_path), os.path.realpath(expected_path)
        )

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_with_existing_folders(self, mock_listdir, mock_makedirs):
        """
        Test that the next sequential job folder (e.g., 'job004') is created when
        existing folders 'job001', 'job002', and 'job003' are present.
        """
        mock_listdir.return_value = ["job001", "job002", "job003"]
        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job004")

        # Normalize paths cross-platform
        normalized_new = os.path.normcase(
            os.path.realpath(os.path.normpath(new_folder_path))
        )
        normalized_expected = os.path.normcase(
            os.path.realpath(os.path.normpath(expected_path))
        )

        self.assertEqual(normalized_new, normalized_expected)

        called_args, called_kwargs = mock_makedirs.call_args
        normalized_called = os.path.normcase(
            os.path.realpath(os.path.normpath(called_args[0]))
        )
        self.assertEqual(normalized_called, normalized_expected)
        self.assertTrue(called_kwargs.get("exist_ok", False))

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_with_non_matching_folders(
        self, mock_listdir, mock_makedirs
    ):
        """
        Test that 'job001' is created when the directory contains only non-job-related
        folders.
        """
        mock_listdir.return_value = ["folderA", "another_one"]

        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job001")

        normalized_new = os.path.normcase(
            os.path.realpath(os.path.normpath(new_folder_path))
        )
        normalized_expected = os.path.normcase(
            os.path.realpath(os.path.normpath(expected_path))
        )
        self.assertEqual(normalized_new, normalized_expected)

        called_args, called_kwargs = mock_makedirs.call_args
        normalized_called = os.path.normcase(
            os.path.realpath(os.path.normpath(called_args[0]))
        )
        self.assertEqual(normalized_called, normalized_expected)
        self.assertTrue(called_kwargs.get("exist_ok", False))

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_mixed_folder_names(self, mock_listdir, mock_makedirs):
        """
        Test that the correct next job folder (e.g., 'job003') is created when both
        job and non-job folders exist in the directory.
        """
        mock_listdir.return_value = ["job001", "abc", "job002", "random"]
        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job003")

        normalized_new = os.path.normcase(
            os.path.realpath(os.path.normpath(new_folder_path))
        )
        normalized_expected = os.path.normcase(
            os.path.realpath(os.path.normpath(expected_path))
        )
        self.assertEqual(normalized_new, normalized_expected)

        called_args, called_kwargs = mock_makedirs.call_args
        normalized_called = os.path.normcase(
            os.path.realpath(os.path.normpath(called_args[0]))
        )
        self.assertEqual(normalized_called, normalized_expected)
        self.assertTrue(called_kwargs.get("exist_ok", False))

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_with_invalid_job_suffix(
        self, mock_listdir, mock_makedirs
    ):
        """
        Test that invalid job folder names like 'jobABC' are ignored when determining
        the next job number.
        """
        # Simulate existing folders, one of which is invalid
        mock_listdir.return_value = ["job001", "jobABC", "job002"]

        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job003")

        normalized_new = os.path.normcase(
            os.path.realpath(os.path.normpath(new_folder_path))
        )
        normalized_expected = os.path.normcase(
            os.path.realpath(os.path.normpath(expected_path))
        )
        self.assertEqual(normalized_new, normalized_expected)

        called_args, called_kwargs = mock_makedirs.call_args
        normalized_called = os.path.normcase(
            os.path.realpath(os.path.normpath(called_args[0]))
        )
        self.assertEqual(normalized_called, normalized_expected)
        self.assertTrue(called_kwargs.get("exist_ok", False))

    @patch("requests.get")
    def test_load_citation_data_success(self, mock_get):
        """Should return parsed dict when CITATION.cff loads successfully."""
        mock_yaml = """
        authors:
          - given-names: Alice
            family-names: Smith
        title: TestProject
        version: 1.0
        date-released: 2025-01-01
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_yaml
        mock_get.return_value = mock_response

        instance = RunManager("dummy")
        data = instance.load_citation_data()

        self.assertIsInstance(data, dict)
        self.assertEqual(data["title"], "TestProject")
        self.assertEqual(data["authors"][0]["given-names"], "Alice")

    @patch("requests.get")
    def test_load_citation_data_network_error(self, mock_get):
        """Should return None if network request fails."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network down")

        instance = RunManager("dummy")
        data = instance.load_citation_data()

        self.assertIsNone(data)

    @patch("requests.get")
    def test_load_citation_data_http_error(self, mock_get):
        """Should return None if HTTP response is non-200."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response

        instance = RunManager("dummy")
        data = instance.load_citation_data()

        self.assertIsNone(data)

    @patch("requests.get")
    def test_load_citation_data_invalid_yaml(self, mock_get):
        """Should raise YAML error if file content is invalid YAML."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "invalid: [oops"
        mock_get.return_value = mock_response

        instance = RunManager("dummy")

        with self.assertRaises(yaml.YAMLError):
            instance.load_citation_data()

    @patch.object(RunManager, "load_citation_data")
    def test_show_splash_with_citation(self, mock_load):
        """Should render full splash screen when citation data is present."""
        mock_load.return_value = {
            "title": "TestProject",
            "version": "1.0",
            "date-released": "2025-01-01",
            "url": "https://example.com",
            "abstract": "This is a test abstract.",
            "authors": [
                {"given-names": "Alice", "family-names": "Smith", "affiliation": "Uni"}
            ],
        }

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False, width=80)

        instance = RunManager("dummy")
        with patch("CodeEntropy.run.console", test_console):
            instance.show_splash()

        output = buf.getvalue()

        self.assertIn("Version 1.0", output)
        self.assertIn("2025-01-01", output)
        self.assertIn("https://example.com", output)
        self.assertIn("This is a test abstract.", output)
        self.assertIn("Alice Smith", output)

    @patch.object(RunManager, "load_citation_data", return_value=None)
    def test_show_splash_without_citation(self, mock_load):
        """Should render minimal splash screen when no citation data."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False, width=80)

        instance = RunManager("dummy")
        with patch("CodeEntropy.run.console", test_console):
            instance.show_splash()

        output = buf.getvalue()

        self.assertNotIn("Version", output)
        self.assertNotIn("Contributors", output)
        self.assertIn("Welcome to CodeEntropy", output)

    @patch.object(RunManager, "load_citation_data")
    def test_show_splash_missing_fields(self, mock_load):
        """Should gracefully handle missing optional fields in citation data."""
        mock_load.return_value = {
            "title": "PartialProject",
            # no version, no date, no authors, no abstract
        }

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False, width=80)

        instance = RunManager("dummy")
        with patch("CodeEntropy.run.console", test_console):
            instance.show_splash()

        output = buf.getvalue()

        self.assertIn("Version ?", output)
        self.assertIn("No description available.", output)

    def test_run_entropy_workflow(self):
        """
        Test the run_entropy_workflow method to ensure it initializes and executes
        correctly with mocked dependencies.
        """
        run_manager = RunManager("mock_folder/job001")
        run_manager._logging_config = MagicMock()
        run_manager._config_manager = MagicMock()
        run_manager.load_citation_data = MagicMock()
        run_manager._data_logger = MagicMock()
        run_manager.folder = self.test_dir

        mock_logger = MagicMock()
        run_manager._logging_config.setup_logging.return_value = mock_logger

        run_manager._config_manager.load_config.return_value = {
            "test_run": {
                "top_traj_file": ["/path/to/tpr", "/path/to/trr"],
                "selection_string": "all",
                "output_file": "output.json",
                "verbose": True,
            }
        }

        run_manager.load_citation_data.return_value = {
            "cff-version": "1.2.0",
            "title": "CodeEntropy",
            "message": (
                "If you use this software, please cite it using the "
                "metadata from this file."
            ),
            "type": "software",
            "authors": [
                {
                    "given-names": "Forename",
                    "family-names": "Sirname",
                    "email": "test@email.ac.uk",
                }
            ],
        }

        mock_args = MagicMock()
        mock_args.output_file = "output.json"
        mock_args.verbose = True
        mock_args.top_traj_file = ["/path/to/tpr", "/path/to/trr"]
        mock_args.selection_string = "all"
        parser = run_manager._config_manager.setup_argparse.return_value
        parser.parse_known_args.return_value = (mock_args, [])

        run_manager._config_manager.merge_configs.return_value = mock_args

        mock_entropy_manager = MagicMock()
        with (
            unittest.mock.patch(
                "CodeEntropy.run.EntropyManager", return_value=mock_entropy_manager
            ),
            unittest.mock.patch("CodeEntropy.run.mda.Universe") as mock_universe,
        ):

            run_manager.run_entropy_workflow()

            mock_universe.assert_called_once_with("/path/to/tpr", ["/path/to/trr"])
            mock_entropy_manager.execute.assert_called_once()

    def test_run_configuration_warning(self):
        """
        Test that a warning is logged when the config entry is not a dictionary.
        """
        run_manager = RunManager("mock_folder/job001")
        run_manager._logging_config = MagicMock()
        run_manager._config_manager = MagicMock()
        run_manager.load_citation_data = MagicMock()
        run_manager._data_logger = MagicMock()
        run_manager.folder = self.test_dir

        mock_logger = MagicMock()
        run_manager._logging_config.setup_logging.return_value = mock_logger

        run_manager._config_manager.load_config.return_value = {
            "invalid_run": "this_should_be_a_dict"
        }

        run_manager.load_citation_data.return_value = {
            "cff-version": "1.2.0",
            "title": "CodeEntropy",
            "message": (
                "If you use this software, please cite it using the "
                "metadata from this file."
            ),
            "type": "software",
            "authors": [
                {
                    "given-names": "Forename",
                    "family-names": "Sirname",
                    "email": "test@email.ac.uk",
                }
            ],
        }

        mock_args = MagicMock()
        mock_args.output_file = "output.json"
        mock_args.verbose = False

        parser = run_manager._config_manager.setup_argparse.return_value
        parser.parse_known_args.return_value = (mock_args, [])
        run_manager._config_manager.merge_configs.return_value = mock_args

        run_manager.run_entropy_workflow()

        mock_logger.warning.assert_called_with(
            "Run configuration for invalid_run is not a dictionary."
        )

    def test_run_entropy_workflow_missing_traj_file(self):
        """
        Test that a ValueError is raised when 'top_traj_file' is missing.
        """
        run_manager = RunManager("mock_folder/job001")
        run_manager._logging_config = MagicMock()
        run_manager._config_manager = MagicMock()
        run_manager.load_citation_data = MagicMock()
        run_manager._data_logger = MagicMock()
        run_manager.folder = self.test_dir

        mock_logger = MagicMock()
        run_manager._logging_config.setup_logging.return_value = mock_logger

        run_manager._config_manager.load_config.return_value = {
            "test_run": {
                "top_traj_file": None,
                "output_file": "output.json",
                "verbose": False,
            }
        }

        run_manager.load_citation_data.return_value = {
            "cff-version": "1.2.0",
            "title": "CodeEntropy",
            "message": (
                "If you use this software, please cite it using the "
                "metadata from this file."
            ),
            "type": "software",
            "authors": [
                {
                    "given-names": "Forename",
                    "family-names": "Sirname",
                    "email": "test@email.ac.uk",
                }
            ],
        }

        mock_args = MagicMock()
        mock_args.output_file = "output.json"
        mock_args.verbose = False
        mock_args.top_traj_file = None
        mock_args.selection_string = None

        parser = run_manager._config_manager.setup_argparse.return_value
        parser.parse_known_args.return_value = (mock_args, [])
        run_manager._config_manager.merge_configs.return_value = mock_args

        with self.assertRaisesRegex(ValueError, "Missing 'top_traj_file' argument."):
            run_manager.run_entropy_workflow()

    def test_run_entropy_workflow_missing_selection_string(self):
        """
        Test that a ValueError is raised when 'selection_string' is missing.
        """
        run_manager = RunManager("mock_folder/job001")
        run_manager._logging_config = MagicMock()
        run_manager._config_manager = MagicMock()
        run_manager.load_citation_data = MagicMock()
        run_manager._data_logger = MagicMock()
        run_manager.folder = self.test_dir

        mock_logger = MagicMock()
        run_manager._logging_config.setup_logging.return_value = mock_logger

        run_manager._config_manager.load_config.return_value = {
            "test_run": {
                "top_traj_file": ["/path/to/tpr", "/path/to/trr"],
                "output_file": "output.json",
                "verbose": False,
            }
        }

        run_manager.load_citation_data.return_value = {
            "cff-version": "1.2.0",
            "title": "CodeEntropy",
            "message": (
                "If you use this software, please cite it using the "
                "metadata from this file."
            ),
            "type": "software",
            "authors": [
                {
                    "given-names": "Forename",
                    "family-names": "Sirname",
                    "email": "test@email.ac.uk",
                }
            ],
        }

        mock_args = MagicMock()
        mock_args.output_file = "output.json"
        mock_args.verbose = False
        mock_args.top_traj_file = ["/path/to/tpr", "/path/to/trr"]
        mock_args.selection_string = None

        parser = run_manager._config_manager.setup_argparse.return_value
        parser.parse_known_args.return_value = (mock_args, [])
        run_manager._config_manager.merge_configs.return_value = mock_args

        with self.assertRaisesRegex(ValueError, "Missing 'selection_string' argument."):
            run_manager.run_entropy_workflow()

    @patch("CodeEntropy.run.AnalysisFromFunction")
    @patch("CodeEntropy.run.mda.Merge")
    def test_new_U_select_frame(self, MockMerge, MockAnalysisFromFunction):
        # Mock Universe and its components
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_trajectory.__len__.return_value = 10
        mock_universe.trajectory = mock_trajectory

        mock_select_atoms = MagicMock()
        mock_universe.select_atoms.return_value = mock_select_atoms

        # Mock AnalysisFromFunction results for coordinates, forces, and dimensions
        coords = np.random.rand(10, 100, 3)
        forces = np.random.rand(10, 100, 3)
        dims = np.random.rand(10, 3)

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        mock_dims_analysis = MagicMock()
        mock_dims_analysis.run.return_value.results = {"timeseries": dims}

        # Set the side effects for the three AnalysisFromFunction calls
        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
            mock_dims_analysis,
        ]

        # Mock the merge operation
        mock_merged_universe = MagicMock()
        MockMerge.return_value = mock_merged_universe

        run_manager = RunManager("mock_folder/job001")
        result = run_manager.new_U_select_frame(mock_universe)

        mock_universe.select_atoms.assert_called_once_with("all", updating=True)
        MockMerge.assert_called_once_with(mock_select_atoms)

        # Ensure the 'load_new' method was called with the correct arguments
        mock_merged_universe.load_new.assert_called_once()
        args, kwargs = mock_merged_universe.load_new.call_args

        # Assert that the arrays are passed correctly
        np.testing.assert_array_equal(args[0], coords)
        np.testing.assert_array_equal(kwargs["forces"], forces)
        np.testing.assert_array_equal(kwargs["dimensions"], dims)

        # Check if format was included in the kwargs
        self.assertIn("format", kwargs)

        # Ensure the result is the mock merged universe
        self.assertEqual(result, mock_merged_universe)

    @patch("CodeEntropy.run.AnalysisFromFunction")
    @patch("CodeEntropy.run.mda.Merge")
    def test_new_U_select_atom(self, MockMerge, MockAnalysisFromFunction):
        # Mock Universe and its components
        mock_universe = MagicMock()
        mock_select_atoms = MagicMock()
        mock_universe.select_atoms.return_value = mock_select_atoms

        # Mock AnalysisFromFunction results for coordinates, forces, and dimensions
        coords = np.random.rand(10, 100, 3)
        forces = np.random.rand(10, 100, 3)
        dims = np.random.rand(10, 3)

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        mock_dims_analysis = MagicMock()
        mock_dims_analysis.run.return_value.results = {"timeseries": dims}

        # Set the side effects for the three AnalysisFromFunction calls
        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
            mock_dims_analysis,
        ]

        # Mock the merge operation
        mock_merged_universe = MagicMock()
        MockMerge.return_value = mock_merged_universe

        run_manager = RunManager("mock_folder/job001")
        result = run_manager.new_U_select_atom(
            mock_universe, select_string="resid 1-10"
        )

        mock_universe.select_atoms.assert_called_once_with("resid 1-10", updating=True)
        MockMerge.assert_called_once_with(mock_select_atoms)

        # Ensure the 'load_new' method was called with the correct arguments
        mock_merged_universe.load_new.assert_called_once()
        args, kwargs = mock_merged_universe.load_new.call_args

        # Assert that the arrays are passed correctly
        np.testing.assert_array_equal(args[0], coords)
        np.testing.assert_array_equal(kwargs["forces"], forces)
        np.testing.assert_array_equal(kwargs["dimensions"], dims)

        # Check if format was included in the kwargs
        self.assertIn("format", kwargs)

        # Ensure the result is the mock merged universe
        self.assertEqual(result, mock_merged_universe)

    @patch("CodeEntropy.run.pickle.dump")
    @patch("CodeEntropy.run.open", create=True)
    def test_write_universe(self, mock_open, mock_pickle_dump):
        # Mock Universe
        mock_universe = MagicMock()

        # Mock the file object returned by open
        mock_file = MagicMock()
        mock_open.return_value = mock_file

        run_manager = RunManager("mock_folder/job001")
        result = run_manager.write_universe(mock_universe, name="test_universe")

        mock_open.assert_called_once_with("test_universe.pkl", "wb")

        # Ensure pickle.dump() was called
        mock_pickle_dump.assert_called_once_with(mock_universe, mock_file)

        # Ensure the method returns the correct filename
        self.assertEqual(result, "test_universe")

    @patch("CodeEntropy.run.pickle.load")
    @patch("CodeEntropy.run.open", create=True)
    def test_read_universe(self, mock_open, mock_pickle_load):
        # Mock the file object returned by open
        mock_file = MagicMock()
        mock_open.return_value = mock_file

        # Mock Universe to return when pickle.load is called
        mock_universe = MagicMock()
        mock_pickle_load.return_value = mock_universe

        # Path to the mock file
        path = "test_universe.pkl"

        run_manager = RunManager("mock_folder/job001")
        result = run_manager.read_universe(path)

        mock_open.assert_called_once_with(path, "rb")

        # Ensure pickle.load() was called with the mock file object
        mock_pickle_load.assert_called_once_with(mock_file)

        # Ensure the method returns the correct mock universe
        self.assertEqual(result, mock_universe)


if __name__ == "__main__":
    unittest.main()
