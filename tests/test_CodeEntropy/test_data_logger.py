import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from CodeEntropy.config.data_logger import DataLogger
from CodeEntropy.main import main


class TestDataLogger(unittest.TestCase):
    """
    Unit tests for the DataLogger class. These tests verify the
    correct behavior of data logging, JSON export, and table
    logging functionalities.
    """

    def setUp(self):
        """
        Set up a temporary test environment before each test.
        Creates a temporary directory and initializes a DataLogger instance.
        """
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.code_entropy = main

        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        self.logger = DataLogger()
        self.output_file = "test_output.json"

    def tearDown(self):
        """
        Clean up the test environment after each test.
        Removes the temporary directory and restores the original working directory.
        """
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init(self):
        """
        Test that the DataLogger initializes with empty molecule and residue data lists.
        """
        self.assertEqual(self.logger.molecule_data, [])
        self.assertEqual(self.logger.residue_data, [])

    def test_add_results_data(self):
        """
        Test that add_results_data correctly appends a molecule-level entry.
        """
        self.logger.add_results_data(
            0, "united_atom", "Transvibrational (J/mol/K)", 653.404
        )
        self.assertEqual(
            self.logger.molecule_data,
            [[0, "united_atom", "Transvibrational (J/mol/K)", "653.404"]],
        )

    def test_add_residue_data(self):
        """
        Test that add_residue_data correctly appends a residue-level entry.
        """
        self.logger.add_residue_data(0, 0, "Transvibrational (J/mol/K)", 122.612)
        self.assertEqual(
            self.logger.residue_data, [[0, 0, "Transvibrational (J/mol/K)", "122.612"]]
        )

    def test_save_dataframes_as_json(self):
        """
        Test that save_dataframes_as_json correctly writes molecule and residue data
        to a JSON file with the expected structure and values.
        """
        molecule_df = pd.DataFrame(
            [
                {
                    "Molecule ID": 0,
                    "Level": "united_atom",
                    "Type": "Transvibrational (J/mol/K)",
                    "Result": 653.404,
                },
                {
                    "Molecule ID": 1,
                    "Level": "united_atom",
                    "Type": "Rovibrational (J/mol/K)",
                    "Result": 236.081,
                },
            ]
        )
        residue_df = pd.DataFrame(
            [
                {
                    "Molecule ID": 0,
                    "Residue": 0,
                    "Type": "Transvibrational (J/mol/K)",
                    "Result": 122.612,
                },
                {
                    "Molecule ID": 1,
                    "Residue": 0,
                    "Type": "Conformational (J/mol/K)",
                    "Result": 6.845,
                },
            ]
        )

        self.logger.save_dataframes_as_json(molecule_df, residue_df, self.output_file)

        with open(self.output_file, "r") as f:
            data = json.load(f)

        self.assertIn("molecule_data", data)
        self.assertIn("residue_data", data)
        self.assertEqual(data["molecule_data"][0]["Type"], "Transvibrational (J/mol/K)")
        self.assertEqual(data["residue_data"][0]["Residue"], 0)

    @patch("CodeEntropy.config.data_logger.logger")
    def test_log_tables(self, mock_logger):
        """
        Test that log_tables logs formatted molecule and residue tables using the
        logger.
        """
        self.logger.add_results_data(
            0, "united_atom", "Transvibrational (J/mol/K)", 653.404
        )
        self.logger.add_residue_data(0, 0, "Transvibrational (J/mol/K)", 122.612)

        self.logger.log_tables()

        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Molecule Data Table:" in c for c in calls))
        self.assertTrue(any("Residue Data Table:" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
