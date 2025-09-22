import json
import unittest

import numpy as np
import pandas as pd

from CodeEntropy.config.data_logger import DataLogger
from CodeEntropy.config.logging_config import LoggingConfig
from CodeEntropy.main import main
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestDataLogger(BaseTestCase):
    """
    Unit tests for the DataLogger class.
    """

    def setUp(self):
        super().setUp()
        self.code_entropy = main
        self.logger = DataLogger()
        self.output_file = "test_output.json"

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
            0, "united_atom", "Transvibrational", 653.4041220313459
        )
        self.assertEqual(
            self.logger.molecule_data,
            [(0, "united_atom", "Transvibrational", 653.4041220313459)],
        )

    def test_add_residue_data(self):
        """
        Test that add_residue_data correctly appends a residue-level entry.
        """
        self.logger.add_residue_data(
            0, "DA", "united_atom", "Transvibrational", 10, 122.61216935211893
        )
        self.assertEqual(
            self.logger.residue_data,
            [[0, "DA", "united_atom", "Transvibrational", 10, 122.61216935211893]],
        )

    def test_add_residue_data_with_numpy_array(self):
        """
        Test that add_residue_data correctly converts a NumPy array to a list.
        """
        frame_array = np.array([10])
        self.logger.add_residue_data(
            1, "DT", "united_atom", "Transvibrational", frame_array, 98.123456789
        )
        self.assertEqual(
            self.logger.residue_data,
            [[1, "DT", "united_atom", "Transvibrational", [10], 98.123456789]],
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

    def test_log_tables_rich_output(self):
        console = LoggingConfig.get_console()
        console.clear_live()

        self.logger.add_results_data(
            0, "united_atom", "Transvibrational", 653.4041220313459
        )
        self.logger.add_residue_data(
            0, "DA", "united_atom", "Transvibrational", 10, 122.61216935211893
        )
        self.logger.add_group_label(0, "DA", 10, 100)

        self.logger.log_tables()

        output = console.export_text()
        assert "Molecule Entropy Results" in output
        assert "Residue Entropy Results" in output
        assert "Group ID to Residue Label Mapping" in output


if __name__ == "__main__":
    unittest.main()
