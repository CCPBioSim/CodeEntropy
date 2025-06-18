import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from CodeEntropy.entropy import EntropyManager, VibrationalEntropy
from CodeEntropy.main import main
from CodeEntropy.run import RunManager


class TestVibrationalEntropy(unittest.TestCase):
    """
    Unit tests for the functionality of Vibrational entropy calculations.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.code_entropy = main

        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        self.entropy_manager = EntropyManager(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

    def tearDown(self):
        """
        Clean up after each test.
        """
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # test when lambda is zero
    def test_frequency_calculation_0(self):
        lambdas = 0
        temp = 298

        run_manager = RunManager("mock_folder")

        ve = VibrationalEntropy(
            run_manager, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        frequencies = ve.frequency_calculation(lambdas, temp)

        assert frequencies == 0

    # test when lambdas are positive
    def test_frequency_calculation_pos(self):
        lambdas = np.array([585495.0917897299, 658074.5130064893, 782425.305888707])
        temp = 298

        # Create a mock RunManager and set return value for get_KT2J
        run_manager = RunManager("mock_folder")

        # Instantiate VibrationalEntropy with mocks
        ve = VibrationalEntropy(
            run_manager, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        # Call the method under test
        frequencies = ve.frequency_calculation(lambdas, temp)

        assert frequencies == pytest.approx(
            [1899594266400.4016, 2013894687315.6213, 2195940987139.7097]
        )

    # TODO test for error handling when lambdas are negative

    # test for matrix_type force, highest level=yes
    def test_vibrational_entropy_polymer_force(self):
        matrix = np.array(
            [
                [4.67476, -0.04069, -0.19714],
                [-0.04069, 3.86300, -0.17922],
                [-0.19714, -0.17922, 3.66307],
            ]
        )
        matrix_type = "force"
        temp = 298
        highest_level = "yes"

        run_manager = RunManager("mock_folder")
        ve = VibrationalEntropy(
            run_manager, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        S_vib = ve.vibrational_entropy_calculation(
            matrix, matrix_type, temp, highest_level
        )

        assert S_vib == pytest.approx(52.88123410327823)

    # test for matrix_type force, highest level=no

    # test for matrix_type torque
    def test_vibrational_entropy_polymer_torque(self):
        matrix = np.array(
            [
                [6.69611, 0.39754, 0.57763],
                [0.39754, 4.63265, 0.38648],
                [0.57763, 0.38648, 6.34589],
            ]
        )
        matrix_type = "torque"
        temp = 298
        highest_level = "yes"

        run_manager = RunManager("mock_folder")
        ve = VibrationalEntropy(
            run_manager, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        S_vib = ve.vibrational_entropy_calculation(
            matrix, matrix_type, temp, highest_level
        )

        assert S_vib == pytest.approx(48.45003266069881)

    def test_calculate_water_orientational_entropy(self):
        """
        Test that orientational entropy values are correctly extracted from Sorient_dict
        and logged using add_residue_data.
        """
        Sorient_dict = {1: {"mol1": [1.0, 2]}, 2: {"mol1": [3.0, 4]}}

        self.entropy_manager._calculate_water_orientational_entropy(Sorient_dict)

        self.entropy_manager._data_logger.add_residue_data.assert_has_calls(
            [
                call(1, "mol1", "Water", "Orientational", 1.0),
                call(2, "mol1", "Water", "Orientational", 3.0),
            ]
        )

    def test_calculate_water_vibrational_translational_entropy(self):
        """
        Test that translational vibrational entropy values are correctly summed
        and logged per residue using add_residue_data. Also verifies that the
        molecule-level average is computed and logged using _log_result.
        """
        mock_vibrations = MagicMock()
        mock_vibrations.translational_S = {
            ("res1", "mol1"): [1.0, 2.0],
            ("resB_invalid", "mol1"): 4.0,
            ("res2", "mol1"): 3.0,
        }

        self.entropy_manager._calculate_water_vibrational_translational_entropy(
            mock_vibrations
        )

        self.entropy_manager._data_logger.add_residue_data.assert_has_calls(
            [
                call(-1, "res1", "Water", "Transvibrational", 3.0),
                call(-1, "resB", "Water", "Transvibrational", 4.0),
                call(-1, "res2", "Water", "Transvibrational", 3.0),
            ]
        )

    def test_empty_vibrational_entropy_dicts(self):
        """
        Test that no logging occurs when both translational and rotational
        entropy dictionaries are empty. Ensures that the methods handle empty
        input gracefully without errors or unnecessary logging.
        """
        self.entropy_manager._log_residue_data = MagicMock()
        self.entropy_manager._log_result = MagicMock()

        mock_vibrations = MagicMock()
        mock_vibrations.translational_S = {}
        mock_vibrations.rotational_S = {}

        self.entropy_manager._calculate_water_vibrational_translational_entropy(
            mock_vibrations
        )
        self.entropy_manager._calculate_water_vibrational_rotational_entropy(
            mock_vibrations
        )

        self.entropy_manager._log_residue_data.assert_not_called()
        self.entropy_manager._log_result.assert_not_called()

    def test_calculate_water_vibrational_rotational_entropy(self):
        """
        Test that rotational vibrational entropy values are correctly summed
        and logged per residue using add_residue_data. Also verifies that the
        residue ID parsing handles both valid and invalid formats.
        """
        mock_vibrations = MagicMock()
        mock_vibrations.rotational_S = {
            ("resA_101", "mol1"): [2.0, 3.0],
            ("resB_invalid", "mol1"): 4.0,
            ("resC", "mol1"): 5.0,
        }

        self.entropy_manager._calculate_water_vibrational_rotational_entropy(
            mock_vibrations
        )

        self.entropy_manager._data_logger.add_residue_data.assert_has_calls(
            [
                call(101, "resA", "Water", "Rovibrational", 5.0),
                call(-1, "resB", "Water", "Rovibrational", 4.0),
                call(-1, "resC", "Water", "Rovibrational", 5.0),
            ]
        )

    @patch(
        "waterEntropy.recipes.interfacial_solvent.get_interfacial_water_orient_entropy"
    )
    def test_calculate_water_entropy(self, mock_get_entropy):
        """
        Integration-style test that verifies _calculate_water_entropy correctly
        delegates to the orientational and vibrational entropy methods and logs
        the expected values.
        """
        mock_vibrations = MagicMock()
        mock_vibrations.translational_S = {("res1", "mol1"): 2.0}
        mock_vibrations.rotational_S = {("res1", "mol1"): 3.0}

        mock_get_entropy.return_value = (
            {1: {"mol1": [1.0, 5]}},
            None,
            mock_vibrations,
            None,
        )

        mock_universe = MagicMock()
        self.entropy_manager._calculate_water_entropy(mock_universe, 0, 10, 1)

        self.entropy_manager._data_logger.add_residue_data.assert_has_calls(
            [
                call(1, "mol1", "Water", "Orientational", 1.0),
                call(-1, "res1", "Water", "Transvibrational", 2.0),
                call(-1, "res1", "Water", "Rovibrational", 3.0),
            ]
        )

    @patch(
        "waterEntropy.recipes.interfacial_solvent.get_interfacial_water_orient_entropy"
    )
    def test_calculate_water_entropy_minimal(self, mock_get_entropy):
        """
        Verifies that _calculate_water_entropy correctly logs entropy components
        and total for a single molecule with minimal data.
        """
        mock_get_entropy.return_value = (
            {},
            None,
            MagicMock(
                translational_S={("ACE_1", "WAT"): 10.0},
                rotational_S={("ACE_1", "WAT"): 2.0},
            ),
            None,
        )

        # Simulate residue-level results already collected
        self.entropy_manager._data_logger.residue_data = [
            [1, "ACE", "Water", "Orientational", 5.0],
            [1, "ACE_1", "Water", "Transvibrational", 10.0],
            [1, "ACE_1", "Water", "Rovibrational", 2.0],
        ]

        mock_universe = MagicMock()
        self.entropy_manager._calculate_water_entropy(mock_universe, 0, 10, 1)

        self.entropy_manager._data_logger.add_results_data.assert_has_calls(
            [
                call("ACE", "water", "Orientational", 5.0),
                call("ACE", "water", "Transvibrational", 0.0),
                call("ACE", "water", "Rovibrational", 0.0),
                call("ACE_1", "water", "Orientational", 0.0),
                call("ACE_1", "water", "Transvibrational", 10.0),
                call("ACE_1", "water", "Rovibrational", 2.0),
            ]
        )

    # TODO test for error handling on invalid inputs


class TestConformationalEntropy(unittest.TestCase):
    """
    Unit tests for the functionality of conformational entropy calculations.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.code_entropy = main

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


class TestOrientationalEntropy(unittest.TestCase):
    """
    Unit tests for the functionality of orientational entropy calculations.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.code_entropy = main

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


if __name__ == "__main__":
    unittest.main()
