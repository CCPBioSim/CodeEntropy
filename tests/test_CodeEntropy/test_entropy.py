import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest

import tests.data as data
from CodeEntropy.entropy import EntropyManager, VibrationalEntropy

# from CodeEntropy.levels import LevelManager
from CodeEntropy.main import main
from CodeEntropy.run import ConfigManager, RunManager


class TestEntropyManager(unittest.TestCase):
    """
    Unit tests for the functionality of EntropyManager.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.test_data_dir = os.path.dirname(data.__file__)
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

    def test_results_df_property(self):
        """ """
        entropy_manager = EntropyManager(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        # Access the property
        df = entropy_manager.results_df

        # Check that it's a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Check that it has the correct columns
        expected_columns = ["Molecule ID", "Level", "Type", "Result"]
        self.assertListEqual(list(df.columns), expected_columns)

        # Check that it's initially empty
        self.assertTrue(df.empty)

    def test_residue_results_df(self):
        """ """
        entropy_manager = EntropyManager(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        # Access the property
        df = entropy_manager.residue_results_df

        # Check that it's a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Check that it has the correct columns
        expected_columns = ["Molecule ID", "Residue", "Type", "Result"]
        self.assertListEqual(list(df.columns), expected_columns)

        # Check that it's initially empty
        self.assertTrue(df.empty)

    def test_get_trajectory_bounds(self):
        """"""

        config_manager = ConfigManager()

        parser = config_manager.setup_argparse()
        args, _ = parser.parse_known_args()

        entropy_manager = EntropyManager(
            MagicMock(), args, MagicMock(), MagicMock(), MagicMock()
        )

        self.assertIsInstance(entropy_manager._args.start, int)
        self.assertIsInstance(entropy_manager._args.end, int)
        self.assertIsInstance(entropy_manager._args.step, int)

        self.assertEqual(entropy_manager._get_trajectory_bounds(), (0, -1, 1))

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            start=0,
            end=-1,
            step=1,
        ),
    )
    def test_get_number_frames(self, mock_args):
        """"""

        config_manager = ConfigManager()

        parser = config_manager.setup_argparse()
        args = parser.parse_args()

        entropy_manager = EntropyManager(
            MagicMock(), args, MagicMock(), MagicMock(), MagicMock()
        )
        entropy_manager._get_trajectory_bounds()
        number_frames = entropy_manager._get_number_frames(
            entropy_manager._args.start,
            entropy_manager._args.end,
            entropy_manager._args.step,
        )

        self.assertEqual(number_frames, 0)

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            start=0,
            end=20,
            step=1,
        ),
    )
    def test_get_number_frames_sliced_trajectory(self, mock_args):
        """"""

        config_manager = ConfigManager()

        parser = config_manager.setup_argparse()
        args = parser.parse_args()

        entropy_manager = EntropyManager(
            MagicMock(), args, MagicMock(), MagicMock(), MagicMock()
        )
        entropy_manager._get_trajectory_bounds()
        number_frames = entropy_manager._get_number_frames(
            entropy_manager._args.start,
            entropy_manager._args.end,
            entropy_manager._args.step,
        )

        self.assertEqual(number_frames, 21)

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            start=0,
            end=-1,
            step=5,
        ),
    )
    def test_get_number_frames_sliced_trajectory_step(self, mock_args):
        """"""

        config_manager = ConfigManager()

        parser = config_manager.setup_argparse()
        args = parser.parse_args()

        entropy_manager = EntropyManager(
            MagicMock(), args, MagicMock(), MagicMock(), MagicMock()
        )
        entropy_manager._get_trajectory_bounds()
        number_frames = entropy_manager._get_number_frames(
            entropy_manager._args.start,
            entropy_manager._args.end,
            entropy_manager._args.step,
        )

        self.assertEqual(number_frames, 0)

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            selection_string="all",
        ),
    )
    def test_get_reduced_universe_all(self, mock_args):
        """"""

        # Load MDAnalysis Universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        config_manager = ConfigManager()

        parser = config_manager.setup_argparse()
        args = parser.parse_args()

        entropy_manager = EntropyManager(MagicMock(), args, u, MagicMock(), MagicMock())

        entropy_manager._get_reduced_universe()

        self.assertEqual(entropy_manager._universe.atoms.n_atoms, 254)

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            selection_string="resname DA",
        ),
    )
    def test_get_reduced_universe_reduced(self, mock_args):
        """"""

        # Load MDAnalysis Universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        config_manager = ConfigManager()
        run_manager = RunManager("temp_folder")

        parser = config_manager.setup_argparse()
        args = parser.parse_args()

        entropy_manager = EntropyManager(run_manager, args, u, MagicMock(), MagicMock())

        reduced_u = entropy_manager._get_reduced_universe()

        # Assert that the reduced universe has fewer atoms
        assert len(reduced_u.atoms) < len(u.atoms)

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(
            selection_string="all",
        ),
    )
    def test_get_molecule_container(self, mock_args):
        """"""

        # Load a test universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        # Assume the universe has at least one fragment
        assert len(u.atoms.fragments) > 0

        # Setup managers
        config_manager = ConfigManager()
        run_manager = RunManager("temp_folder")

        parser = config_manager.setup_argparse()
        args = parser.parse_args()

        entropy_manager = EntropyManager(run_manager, args, u, MagicMock(), MagicMock())

        # Call the method
        molecule_id = 0
        mol_universe = entropy_manager._get_molecule_container(u, molecule_id)

        # Get the original fragment
        original_fragment = u.atoms.fragments[molecule_id]

        # Assert that the atoms in the returned universe match the fragment
        selected_indices = mol_universe.atoms.indices
        expected_indices = original_fragment.indices

        assert set(selected_indices) == set(expected_indices)
        assert len(mol_universe.atoms) == len(original_fragment)


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
