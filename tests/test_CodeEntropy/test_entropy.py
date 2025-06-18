import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import MDAnalysis as mda
import numpy as np
import pytest

import tests.data as data
from CodeEntropy.config.data_logger import DataLogger
from CodeEntropy.entropy import (
    ConformationalEntropy,
    EntropyManager,
    OrientationalEntropy,
    VibrationalEntropy,
)
from CodeEntropy.levels import LevelManager
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

    def test_execute_full_workflow(self):
        """
        Tests that `execute` runs the full entropy workflow for a known system,
        triggering all processing branches and logging expected results.
        """
        # Load test universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        args = MagicMock(
            bin_width=0.1, temperature=300, selection_string="all", water_entropy=False
        )
        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()
        entropy_manager = EntropyManager(
            run_manager, args, u, data_logger, level_manager
        )

        entropy_manager._get_trajectory_bounds = MagicMock(return_value=(0, 10, 1))
        entropy_manager._get_number_frames = MagicMock(return_value=11)
        entropy_manager._get_reduced_universe = MagicMock(
            return_value="reduced_universe"
        )
        entropy_manager._get_molecule_container = MagicMock(
            return_value=MagicMock(residues=[1, 2, 3])
        )
        entropy_manager._finalize_molecule_results = MagicMock()
        entropy_manager._data_logger.log_tables = MagicMock()

        entropy_manager._level_manager.select_levels = MagicMock(
            return_value=(1, [["united_atom", "polymer", "residue"]])
        )

        # Patch entropy classes and processing methods
        ve = MagicMock()
        ce = MagicMock()
        with (
            patch("CodeEntropy.entropy.VibrationalEntropy", return_value=ve),
            patch("CodeEntropy.entropy.ConformationalEntropy", return_value=ce),
        ):

            entropy_manager._process_united_atom_level = MagicMock(
                side_effect=lambda *args, **kwargs: data_logger.add_results_data(
                    "A", "united_atom", "Conformational", 1.0
                )
            )
            entropy_manager._process_vibrational_only_levels = MagicMock(
                side_effect=lambda *args, **kwargs: data_logger.add_results_data(
                    "A", "polymer", "Transvibrational", 2.0
                )
            )
            entropy_manager._process_conformational_residue_level = MagicMock(
                side_effect=lambda *args, **kwargs: data_logger.add_residue_data(
                    0, "A", "residue", "Conformational", 3.0
                )
            )

            entropy_manager.execute()

        # Assertions
        entropy_manager._process_united_atom_level.assert_called_once()
        self.assertEqual(entropy_manager._process_vibrational_only_levels.call_count, 2)
        entropy_manager._process_conformational_residue_level.assert_called_once()
        entropy_manager._finalize_molecule_results.assert_called_once()
        entropy_manager._data_logger.log_tables.assert_called_once()

        # Check molecule-level entropy types
        molecule_types = set(entry[2] for entry in data_logger.molecule_data)
        self.assertIn("Conformational", molecule_types)
        self.assertIn("Transvibrational", molecule_types)

        # Check residue-level entropy types
        residue_types = set(entry[3] for entry in data_logger.residue_data)
        self.assertIn("Conformational", residue_types)

    def test_water_entropy_sets_selection_string_when_all(self):
        """
        Tests that when `selection_string` is initially 'all' and water entropy is
        enabled, the `execute` method sets `selection_string` to 'not water' after
        calculating water entropy.
        """
        mock_universe = MagicMock()
        mock_universe.select_atoms.return_value.n_atoms = 5

        args = MagicMock(water_entropy=True, selection_string="all")
        run_manager = MagicMock()
        level_manager = MagicMock()
        data_logger = DataLogger()

        manager = EntropyManager(
            run_manager, args, mock_universe, data_logger, level_manager
        )
        manager._get_trajectory_bounds = MagicMock(return_value=(0, 10, 1))
        manager._get_number_frames = MagicMock(return_value=11)
        manager._calculate_water_entropy = MagicMock()
        manager._get_reduced_universe = MagicMock(return_value="reduced")
        manager._level_manager.select_levels = MagicMock(return_value=(0, []))
        manager._finalize_molecule_results = MagicMock()
        manager._data_logger.log_tables = MagicMock()

        manager.execute()

        manager._calculate_water_entropy.assert_called_once()
        assert args.selection_string == "not water"

    def test_water_entropy_appends_to_custom_selection_string(self):
        """
        Tests that when `selection_string` is a custom value and water
        entropy is enabled, the `execute` method appends ' and not water'
        to the existing selection string after calculating water entropy.
        """
        mock_universe = MagicMock()
        mock_universe.select_atoms.return_value.n_atoms = 5

        args = MagicMock(water_entropy=True, selection_string="protein")
        run_manager = MagicMock()
        level_manager = MagicMock()
        data_logger = DataLogger()

        manager = EntropyManager(
            run_manager, args, mock_universe, data_logger, level_manager
        )
        manager._get_trajectory_bounds = MagicMock(return_value=(0, 10, 1))
        manager._get_number_frames = MagicMock(return_value=11)
        manager._calculate_water_entropy = MagicMock()
        manager._get_reduced_universe = MagicMock(return_value="reduced")
        manager._level_manager.select_levels = MagicMock(return_value=(0, []))
        manager._finalize_molecule_results = MagicMock()
        manager._data_logger.log_tables = MagicMock()

        manager.execute()

        manager._calculate_water_entropy.assert_called_once()
        assert args.selection_string == "protein and not water"

    def test_get_trajectory_bounds(self):
        """
        Tests that `_get_trajectory_bounds` runs and returns expected types.
        """

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
        """
        Test `_get_number_frames` when the end index is -1 (interpreted as no slicing).

        Ensures that the function returns 0 frames when the trajectory bounds
        result in an empty range.
        """
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
        """
        Test `_get_number_frames` with a valid slicing range.

        Verifies that the function correctly calculates the number of frames
        when slicing from 0 to 20 with a step of 1, expecting 21 frames.
        """
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
        """
        Test `_get_number_frames` with a step that skips all frames.

        Ensures that the function returns 0 when the step size results in
        no frames being selected from the trajectory.
        """

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
        """
        Test `_get_reduced_universe` with 'all' selection.

        Verifies that the full universe is returned when the selection string
        is set to 'all', and the number of atoms remains unchanged.
        """
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
        """
        Test `_get_reduced_universe` with a specific atom selection.

        Ensures that the reduced universe contains fewer atoms than the original
        when a specific selection string is used.
        """

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
        """
        Test `_get_molecule_container` for extracting a molecule fragment.

        Verifies that the returned universe contains the correct atoms corresponding
        to the specified molecule ID's fragment from the original universe.
        """

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

    def test_process_united_atom_level(self):
        """
        Tests that `_process_united_atom_level` correctly logs global and residue-level
        entropy results for a known molecular system using MDAnalysis.
        """

        # Load a known test universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        # Setup managers and arguments
        args = MagicMock(bin_width=0.1, temperature=300, selection_string="all")
        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()
        manager = EntropyManager(run_manager, args, u, data_logger, level_manager)

        reduced_atom = manager._get_reduced_universe()
        mol_container = manager._get_molecule_container(reduced_atom, 0)
        n_residues = len(mol_container.residues)

        ve = VibrationalEntropy(run_manager, args, u, data_logger, level_manager)
        ce = ConformationalEntropy(run_manager, args, u, data_logger, level_manager)

        # Run the function
        manager._process_united_atom_level(
            mol_id=0,
            mol_container=mol_container,
            ve=ve,
            ce=ce,
            level="united_atom",
            start=1,
            end=1,
            step=1,
            n_frames=1,
            highest=True,
        )

        # Check that results were logged for each entropy type
        df = data_logger.molecule_data
        self.assertEqual(len(df), 3)  # Trans, Rot, Conf

        # Check that residue-level results were logged
        residue_df = data_logger.residue_data
        self.assertEqual(len(residue_df), 3 * n_residues)  # 3 types per residue

        # Check that all expected types are present
        expected_types = {
            "Transvibrational",
            "Rovibrational",
            "Conformational",
        }

        actual_types = set(entry[2] for entry in df)
        self.assertSetEqual(actual_types, expected_types)

        residue_types = set(entry[3] for entry in residue_df)
        self.assertSetEqual(residue_types, expected_types)

    def test_process_vibrational_only_levels(self):
        """
        Tests that `_process_vibrational_only_levels` correctly logs vibrational
        entropy results for a known molecular system using MDAnalysis.
        """

        # Load a known test universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        # Setup managers and arguments
        args = MagicMock(bin_width=0.1, temperature=300, selection_string="all")
        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()
        manager = EntropyManager(run_manager, args, u, data_logger, level_manager)

        reduced_atom = manager._get_reduced_universe()
        mol_container = manager._get_molecule_container(reduced_atom, 0)

        # Patch methods to isolate the test
        manager._level_manager.get_matrices = MagicMock(
            return_value=("mock_force", "mock_torque")
        )

        ve = VibrationalEntropy(run_manager, args, u, data_logger, level_manager)
        ve.vibrational_entropy_calculation = MagicMock(side_effect=[1.11, 2.22])

        # Run the function
        manager._process_vibrational_only_levels(
            mol_id=0,
            mol_container=mol_container,
            ve=ve,
            level="Vibrational",
            start=1,
            end=1,
            step=1,
            n_frames=1,
            highest=True,
        )

        # Check that results were logged
        df = data_logger.molecule_data
        self.assertEqual(len(df), 2)  # Transvibrational and Rovibrational

        expected_types = {
            "Transvibrational",
            "Rovibrational",
        }
        actual_types = set(entry[2] for entry in df)
        self.assertSetEqual(actual_types, expected_types)

        # Check entropy values
        results = [entry[3] for entry in df]
        self.assertIn(1.11, results)
        self.assertIn(2.22, results)

    def test_process_conformational_residue_level(self):
        """
        Tests that `_process_conformational_residue_level` correctly logs conformational
        entropy results at the residue level for a known molecular system using
        MDAnalysis.
        """

        # Load a known test universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        # Setup managers and arguments
        args = MagicMock(bin_width=0.1, temperature=300, selection_string="all")
        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()
        manager = EntropyManager(run_manager, args, u, data_logger, level_manager)

        reduced_atom = manager._get_reduced_universe()
        mol_container = manager._get_molecule_container(reduced_atom, 0)

        # Patch methods to isolate the test
        mock_dihedrals = ["phi", "psi", "chi1"]
        manager._level_manager.get_dihedrals = MagicMock(return_value=mock_dihedrals)

        ce = ConformationalEntropy(run_manager, args, u, data_logger, level_manager)
        ce.conformational_entropy_calculation = MagicMock(return_value=3.33)

        # Run the function
        manager._process_conformational_residue_level(
            mol_id=0,
            mol_container=mol_container,
            ce=ce,
            level="residue",
            start=1,
            end=1,
            step=1,
            n_frames=1,
        )

        # Check that results were logged
        df = data_logger.molecule_data
        self.assertEqual(len(df), 1)

        expected_types = {"Conformational"}
        actual_types = set(entry[2] for entry in df)
        self.assertSetEqual(actual_types, expected_types)

        # Check entropy values
        results = [entry[3] for entry in df]
        self.assertIn(3.33, results)

    def test_finalize_molecule_results_aggregates_and_logs_total_entropy(self):
        """
        Tests that `_finalize_molecule_results` correctly aggregates entropy values per
        molecule from `molecule_data`, appends a 'Molecule Total' entry, and calls
        `save_dataframes_as_json` with the expected DataFrame structure.
        """
        # Setup
        args = MagicMock(output_file="mock_output.json")
        data_logger = DataLogger()
        data_logger.molecule_data = [
            ("mol1", "united_atom", "Transvibrational", 1.0),
            ("mol1", "united_atom", "Rovibrational", 2.0),
            ("mol1", "united_atom", "Conformational", 3.0),
            ("mol2", "polymer", "Transvibrational", 4.0),
        ]
        data_logger.residue_data = []

        manager = EntropyManager(None, args, None, data_logger, None)

        # Patch save method
        data_logger.save_dataframes_as_json = MagicMock()

        # Execute
        manager._finalize_molecule_results()

        # Check that totals were added
        totals = [
            entry for entry in data_logger.molecule_data if entry[1] == "Molecule Total"
        ]
        self.assertEqual(len(totals), 2)

        # Check correct aggregation
        mol1_total = next(entry for entry in totals if entry[0] == "mol1")[3]
        mol2_total = next(entry for entry in totals if entry[0] == "mol2")[3]
        self.assertEqual(mol1_total, 6.0)
        self.assertEqual(mol2_total, 4.0)

        # Check save was called
        data_logger.save_dataframes_as_json.assert_called_once()

    @patch("CodeEntropy.entropy.logger")
    def test_finalize_molecule_results_skips_invalid_entries(self, mock_logger):
        """
        Tests that `_finalize_molecule_results` skips entries with non-numeric entropy
        values and logs a warning without raising an exception.
        """
        args = MagicMock(output_file="mock_output.json")
        data_logger = DataLogger()
        data_logger.molecule_data = [
            ("mol1", "united_atom", "Transvibrational", 1.0),
            (
                "mol1",
                "united_atom",
                "Rovibrational",
                "not_a_number",
            ),  # Should trigger ValueError
            ("mol1", "united_atom", "Conformational", 2.0),
        ]
        data_logger.residue_data = []

        manager = EntropyManager(None, args, None, data_logger, None)

        # Patch save method
        data_logger.save_dataframes_as_json = MagicMock()

        # Run the method
        manager._finalize_molecule_results()

        # Check that only valid values were aggregated
        totals = [
            entry for entry in data_logger.molecule_data if entry[1] == "Molecule Total"
        ]
        self.assertEqual(len(totals), 1)
        self.assertEqual(totals[0][3], 3.0)  # 1.0 + 2.0

        # Check that a warning was logged
        mock_logger.warning.assert_called_once_with(
            "Skipping invalid entry: mol1, not_a_number"
        )


class TestVibrationalEntropy(unittest.TestCase):
    """
    Unit tests for the functionality of Vibrational entropy calculations.
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

    def test_vibrational_entropy_init(self):
        """
        Test initialization of the `VibrationalEntropy` class.

        Verifies that the object is correctly instantiated and that key arguments
        such as temperature and bin width are properly assigned.
        """
        # Mock dependencies
        universe = MagicMock()
        args = MagicMock()
        args.bin_width = 0.1
        args.temperature = 300
        args.selection_string = "all"

        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()

        # Instantiate VibrationalEntropy
        ve = VibrationalEntropy(run_manager, args, universe, data_logger, level_manager)

        # Basic assertions to check initialization
        self.assertIsInstance(ve, VibrationalEntropy)
        self.assertEqual(ve._args.temperature, 300)
        self.assertEqual(ve._args.bin_width, 0.1)

    # test when lambda is zero
    def test_frequency_calculation_0(self):
        """
        Test `frequency_calculation` with zero eigenvalue.

        Ensures that the method returns 0 when the input eigenvalue (lambda) is zero.
        """
        lambdas = 0
        temp = 298

        run_manager = RunManager("mock_folder")

        ve = VibrationalEntropy(
            run_manager, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        frequencies = ve.frequency_calculation(lambdas, temp)

        assert frequencies == 0

    def test_frequency_calculation_positive(self):
        """
        Test `frequency_calculation` with positive eigenvalues.

        Verifies that the method correctly computes frequencies from a set of
        positive eigenvalues at a given temperature.
        """
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

    def test_frequency_calculation_negative(self):
        """
        Test `frequency_calculation` with a negative eigenvalue.

        Ensures that the method raises a `ValueError` when any eigenvalue is negative,
        as this is physically invalid for frequency calculations.
        """
        lambdas = np.array([585495.0917897299, -658074.5130064893, 782425.305888707])
        temp = 298

        # Create a mock RunManager and set return value for get_KT2J
        run_manager = RunManager("temp_folder")
        run_manager.get_KT2J

        # Instantiate VibrationalEntropy with mocks
        ve = VibrationalEntropy(
            run_manager, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        # Assert that ValueError is raised due to negative eigenvalue
        with self.assertRaises(ValueError) as context:
            ve.frequency_calculation(lambdas, temp)

        self.assertIn("Negative eigenvalues", str(context.exception))

    def test_vibrational_entropy_calculation_force_not_highest(self):
        """
        Test `vibrational_entropy_calculation` for a force matrix with
        `highest_level=False`.

        Verifies that the entropy is correctly computed using mocked frequency values
        and a dummy identity matrix, excluding the first six modes.
        """
        # Mock RunManager
        run_manager = MagicMock()
        run_manager.change_lambda_units.return_value = np.array([1e-20] * 12)
        run_manager.get_KT2J.return_value = 2.47e-21

        # Instantiate VibrationalEntropy with mocks
        ve = VibrationalEntropy(
            run_manager, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        # Patch frequency_calculation to return known frequencies
        ve.frequency_calculation = MagicMock(return_value=np.array([1.0] * 12))

        # Create a dummy 12x12 matrix
        matrix = np.identity(12)

        # Run the method
        result = ve.vibrational_entropy_calculation(
            matrix=matrix, matrix_type="force", temp=298, highest_level=False
        )

        # Manually compute expected entropy components
        exponent = ve._PLANCK_CONST * 1.0 / 2.47e-21
        power_positive = np.exp(exponent)
        power_negative = np.exp(-exponent)
        S_component = exponent / (power_positive - 1) - np.log(1 - power_negative)
        S_component *= ve._GAS_CONST
        expected = S_component * 6  # sum of components[6:]

        self.assertAlmostEqual(result, expected, places=5)

    def test_vibrational_entropy_polymer_force(self):
        """
        Test `vibrational_entropy_calculation` with a real force matrix and
        `highest_level='yes'`.

        Ensures that the entropy is computed correctly for a small polymer system
        using a known force matrix and temperature.
        """
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

    def test_vibrational_entropy_polymer_torque(self):
        """
        Test `vibrational_entropy_calculation` with a torque matrix and
        `highest_level='yes'`.

        Verifies that the entropy is computed correctly for a torque matrix,
        simulating rotational degrees of freedom.
        """
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

    def test_confirmational_entropy_init(self):
        """
        Test initialization of the `ConformationalEntropy` class.

        Verifies that the object is correctly instantiated and that key arguments
        such as temperature and bin width are properly assigned during initialization.
        """
        # Mock dependencies
        universe = MagicMock()
        args = MagicMock()
        args.bin_width = 0.1
        args.temperature = 300
        args.selection_string = "all"

        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()

        # Instantiate ConformationalEntropy
        ce = ConformationalEntropy(
            run_manager, args, universe, data_logger, level_manager
        )

        # Basic assertions to check initialization
        self.assertIsInstance(ce, ConformationalEntropy)
        self.assertEqual(ce._args.temperature, 300)
        self.assertEqual(ce._args.bin_width, 0.1)

    def test_assign_conformation(self):
        """
        Test the `assign_conformation` method for correct binning of dihedral angles.

        Mocks a dihedral angle with specific values across frames and checks that:
        - The returned result is a NumPy array.
        - The array has the expected length.
        - All values are non-negative and of floating-point type.
        """
        # Mock dihedral with predefined values
        dihedral = MagicMock()
        dihedral.value = MagicMock(side_effect=[-30, 350, 350, 250, 10, 10])

        # Create a list of mock timesteps with frame numbers
        mock_timesteps = [MagicMock(frame=i) for i in range(6)]

        # Mock data_container with a trajectory that returns the mock timesteps
        data_container = MagicMock()
        data_container.trajectory.__getitem__.return_value = mock_timesteps

        # Load test universe
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")
        u = mda.Universe(tprfile, trrfile)

        # Setup managers and arguments
        args = MagicMock(bin_width=0.1, temperature=300, selection_string="all")
        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()

        ce = ConformationalEntropy(run_manager, args, u, data_logger, level_manager)

        result = ce.assign_conformation(
            data_container=data_container,
            dihedral=dihedral,
            number_frames=6,
            bin_width=60,
            start=0,
            end=6,
            step=1,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert np.all(result >= 0)
        assert np.issubdtype(result.dtype, np.floating)


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

    def test_orientational_entropy_init(self):
        """
        Test initialization of the `OrientationalEntropy` class.

        Verifies that the object is correctly instantiated and that key arguments
        such as temperature and bin width are properly assigned during initialization.
        """
        # Mock dependencies
        universe = MagicMock()
        args = MagicMock()
        args.bin_width = 0.1
        args.temperature = 300
        args.selection_string = "all"

        run_manager = RunManager("temp_folder")
        level_manager = LevelManager()
        data_logger = DataLogger()

        # Instantiate OrientationalEntropy
        oe = OrientationalEntropy(
            run_manager, args, universe, data_logger, level_manager
        )

        # Basic assertions to check initialization
        self.assertIsInstance(oe, OrientationalEntropy)
        self.assertEqual(oe._args.temperature, 300)
        self.assertEqual(oe._args.bin_width, 0.1)


if __name__ == "__main__":
    unittest.main()
