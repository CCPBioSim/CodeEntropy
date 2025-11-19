import logging
import os
from unittest.mock import MagicMock, patch

import MDAnalysis as mda
import numpy as np

import tests.data as data
from CodeEntropy.mda_universe_operations import UniverseOperations
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestUniverseOperations(BaseTestCase):
    """
    Unit tests for UniverseOperations.
    """

    def setUp(self):
        super().setUp()
        self.test_data_dir = os.path.dirname(data.__file__)

        # Disable MDAnalysis and commands file logging entirely
        logging.getLogger("MDAnalysis").handlers = [logging.NullHandler()]
        logging.getLogger("commands").handlers = [logging.NullHandler()]

    @patch("CodeEntropy.run.mda.analysis.base.AnalysisFromFunction")
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

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        # Set the side effects for the three AnalysisFromFunction calls
        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
        ]

        # Mock the merge operation
        mock_merged_universe = MagicMock()
        MockMerge.return_value = mock_merged_universe

        result = UniverseOperations.new_U_select_frame(mock_universe)

        mock_universe.select_atoms.assert_called_once_with("all", updating=True)
        MockMerge.assert_called_once_with(mock_select_atoms)

        # Ensure the 'load_new' method was called with the correct arguments
        mock_merged_universe.load_new.assert_called_once()
        args, kwargs = mock_merged_universe.load_new.call_args

        # Assert that the arrays are passed correctly
        np.testing.assert_array_equal(args[0], coords)
        np.testing.assert_array_equal(kwargs["forces"], forces)

        # Check if format was included in the kwargs
        self.assertIn("format", kwargs)

        # Ensure the result is the mock merged universe
        self.assertEqual(result, mock_merged_universe)

    @patch("CodeEntropy.run.mda.analysis.base.AnalysisFromFunction")
    @patch("CodeEntropy.run.mda.Merge")
    def test_new_U_select_atom(self, MockMerge, MockAnalysisFromFunction):
        # Mock Universe and its components
        mock_universe = MagicMock()
        mock_select_atoms = MagicMock()
        mock_universe.select_atoms.return_value = mock_select_atoms

        # Mock AnalysisFromFunction results for coordinates, forces, and dimensions
        coords = np.random.rand(10, 100, 3)
        forces = np.random.rand(10, 100, 3)

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        # Set the side effects for the three AnalysisFromFunction calls
        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
        ]

        # Mock the merge operation
        mock_merged_universe = MagicMock()
        MockMerge.return_value = mock_merged_universe

        result = UniverseOperations.new_U_select_atom(
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

        # Check if format was included in the kwargs
        self.assertIn("format", kwargs)

        # Ensure the result is the mock merged universe
        self.assertEqual(result, mock_merged_universe)

    def test_get_molecule_container(self):
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
        universe_operations = UniverseOperations()

        # Call the method
        molecule_id = 0
        mol_universe = universe_operations.get_molecule_container(u, molecule_id)

        # Get the original fragment
        original_fragment = u.atoms.fragments[molecule_id]

        # Assert that the atoms in the returned universe match the fragment
        selected_indices = mol_universe.indices
        #    selected_indices = mol_universe.atoms.indices
        expected_indices = original_fragment.indices

        assert set(selected_indices) == set(expected_indices)
        assert len(mol_universe.atoms) == len(original_fragment)
