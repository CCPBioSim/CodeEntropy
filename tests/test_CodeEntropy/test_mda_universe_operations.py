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

    @patch("CodeEntropy.mda_universe_operations.AnalysisFromFunction")
    @patch("CodeEntropy.mda_universe_operations.mda.Merge")
    def test_new_U_select_frame(self, MockMerge, MockAnalysisFromFunction):
        """
        Unit test for UniverseOperations.new_U_select_frame().

        Verifies that:
        - The Universe is queried with select_atoms("all", updating=True)
        - AnalysisFromFunction is called to obtain coordinates and forces
        - mda.Merge is called with the selected AtomGroup
        - The new universe returned by Merge.load_new receives the correct arrays
        - The method returns the merged universe
        """
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

        ops = UniverseOperations()
        result = ops.new_U_select_frame(mock_universe)

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

    @patch("CodeEntropy.mda_universe_operations.AnalysisFromFunction")
    @patch("CodeEntropy.mda_universe_operations.mda.Merge")
    def test_new_U_select_atom(self, MockMerge, MockAnalysisFromFunction):
        """
        Unit test for UniverseOperations.new_U_select_atom().

        Ensures that:
        - The Universe is queried with the correct selection string
        - Coordinates and forces are extracted via AnalysisFromFunction
        - mda.Merge receives the AtomGroup from select_atoms
        - The new universe is populated with the expected data via load_new()
        - The returned universe is the object created by Merge
        """
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

        ops = UniverseOperations()

        result = ops.new_U_select_atom(mock_universe, select_string="resid 1-10")

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
        Integration test for UniverseOperations.get_molecule_container().

        Uses a real MDAnalysis Universe loaded from test trajectory files.
        Confirms that:
        - The correct fragment for a given molecule index is selected
        - The returned reduced Universe contains exactly the expected atom indices
        - The number of atoms matches the original fragment
        """
        tprfile = os.path.join(self.test_data_dir, "md_A4_dna.tpr")
        trrfile = os.path.join(self.test_data_dir, "md_A4_dna_xf.trr")

        u = mda.Universe(tprfile, trrfile)

        ops = UniverseOperations()

        molecule_id = 0

        fragment = u.atoms.fragments[molecule_id]
        expected_indices = fragment.indices

        mol_u = ops.get_molecule_container(u, molecule_id)

        selected_indices = mol_u.atoms.indices

        self.assertSetEqual(set(selected_indices), set(expected_indices))
        self.assertEqual(len(selected_indices), len(expected_indices))

    @patch("CodeEntropy.mda_universe_operations.AnalysisFromFunction")
    @patch("CodeEntropy.mda_universe_operations.mda.Merge")
    @patch("CodeEntropy.mda_universe_operations.mda.Universe")
    def test_merge_forces(self, MockUniverse, MockMerge, MockAnalysisFromFunction):
        """
        Unit test for UniverseOperations.merge_forces().

        This test ensures that:
        - Two MDAnalysis Universes are created: one for coordinates
        (tprfile + trrfile) and one for forces (tprfile + forcefile).
        - Both Universes correctly return AtomGroups via select_atoms("all").
        - Coordinates and forces are extracted using AnalysisFromFunction.
        - mda.Merge is called with the coordinate AtomGroup.
        - The merged Universe receives the correct coordinate and force arrays
        through load_new().
        - When kcal=False, force values are passed through unchanged
        (no kcal→kJ conversion).
        - The returned universe is the same object returned by mda.Merge().
        """

        mock_u_coords = MagicMock()
        mock_u_force = MagicMock()
        MockUniverse.side_effect = [mock_u_coords, mock_u_force]

        # Each universe returns a mock AtomGroup from select_atoms("all")
        mock_ag_coords = MagicMock()
        mock_ag_force = MagicMock()
        mock_u_coords.select_atoms.return_value = mock_ag_coords
        mock_u_force.select_atoms.return_value = mock_ag_force

        coords = np.random.rand(5, 10, 3)
        forces = np.random.rand(5, 10, 3)

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        # Two calls: first for coordinates, second for forces
        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
        ]

        mock_merged = MagicMock()
        MockMerge.return_value = mock_merged

        ops = UniverseOperations()
        result = ops.merge_forces(
            tprfile="topol.tpr",
            trrfile="traj.trr",
            forcefile="forces.trr",
            fileformat=None,
            kcal=False,
        )

        self.assertEqual(MockUniverse.call_count, 2)
        MockUniverse.assert_any_call("topol.tpr", "traj.trr", format=None)
        MockUniverse.assert_any_call("topol.tpr", "forces.trr", format=None)

        mock_u_coords.select_atoms.assert_called_once_with("all")
        mock_u_force.select_atoms.assert_called_once_with("all")

        self.assertEqual(MockAnalysisFromFunction.call_count, 2)

        MockMerge.assert_called_once_with(mock_ag_coords)

        mock_merged.load_new.assert_called_once()
        args, kwargs = mock_merged.load_new.call_args

        # Coordinates passed positionally
        np.testing.assert_array_equal(args[0], coords)

        # Forces passed via kwargs
        np.testing.assert_array_equal(kwargs["forces"], forces)

        # Finally the function returns the merged universe
        self.assertEqual(result, mock_merged)

    @patch("CodeEntropy.mda_universe_operations.AnalysisFromFunction")
    @patch("CodeEntropy.mda_universe_operations.mda.Merge")
    @patch("CodeEntropy.mda_universe_operations.mda.Universe")
    def test_merge_forces_kcal_conversion(
        self, MockUniverse, MockMerge, MockAnalysisFromFunction
    ):
        """
        Unit test for UniverseOperations.merge_forces() covering the kcal→kJ
        conversion branch.

        Verifies that:
        - Two Universe objects are constructed for coords and forces.
        - Each Universe returns an AtomGroup via select_atoms("all").
        - AnalysisFromFunction is called twice.
        - Forces are multiplied EXACTLY once by 4.184 when kcal=True.
        - Merge() is called with the coordinate AtomGroup.
        - load_new() receives the correct coordinates and converted forces.
        - The returned Universe is the Merge() result.
        """
        mock_u_coords = MagicMock()
        mock_u_force = MagicMock()
        MockUniverse.side_effect = [mock_u_coords, mock_u_force]

        mock_ag_coords = MagicMock()
        mock_ag_force = MagicMock()
        mock_u_coords.select_atoms.return_value = mock_ag_coords
        mock_u_force.select_atoms.return_value = mock_ag_force

        coords = np.ones((2, 3, 3))

        original_forces = np.ones((2, 3, 3))
        mock_forces_array = original_forces.copy()

        # Mock AnalysisFromFunction return values
        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {
            "timeseries": mock_forces_array
        }

        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
        ]

        mock_merged = MagicMock()
        MockMerge.return_value = mock_merged

        ops = UniverseOperations()
        result = ops.merge_forces("t.tpr", "c.trr", "f.trr", kcal=True)

        _, kwargs = mock_merged.load_new.call_args

        expected_forces = original_forces * 4.184
        np.testing.assert_array_equal(kwargs["forces"], expected_forces)
        np.testing.assert_array_equal(mock_merged.load_new.call_args[0][0], coords)

        self.assertEqual(result, mock_merged)
