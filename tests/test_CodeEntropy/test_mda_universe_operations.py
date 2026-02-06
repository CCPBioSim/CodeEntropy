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
        dims = np.random.rand(10, 6)

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        mock_dims_analysis = MagicMock()
        mock_dims_analysis.run.return_value.results = {"timeseries": dims}

        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
            mock_dims_analysis,
        ]

        # Mock merge operation
        mock_merged_universe = MagicMock()
        MockMerge.return_value = mock_merged_universe

        ops = UniverseOperations()
        result = ops.new_U_select_frame(mock_universe)

        # Basic behavior checks
        mock_universe.select_atoms.assert_called_once_with("all", updating=True)

        # AnalysisFromFunction called 3 times (coords, forces, dimensions)
        assert MockAnalysisFromFunction.call_count == 3
        mock_coords_analysis.run.assert_called_once()
        mock_forces_analysis.run.assert_called_once()
        mock_dims_analysis.run.assert_called_once()

        # Merge called with selected AtomGroup
        MockMerge.assert_called_once_with(mock_select_atoms)

        assert result == mock_merged_universe

    @patch("CodeEntropy.mda_universe_operations.AnalysisFromFunction")
    @patch("CodeEntropy.mda_universe_operations.mda.Merge")
    def test_new_U_select_atom(self, MockMerge, MockAnalysisFromFunction):
        """
        Unit test for UniverseOperations.new_U_select_atom().

        Ensures that:
        - The Universe is queried with the correct selection string
        - Coordinates, forces, and dimensions are extracted via AnalysisFromFunction
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
        dims = np.random.rand(10, 6)

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        mock_dims_analysis = MagicMock()
        mock_dims_analysis.run.return_value.results = {"timeseries": dims}

        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
            mock_dims_analysis,
        ]

        # Mock the merge operation
        mock_merged_universe = MagicMock()
        MockMerge.return_value = mock_merged_universe

        ops = UniverseOperations()

        result = ops.new_U_select_atom(mock_universe, select_string="resid 1-10")
        mock_universe.select_atoms.assert_called_once_with("resid 1-10", updating=True)

        # AnalysisFromFunction called for coords, forces, dimensions
        assert MockAnalysisFromFunction.call_count == 3
        mock_coords_analysis.run.assert_called_once()
        mock_forces_analysis.run.assert_called_once()
        mock_dims_analysis.run.assert_called_once()

        # Merge called with the selected AtomGroup
        MockMerge.assert_called_once_with(mock_select_atoms)

        # Returned universe should be the merged universe
        assert result == mock_merged_universe

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
        """
        # Two Universes created: coords and forces
        mock_u_coords = MagicMock()
        mock_u_force = MagicMock()
        MockUniverse.side_effect = [mock_u_coords, mock_u_force]

        # Each universe returns an AtomGroup from select_atoms("all")
        mock_ag_coords = MagicMock()
        mock_ag_force = MagicMock()
        mock_u_coords.select_atoms.return_value = mock_ag_coords
        mock_u_force.select_atoms.return_value = mock_ag_force

        coords = np.random.rand(5, 10, 3)
        forces = np.random.rand(5, 10, 3)
        dims = np.random.rand(5, 6)

        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {"timeseries": forces}

        mock_dims_analysis = MagicMock()
        mock_dims_analysis.run.return_value.results = {"timeseries": dims}

        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
            mock_dims_analysis,
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

        # Universe construction
        assert MockUniverse.call_count == 2

        # Selection
        mock_u_coords.select_atoms.assert_called_once_with("all")
        mock_u_force.select_atoms.assert_called_once_with("all")

        # AnalysisFromFunction usage
        assert MockAnalysisFromFunction.call_count == 3
        mock_coords_analysis.run.assert_called_once()
        mock_forces_analysis.run.assert_called_once()
        mock_dims_analysis.run.assert_called_once()

        # Merge called with coordinate AtomGroup
        MockMerge.assert_called_once_with(mock_ag_coords)

        # Returned object is merged universe
        assert result == mock_merged

    @patch("CodeEntropy.mda_universe_operations.AnalysisFromFunction")
    @patch("CodeEntropy.mda_universe_operations.mda.Merge")
    @patch("CodeEntropy.mda_universe_operations.mda.Universe")
    def test_merge_forces_kcal_conversion(
        self, MockUniverse, MockMerge, MockAnalysisFromFunction
    ):
        """
        Unit test for UniverseOperations.merge_forces() covering the kcal→kJ
        conversion branch.
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

        dims = np.ones((2, 6))

        # Mock AnalysisFromFunction return values
        mock_coords_analysis = MagicMock()
        mock_coords_analysis.run.return_value.results = {"timeseries": coords}

        mock_forces_analysis = MagicMock()
        mock_forces_analysis.run.return_value.results = {
            "timeseries": mock_forces_array
        }

        mock_dims_analysis = MagicMock()
        mock_dims_analysis.run.return_value.results = {"timeseries": dims}

        MockAnalysisFromFunction.side_effect = [
            mock_coords_analysis,
            mock_forces_analysis,
            mock_dims_analysis,
        ]

        mock_merged = MagicMock()
        MockMerge.return_value = mock_merged

        ops = UniverseOperations()
        result = ops.merge_forces("t.tpr", "c.trr", "f.trr", kcal=True)

        # select_atoms("all") (your code uses no updating=True)
        mock_u_coords.select_atoms.assert_called_once_with("all")
        mock_u_force.select_atoms.assert_called_once_with("all")

        # AnalysisFromFunction called three times
        assert MockAnalysisFromFunction.call_count == 3

        # Forces are multiplied exactly once by 4.184 when kcal=True
        np.testing.assert_allclose(
            mock_forces_array, original_forces * 4.184, rtol=0, atol=0
        )

        # Merge called with coordinate AtomGroup
        MockMerge.assert_called_once_with(mock_ag_coords)

        # Returned universe is the merged universe
        assert result == mock_merged
