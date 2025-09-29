from unittest.mock import MagicMock, patch

import numpy as np

from CodeEntropy.levels import LevelManager
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestLevels(BaseTestCase):
    """
    Unit tests for Levels.
    """

    def setUp(self):
        super().setUp()

    def test_select_levels(self):
        """
        Test `select_levels` with a mocked data container containing two molecules:
        - The first molecule has 2 atoms and 1 residue (should return 'united_atom' and
        'residue').
        - The second molecule has 3 atoms and 2 residues (should return all three
        levels).

        Asserts that the number of molecules and the levels list match expected values.
        """
        # Create a mock data_container
        data_container = MagicMock()

        # Mock fragments (2 molecules)
        fragment1 = MagicMock()
        fragment2 = MagicMock()

        # Mock select_atoms return values
        atoms1 = MagicMock()
        atoms1.__len__.return_value = 2
        atoms1.residues = [1]  # 1 residue

        atoms2 = MagicMock()
        atoms2.__len__.return_value = 3
        atoms2.residues = [1, 2]  # 2 residues

        fragment1.select_atoms.return_value = atoms1
        fragment2.select_atoms.return_value = atoms2

        data_container.atoms.fragments = [fragment1, fragment2]

        # Import the class and call the method
        level_manager = LevelManager()
        number_molecules, levels = level_manager.select_levels(data_container)

        # Assertions
        self.assertEqual(number_molecules, 2)
        self.assertEqual(
            levels, [["united_atom", "residue"], ["united_atom", "residue", "polymer"]]
        )

    def test_get_matrices(self):
        """
        Test `get_matrices` with mocked internal methods and a simple trajectory.
        Ensures that the method returns correctly shaped matrices after filtering.
        """

        # Create a mock LevelManager level_manager
        level_manager = LevelManager()

        # Mock internal methods
        level_manager.get_beads = MagicMock(return_value=["bead1", "bead2"])
        level_manager.get_axes = MagicMock(return_value=("trans_axes", "rot_axes"))
        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )
        level_manager.create_submatrix = MagicMock(return_value=np.identity(3))
        level_manager.filter_zero_rows_columns = MagicMock(side_effect=lambda x: x)

        # Mock data_container and trajectory
        data_container = MagicMock()
        timestep1 = MagicMock()
        timestep1.frame = 0
        timestep2 = MagicMock()
        timestep2.frame = 1
        data_container.trajectory.__getitem__.return_value = [timestep1, timestep2]

        # Call the method
        force_matrix, torque_matrix = level_manager.get_matrices(
            data_container=data_container,
            level="residue",
            number_frames=2,
            highest_level=True,
            force_matrix=None,
            torque_matrix=None,
        )

        # Assertions
        self.assertTrue(isinstance(force_matrix, np.ndarray))
        self.assertTrue(isinstance(torque_matrix, np.ndarray))
        self.assertEqual(force_matrix.shape, (6, 6))  # 2 beads × 3D
        self.assertEqual(torque_matrix.shape, (6, 6))

        # Check that internal methods were called
        self.assertEqual(level_manager.get_beads.call_count, 1)
        self.assertEqual(level_manager.get_axes.call_count, 2)  # 2 beads
        self.assertEqual(
            level_manager.create_submatrix.call_count, 6
        )  # 3 force + 3 torque

    def test_get_matrices_force_shape_mismatch(self):
        """
        Test that get_matrices raises a ValueError when the provided force_matrix
        has a shape mismatch with the computed force block matrix.
        """
        level_manager = LevelManager()

        # Mock internal methods
        level_manager.get_beads = MagicMock(return_value=["bead1", "bead2"])
        level_manager.get_axes = MagicMock(return_value=("trans_axes", "rot_axes"))
        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )
        level_manager.create_submatrix = MagicMock(return_value=np.identity(3))

        data_container = MagicMock()

        # Incorrect shape for force matrix (should be 6x6 for 2 beads)
        bad_force_matrix = np.zeros((3, 3))
        correct_torque_matrix = np.zeros((6, 6))

        with self.assertRaises(ValueError) as context:
            level_manager.get_matrices(
                data_container=data_container,
                level="residue",
                number_frames=2,
                highest_level=True,
                force_matrix=bad_force_matrix,
                torque_matrix=correct_torque_matrix,
            )

        self.assertIn("Inconsistent force matrix shape", str(context.exception))

    def test_get_matrices_torque_shape_mismatch(self):
        """
        Test that get_matrices raises a ValueError when the provided torque_matrix
        has a shape mismatch with the computed torque block matrix.
        """
        level_manager = LevelManager()

        # Mock internal methods
        level_manager.get_beads = MagicMock(return_value=["bead1", "bead2"])
        level_manager.get_axes = MagicMock(return_value=("trans_axes", "rot_axes"))
        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )
        level_manager.create_submatrix = MagicMock(return_value=np.identity(3))

        data_container = MagicMock()

        correct_force_matrix = np.zeros((6, 6))
        bad_torque_matrix = np.zeros((3, 3))  # Incorrect shape

        with self.assertRaises(ValueError) as context:
            level_manager.get_matrices(
                data_container=data_container,
                level="residue",
                number_frames=2,
                highest_level=True,
                force_matrix=correct_force_matrix,
                torque_matrix=bad_torque_matrix,
            )

        self.assertIn("Inconsistent torque matrix shape", str(context.exception))

    def test_get_matrices_torque_consistency(self):
        """
        Test that get_matrices returns consistent torque and force matrices
        when called multiple times with the same inputs.
        """
        level_manager = LevelManager()

        level_manager.get_beads = MagicMock(return_value=["bead1", "bead2"])
        level_manager.get_axes = MagicMock(return_value=("trans_axes", "rot_axes"))
        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )
        level_manager.create_submatrix = MagicMock(return_value=np.identity(3))

        data_container = MagicMock()

        initial_force_matrix = np.zeros((6, 6))
        initial_torque_matrix = np.zeros((6, 6))

        force_matrix_1, torque_matrix_1 = level_manager.get_matrices(
            data_container=data_container,
            level="residue",
            number_frames=2,
            highest_level=True,
            force_matrix=initial_force_matrix.copy(),
            torque_matrix=initial_torque_matrix.copy(),
        )

        force_matrix_2, torque_matrix_2 = level_manager.get_matrices(
            data_container=data_container,
            level="residue",
            number_frames=2,
            highest_level=True,
            force_matrix=initial_force_matrix.copy(),
            torque_matrix=initial_torque_matrix.copy(),
        )

        # Check that repeated calls produce the same output
        self.assertTrue(np.allclose(torque_matrix_1, torque_matrix_2, atol=1e-8))
        self.assertTrue(np.allclose(force_matrix_1, force_matrix_2, atol=1e-8))

    def test_get_dihedrals_united_atom(self):
        """
        Test `get_dihedrals` for 'united_atom' level.
        Ensures it returns the dihedrals directly from the data container.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        mock_dihedrals = ["d1", "d2", "d3"]
        data_container.dihedrals = mock_dihedrals

        result = level_manager.get_dihedrals(data_container, level="united_atom")
        self.assertEqual(result, mock_dihedrals)

    def test_get_dihedrals_residue(self):
        """
        Test `get_dihedrals` for 'residue' level with 5 residues.
        Mocks bonded atom selections and verifies that dihedrals are constructed.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        data_container.residues = [0, 1, 2, 3, 4]  # 5 residues

        # Mock select_atoms to return atom groups with .dihedral
        mock_dihedral = MagicMock()
        mock_atom_group = MagicMock()
        mock_atom_group.__add__.return_value = mock_atom_group
        mock_atom_group.dihedral = mock_dihedral
        data_container.select_atoms.return_value = mock_atom_group

        result = level_manager.get_dihedrals(data_container, level="residue")

        # Should create 2 dihedrals for 5 residues (residues 0–3 and 1–4)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(d == mock_dihedral for d in result))

    def test_get_dihedrals_no_residue(self):
        """
        Test `get_dihedrals` for 'residue' level with 3 residues.
        Mocks bonded atom selections and verifies that dihedrals are constructed.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        data_container.residues = [0, 1, 2]  # 3 residues

        # Mock select_atoms to return atom groups with .dihedral
        mock_dihedral = MagicMock()
        mock_atom_group = MagicMock()
        mock_atom_group.__add__.return_value = mock_atom_group
        mock_atom_group.dihedral = mock_dihedral
        data_container.select_atoms.return_value = mock_atom_group

        result = level_manager.get_dihedrals(data_container, level="residue")

        # Should result in no resdies
        self.assertEqual(result, [])

    def test_compute_dihedral_conformations(self):
        """
        Test `compute_dihedral_conformations` to ensure it correctly calls
        `assign_conformation` on each dihedral and returns the expected
        list of conformation strings.
        """

        # Setup
        level_manager = LevelManager()

        # Mock selector (can be anything since we're mocking internals)
        selector = MagicMock()

        # Mock dihedrals: pretend we have 3 dihedrals
        mocked_dihedrals = ["d1", "d2", "d3"]
        level_manager.get_dihedrals = MagicMock(return_value=mocked_dihedrals)

        # Mock the conformation entropy (ce) object with assign_conformation method
        ce = MagicMock()
        # For each dihedral, assign_conformation returns a numpy array of ints
        ce.assign_conformation = MagicMock(
            side_effect=[
                np.array([0, 1, 2]),
                np.array([1, 0, 1]),
                np.array([2, 2, 0]),
            ]
        )

        number_frames = 3
        bin_width = 10
        start = 0
        end = 3
        step = 1
        level = "residue"

        # Call the method
        states = level_manager.compute_dihedral_conformations(
            selector, level, number_frames, bin_width, start, end, step, ce
        )

        # Expected states per frame
        expected_states = [
            "012",  # frame 0: d1=0, d2=1, d3=2
            "102",  # frame 1: d1=1, d2=0, d3=2
            "210",  # frame 2: d1=2, d2=1, d3=0
        ]

        # Verify the call count matches the number of dihedrals
        self.assertEqual(ce.assign_conformation.call_count, len(mocked_dihedrals))

        # Verify returned states are as expected
        self.assertEqual(states, expected_states)

        # Verify get_dihedrals was called once with correct arguments
        level_manager.get_dihedrals.assert_called_once_with(selector, level)

    def test_compute_dihedral_conformations_no_dihedrals(self):
        """
        Test `compute_dihedral_conformations` when no dihedrals are found.
        Ensures it returns an empty list of states.
        """
        level_manager = LevelManager()

        level_manager.get_dihedrals = MagicMock(return_value=[])

        selector = MagicMock()

        result = level_manager.compute_dihedral_conformations(
            selector=selector,
            level="united_atom",
            number_frames=10,
            bin_width=10.0,
            start=0,
            end=10,
            step=1,
            ce=MagicMock(),
        )

        self.assertEqual(result, [])

    def test_get_beads_polymer_level(self):
        """
        Test `get_beads` for 'polymer' level.
        Should return a single atom group representing the whole system.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        mock_atom_group = MagicMock()

        data_container.select_atoms.return_value = mock_atom_group

        result = level_manager.get_beads(data_container, level="polymer")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_atom_group)
        data_container.select_atoms.assert_called_once_with("all")

    def test_get_beads_residue_level(self):
        """
        Test `get_beads` for 'residue' level.
        Should return one atom group per residue.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        data_container.residues = [0, 1, 2]  # 3 residues
        mock_atom_group = MagicMock()
        data_container.select_atoms.return_value = mock_atom_group

        result = level_manager.get_beads(data_container, level="residue")

        self.assertEqual(len(result), 3)
        self.assertTrue(all(bead == mock_atom_group for bead in result))
        self.assertEqual(data_container.select_atoms.call_count, 3)

    def test_get_beads_united_atom_level(self):
        """
        Test `get_beads` for 'united_atom' level.
        Should return one bead per heavy atom, including bonded hydrogens.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        heavy_atoms = [MagicMock(index=i) for i in range(3)]
        data_container.select_atoms.side_effect = [
            heavy_atoms,
            "bead0",
            "bead1",
            "bead2",
        ]

        result = level_manager.get_beads(data_container, level="united_atom")

        self.assertEqual(len(result), 3)
        self.assertEqual(result, ["bead0", "bead1", "bead2"])
        self.assertEqual(
            data_container.select_atoms.call_count, 4
        )  # 1 for heavy_atoms + 3 beads

    def test_get_axes_united_atom_no_bonds(self):
        """
        Test `get_axes` for 'united_atom' level when no bonded atoms are found.
        Ensures that rotational axes fall back to residues' principal axes.
        """
        level_manager = LevelManager()

        data_container = MagicMock()

        # Mock principal axes for translation and rotation
        mock_rot_axes = MagicMock(name="rot_axes")

        data_container.residues.principal_axes.return_value = mock_rot_axes
        data_container.residues.principal_axes.return_value = mock_rot_axes
        data_container.residues.principal_axes.return_value = mock_rot_axes  # fallback

        # First select_atoms returns empty bonded atom set
        atom_set = MagicMock()
        atom_set.__len__.return_value = 0  # triggers fallback

        data_container.select_atoms.side_effect = [atom_set]

        trans_axes, rot_axes = level_manager.get_axes(
            data_container=data_container, level="united_atom", index=5
        )

        # Assertions
        self.assertEqual(trans_axes, mock_rot_axes)
        self.assertEqual(rot_axes, mock_rot_axes)
        data_container.residues.principal_axes.assert_called()
        self.assertEqual(data_container.select_atoms.call_count, 1)

    def test_get_axes_polymer_level(self):
        """
        Test `get_axes` for 'polymer' level.
        Should return principal axes of the full system for both
        translation and rotation.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        principal_axes = np.identity(3)
        data_container.atoms.principal_axes.return_value = principal_axes

        trans_axes, rot_axes = level_manager.get_axes(data_container, level="polymer")

        self.assertTrue((trans_axes == principal_axes).all())
        self.assertTrue((rot_axes == principal_axes).all())

    def test_get_axes_residue_level_with_bonds(self):
        """
        Test `get_axes` for 'residue' level with bonded neighbors.
        Should use spherical coordinate axes for rotation.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        data_container.atoms.principal_axes.return_value = "trans_axes"

        atom_set = MagicMock()
        atom_set.__len__.return_value = 1

        residue = MagicMock()
        residue.atoms.center_of_mass.return_value = "center"
        residue.atoms.principal_axes.return_value = "fallback_rot_axes"

        data_container.select_atoms.side_effect = [atom_set, residue]

        level_manager.get_avg_pos = MagicMock(return_value="vector")
        level_manager.get_sphCoord_axes = MagicMock(return_value="rot_axes")

        trans_axes, rot_axes = level_manager.get_axes(
            data_container, level="residue", index=2
        )

        self.assertEqual(trans_axes, "trans_axes")
        self.assertEqual(rot_axes, "rot_axes")

    def test_get_axes_residue_level_without_bonds(self):
        """
        Test `get_axes` for 'residue' level with no bonded neighbors.
        Should use principal axes of the residue for rotation.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        data_container.atoms.principal_axes.return_value = "trans_axes"

        empty_atom_set = []
        residue = MagicMock()
        residue.atoms.principal_axes.return_value = "rot_axes"

        data_container.select_atoms.side_effect = [empty_atom_set, residue]

        trans_axes, rot_axes = level_manager.get_axes(
            data_container, level="residue", index=2
        )

        self.assertEqual(trans_axes, "trans_axes")
        self.assertEqual(rot_axes, "rot_axes")

    def test_get_axes_united_atom_level(self):
        """
        Test `get_axes` for 'united_atom' level.
        Should use residue principal axes for translation and spherical
        axes for rotation.
        """
        level_manager = LevelManager()

        data_container = MagicMock()
        data_container.residues.principal_axes.return_value = "trans_axes"

        atom_set = MagicMock()
        atom_set.__len__.return_value = 1

        atom_group = MagicMock()
        atom_group.positions = [[1.0, 2.0, 3.0]]

        data_container.select_atoms.side_effect = [atom_set, atom_group]

        level_manager.get_avg_pos = MagicMock(return_value="vector")
        level_manager.get_sphCoord_axes = MagicMock(return_value="rot_axes")

        trans_axes, rot_axes = level_manager.get_axes(
            data_container, level="united_atom", index=5
        )

        self.assertEqual(trans_axes, "trans_axes")
        self.assertEqual(rot_axes, "rot_axes")

    def test_get_avg_pos_with_atoms(self):
        """
        Test `get_avg_pos` with a non-empty atom set.
        Should return the average of atom positions minus the center.
        """
        level_manager = LevelManager()

        atom1 = MagicMock()
        atom1.position = np.array([1.0, 2.0, 3.0])
        atom2 = MagicMock()
        atom2.position = np.array([4.0, 5.0, 6.0])

        atom_set = MagicMock()
        atom_set.names = ["A", "B"]
        atom_set.atoms = [atom1, atom2]

        center = np.array([1.0, 1.0, 1.0])
        expected_avg = ((atom1.position + atom2.position) / 2) - center

        result = level_manager.get_avg_pos(atom_set, center)
        np.testing.assert_array_almost_equal(result, expected_avg)

    @patch("numpy.random.random")
    def test_get_avg_pos_empty(self, mock_random):
        """
        Test `get_avg_pos` with an empty atom set.
        Should return a random vector minus the center.
        """
        level_manager = LevelManager()

        atom_set = MagicMock()
        atom_set.names = []
        atom_set.atoms = []

        center = np.array([1.0, 1.0, 1.0])
        mock_random.return_value = np.array([0.5, 0.5, 0.5])

        result = level_manager.get_avg_pos(atom_set, center)
        expected = np.array([0.5, 0.5, 0.5]) - center

        np.testing.assert_array_almost_equal(result, expected)

    def test_get_sphCoord_axes_valid_vector(self):
        """
        Test with a valid non-zero vector.
        Should return a 3x3 orthonormal basis matrix.
        """
        level_manager = LevelManager()

        vector = np.array([1.0, 1.0, 1.0])
        result = level_manager.get_sphCoord_axes(vector)

        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_get_sphCoord_axes_vector_on_z_axis_raises(self):
        """
        Test with a vector along the z-axis (x2y2 == 0).
        Should raise ValueError due to undefined phi.
        """
        level_manager = LevelManager()

        vector = np.array([0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            level_manager.get_sphCoord_axes(vector)

    def test_get_sphCoord_axes_negative_x2y2_div_r2(self):
        """
        Test with a vector that would cause x2y2 / r2 < 0.
        """
        level_manager = LevelManager()

        vector = np.array([1e-10, 1e-10, 1e10])  # x2y2 is tiny, r2 is huge
        result = level_manager.get_sphCoord_axes(vector)
        self.assertEqual(result.shape, (3, 3))

    def test_get_sphCoord_axes_zero_vector_raises(self):
        """
        Test with a zero vector.
        Should raise ValueError due to r2 == 0.
        """
        level_manager = LevelManager()

        vector = np.array([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError) as context:
            level_manager.get_sphCoord_axes(vector)
        self.assertIn("r2 is zero", str(context.exception))

    def test_get_sphCoord_axes_x2y2_zero_raises(self):
        """
        Test with a vector along the z-axis (x2y2 == 0, r2 != 0).
        Should raise ValueError due to undefined phi.
        """
        level_manager = LevelManager()

        vector = np.array([0.0, 0.0, 1.0])  # r2 = 1.0, x2y2 = 0.0
        with self.assertRaises(ValueError) as context:
            level_manager.get_sphCoord_axes(vector)
        self.assertIn("x2y2 is zero", str(context.exception))

    def test_get_weighted_force_with_partitioning(self):
        """
        Test correct weighted force calculation with partitioning enabled.
        """
        level_manager = LevelManager()

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.total_mass.return_value = 4.0

        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.force = np.array([2.0, 0.0, 0.0])

        trans_axes = np.identity(3)

        result = level_manager.get_weighted_forces(
            data_container, bead, trans_axes, highest_level=True
        )

        expected = (0.5 * np.array([2.0, 0.0, 0.0])) / np.sqrt(4.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_weighted_force_without_partitioning(self):
        """
        Test correct weighted force calculation with partitioning disabled.
        """
        level_manager = LevelManager()

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.total_mass.return_value = 1.0

        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.force = np.array([3.0, 0.0, 0.0])

        trans_axes = np.identity(3)

        result = level_manager.get_weighted_forces(
            data_container, bead, trans_axes, highest_level=False
        )

        expected = np.array([3.0, 0.0, 0.0]) / np.sqrt(1.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_weighted_forces_zero_mass_raises_value_error(self):
        """
        Test that a zero mass raises a ValueError.
        """
        level_manager = LevelManager()

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.total_mass.return_value = 0.0

        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.force = np.array([1.0, 0.0, 0.0])

        trans_axes = np.identity(3)

        with self.assertRaises(ValueError):
            level_manager.get_weighted_forces(
                data_container, bead, trans_axes, highest_level=True
            )

    def test_get_weighted_forces_negative_mass_raises_value_error(self):
        """
        Test that a negative mass raises a ValueError.
        """
        level_manager = LevelManager()

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.total_mass.return_value = -1.0

        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.force = np.array([1.0, 0.0, 0.0])

        trans_axes = np.identity(3)

        with self.assertRaises(ValueError):
            level_manager.get_weighted_forces(
                data_container, bead, trans_axes, highest_level=True
            )

    def test_get_weighted_torques_weighted_torque_basic(self):
        """
        Test basic torque calculation with non-zero moment of inertia and torques.
        """
        level_manager = LevelManager()

        # Mock atom
        atom = MagicMock()
        atom.index = 0

        # Mock bead
        bead = MagicMock()
        bead.atoms = [atom]
        bead.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
        bead.moment_of_inertia.return_value = np.identity(3)

        # Mock data_container
        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.position = np.array(
            [1.0, 0.0, 0.0]
        )
        data_container.atoms.__getitem__.return_value.force = np.array([0.0, 1.0, 0.0])

        # Rotation axes (identity matrix)
        rot_axes = np.identity(3)

        result = level_manager.get_weighted_torques(data_container, bead, rot_axes)

        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0, 0.5]))

    def test_get_weighted_torques_zero_torque_skips_division(self):
        """
        Test that zero torque components skip division and remain zero.
        """
        level_manager = LevelManager()

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
        bead.moment_of_inertia.return_value = np.identity(3)

        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.position = np.array(
            [0.0, 0.0, 0.0]
        )
        data_container.atoms.__getitem__.return_value.force = np.array([0.0, 0.0, 0.0])

        rot_axes = np.identity(3)

        result = level_manager.get_weighted_torques(data_container, bead, rot_axes)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_get_weighted_torques_zero_moi_raises(self):
        """
        Should raise ZeroDivisionError when moment of inertia is zero in a dimension
        and torque in that dimension is non-zero.
        """
        level_manager = LevelManager()

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

        # Set moment of inertia with zero in dimension 2
        moi = np.identity(3)
        moi[2, 2] = 0.0
        bead.moment_of_inertia.return_value = moi

        data_container = MagicMock()
        # Position and force that will produce a non-zero torque in z (dimension 2)
        data_container.atoms.__getitem__.return_value.position = np.array(
            [1.0, 0.0, 0.0]
        )
        data_container.atoms.__getitem__.return_value.force = np.array([0.0, 1.0, 0.0])

        rot_axes = np.identity(3)

        with self.assertRaises(ZeroDivisionError):
            level_manager.get_weighted_torques(data_container, bead, rot_axes)

    def test_get_weighted_torques_negative_moi_raises(self):
        """
        Should raise ValueError when moment of inertia is negative in a dimension
        and torque in that dimension is non-zero.
        """
        level_manager = LevelManager()

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

        # Set moment of inertia with negative value in dimension 2
        moi = np.identity(3)
        moi[2, 2] = -1.0
        bead.moment_of_inertia.return_value = moi

        data_container = MagicMock()
        # Position and force that will produce a non-zero torque in z (dimension 2)
        data_container.atoms.__getitem__.return_value.position = np.array(
            [1.0, 0.0, 0.0]
        )
        data_container.atoms.__getitem__.return_value.force = np.array([0.0, 1.0, 0.0])

        rot_axes = np.identity(3)

        with self.assertRaises(ValueError) as context:
            level_manager.get_weighted_torques(data_container, bead, rot_axes)

        self.assertIn(
            "Negative value encountered for moment of inertia", str(context.exception)
        )

    def test_create_submatrix_basic_outer_product(self):
        """
        Test with known vectors to verify correct outer product.
        """
        level_manager = LevelManager()

        data_i = np.array([1, 0, 0])
        data_j = np.array([0, 1, 0])

        expected = np.outer(data_i, data_j)
        result = level_manager.create_submatrix(data_i, data_j)

        np.testing.assert_array_equal(result, expected)

    def test_create_submatrix_zero_vectors_returns_zero_matrix(self):
        """
        Test that all-zero input vectors return a zero matrix.
        """
        level_manager = LevelManager()

        data_i = np.zeros(3)
        data_j = np.zeros(3)
        result = level_manager.create_submatrix(data_i, data_j)

        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_create_submatrix_single_frame(self):
        """
        Test that one frame should return the outer product of the single pair of
        vectors.
        """
        level_manager = LevelManager()

        vec_i = np.array([1, 2, 3])
        vec_j = np.array([4, 5, 6])
        expected = np.outer(vec_i, vec_j)

        result = level_manager.create_submatrix([vec_i], [vec_j])
        np.testing.assert_array_almost_equal(result, expected)

    def test_create_submatrix_symmetric_result_when_data_equal(self):
        """
        Test that if data_i == data_j, the result is symmetric.
        """
        level_manager = LevelManager()

        data = np.array([1, 2, 3])
        result = level_manager.create_submatrix(data, data)

        self.assertTrue(np.allclose(result, result.T))  # Check symmetry

    def test_build_covariance_matrices_atomic(self):
        """
        Test `build_covariance_matrices` to ensure it correctly orchestrates
        calls and returns dictionaries with the expected structure.

        This test mocks dependencies including the entropy_manager, reduced_atom
        trajectory, levels, groups, and internal method
        `update_force_torque_matrices`.
        """

        # Instantiate your class (replace YourClass with actual class name)
        level_manager = LevelManager()

        # Mock entropy_manager and _get_molecule_container
        entropy_manager = MagicMock()

        # Fake atom with minimal attributes
        atom = MagicMock()
        atom.resname = "RES"
        atom.resid = 1
        atom.segid = "A"

        # Fake molecule with atoms list
        fake_mol = MagicMock()
        fake_mol.atoms = [atom]

        # Always return fake_mol from _get_molecule_container
        entropy_manager._get_molecule_container = MagicMock(return_value=fake_mol)

        # Mock reduced_atom with trajectory yielding two timesteps
        timestep1 = MagicMock()
        timestep1.frame = 0
        timestep2 = MagicMock()
        timestep2.frame = 1
        reduced_atom = MagicMock()
        reduced_atom.trajectory.__getitem__.return_value = [timestep1, timestep2]

        # Setup groups and levels dictionaries
        groups = {"ua": ["mol1", "mol2"]}
        levels = {"mol1": ["level1", "level2"], "mol2": ["level1"]}

        # Mock update_force_torque_matrices to just track calls
        level_manager.update_force_torque_matrices = MagicMock()

        # Call the method under test
        force_matrices, torque_matrices, _ = level_manager.build_covariance_matrices(
            entropy_manager=entropy_manager,
            reduced_atom=reduced_atom,
            levels=levels,
            groups=groups,
            start=0,
            end=2,
            step=1,
            number_frames=2,
        )

        # Assert returned matrices are dictionaries with correct keys
        self.assertIsInstance(force_matrices, dict)
        self.assertIsInstance(torque_matrices, dict)
        self.assertSetEqual(set(force_matrices.keys()), {"ua", "res", "poly"})
        self.assertSetEqual(set(torque_matrices.keys()), {"ua", "res", "poly"})

        # Assert 'res' and 'poly' entries are lists of correct length
        self.assertIsInstance(force_matrices["res"], list)
        self.assertIsInstance(force_matrices["poly"], list)
        self.assertEqual(len(force_matrices["res"]), len(groups))
        self.assertEqual(len(force_matrices["poly"]), len(groups))

        # Check _get_molecule_container call count: 2 timesteps * 2 molecules = 4 calls
        self.assertEqual(entropy_manager._get_molecule_container.call_count, 10)

        # Check update_force_torque_matrices call count:
        self.assertEqual(level_manager.update_force_torque_matrices.call_count, 6)

    def test_update_force_torque_matrices_united_atom(self):
        """
        Test that `update_force_torque_matrices` correctly updates force and torque
        matrices for the 'united_atom' level, assigning per-residue matrices and
        incrementing frame counts.
        """
        level_manager = LevelManager()
        entropy_manager = MagicMock()
        run_manager = MagicMock()
        entropy_manager._run_manager = run_manager

        mock_residue_group = MagicMock()
        mock_residue_group.trajectory.__getitem__.return_value = None
        run_manager.new_U_select_atom.return_value = mock_residue_group

        mock_residue1 = MagicMock()
        mock_residue1.atoms.indices = [0, 2]
        mock_residue2 = MagicMock()
        mock_residue2.atoms.indices = [3, 5]

        mol = MagicMock()
        mol.residues = [mock_residue1, mock_residue2]

        f_mat_mock = np.array([[1]])
        t_mat_mock = np.array([[2]])
        level_manager.get_matrices = MagicMock(return_value=(f_mat_mock, t_mat_mock))

        force_avg = {"ua": {}, "res": [None], "poly": [None]}
        torque_avg = {"ua": {}, "res": [None], "poly": [None]}
        frame_counts = {"ua": {}, "res": [None], "poly": [None]}

        level_manager.update_force_torque_matrices(
            entropy_manager=entropy_manager,
            mol=mol,
            group_id=0,
            level="united_atom",
            level_list=["residue", "united_atom"],
            time_index=5,
            num_frames=10,
            force_avg=force_avg,
            torque_avg=torque_avg,
            frame_counts=frame_counts,
        )

        expected_keys = [(0, 0), (0, 1)]
        for key in expected_keys:
            np.testing.assert_array_equal(force_avg["ua"][key], f_mat_mock)
            np.testing.assert_array_equal(torque_avg["ua"][key], t_mat_mock)
            self.assertEqual(frame_counts["ua"][key], 1)

    def test_update_force_torque_matrices_residue(self):
        """
        Test that `update_force_torque_matrices` correctly updates force and torque
        matrices for the 'residue' level, assigning whole-molecule matrices and
        incrementing frame counts.
        """
        level_manager = LevelManager()
        entropy_manager = MagicMock()
        mol = MagicMock()
        mol.trajectory.__getitem__.return_value = None

        f_mat_mock = np.array([[1]])
        t_mat_mock = np.array([[2]])
        level_manager.get_matrices = MagicMock(return_value=(f_mat_mock, t_mat_mock))

        force_avg = {"ua": {}, "res": [None], "poly": [None]}
        torque_avg = {"ua": {}, "res": [None], "poly": [None]}
        frame_counts = {"ua": {}, "res": [None], "poly": [None]}

        level_manager.update_force_torque_matrices(
            entropy_manager=entropy_manager,
            mol=mol,
            group_id=0,
            level="residue",
            level_list=["residue", "united_atom"],
            time_index=3,
            num_frames=10,
            force_avg=force_avg,
            torque_avg=torque_avg,
            frame_counts=frame_counts,
        )

        np.testing.assert_array_equal(force_avg["res"][0], f_mat_mock)
        np.testing.assert_array_equal(torque_avg["res"][0], t_mat_mock)
        self.assertEqual(frame_counts["res"][0], 1)

    def test_update_force_torque_matrices_incremental_average(self):
        """
        Test that `update_force_torque_matrices` correctly applies the incremental
        mean formula when updating force and torque matrices over multiple frames.

        Ensures that float precision is maintained and no casting errors occur.
        """
        level_manager = LevelManager()
        entropy_manager = MagicMock()
        mol = MagicMock()
        mol.trajectory.__getitem__.return_value = None

        # Ensure matrices are float64 to avoid casting errors
        f_mat_1 = np.array([[1.0]], dtype=np.float64)
        t_mat_1 = np.array([[2.0]], dtype=np.float64)
        f_mat_2 = np.array([[3.0]], dtype=np.float64)
        t_mat_2 = np.array([[4.0]], dtype=np.float64)

        level_manager.get_matrices = MagicMock(
            side_effect=[(f_mat_1, t_mat_1), (f_mat_2, t_mat_2)]
        )

        force_avg = {"ua": {}, "res": [None], "poly": [None]}
        torque_avg = {"ua": {}, "res": [None], "poly": [None]}
        frame_counts = {"ua": {}, "res": [None], "poly": [None]}

        # First update
        level_manager.update_force_torque_matrices(
            entropy_manager=entropy_manager,
            mol=mol,
            group_id=0,
            level="residue",
            level_list=["residue", "united_atom"],
            time_index=0,
            num_frames=10,
            force_avg=force_avg,
            torque_avg=torque_avg,
            frame_counts=frame_counts,
        )

        # Second update
        level_manager.update_force_torque_matrices(
            entropy_manager=entropy_manager,
            mol=mol,
            group_id=0,
            level="residue",
            level_list=["residue", "united_atom"],
            time_index=1,
            num_frames=10,
            force_avg=force_avg,
            torque_avg=torque_avg,
            frame_counts=frame_counts,
        )

        expected_force = f_mat_1 + (f_mat_2 - f_mat_1) / 2
        expected_torque = t_mat_1 + (t_mat_2 - t_mat_1) / 2

        np.testing.assert_array_almost_equal(force_avg["res"][0], expected_force)
        np.testing.assert_array_almost_equal(torque_avg["res"][0], expected_torque)
        self.assertEqual(frame_counts["res"][0], 2)

    def test_filter_zero_rows_columns_no_zeros(self):
        """
        Test that matrix with no zero-only rows or columns should return unchanged.
        """
        level_manager = LevelManager()

        matrix = np.array([[1, 2], [3, 4]])
        result = level_manager.filter_zero_rows_columns(matrix)
        np.testing.assert_array_equal(result, matrix)

    def test_filter_zero_rows_columns_remove_rows_and_columns(self):
        """
        Test that matrix with zero-only rows and columns should return reduced matrix.
        """
        level_manager = LevelManager()

        matrix = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
        expected = np.array([[5]])
        result = level_manager.filter_zero_rows_columns(matrix)
        np.testing.assert_array_equal(result, expected)

    def test_filter_zero_rows_columns_all_zeros(self):
        """
        Test that matrix with all zeros should return an empty matrix.
        """
        level_manager = LevelManager()

        matrix = np.zeros((3, 3))
        result = level_manager.filter_zero_rows_columns(matrix)
        self.assertEqual(result.size, 0)
        self.assertEqual(result.shape, (0, 0))

    def test_filter_zero_rows_columns_partial_zero_removal(self):
        """
        Matrix with zeros in specific rows/columns should remove only those.
        """
        level_manager = LevelManager()

        matrix = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]])
        expected = np.array([[1, 2, 3]])
        result = level_manager.filter_zero_rows_columns(matrix)
        np.testing.assert_array_equal(result, expected)

    def test_build_conformational_states_united_atom_accumulates_states(self):
        """
        Test that the 'build_conformational_states' method correctly accumulates
        united atom level conformational states for multiple molecules within the
        same group.

        Specifically, when called with two molecules in the same group, the method
        should append the states returned for the second molecule to the list of
        states for the first molecule, resulting in a nested list structure.

        Verifies:
        - The states_ua dictionary accumulates states as a nested list.
        - The compute_dihedral_conformations method is called once per molecule.
        """
        level_manager = LevelManager()
        entropy_manager = MagicMock()
        reduced_atom = MagicMock()
        ce = MagicMock()

        # Setup mock residue for molecules
        residue = MagicMock()
        residue.atoms.indices = [10, 11, 12]

        # Setup two mock molecules with the same residue
        mol_0 = MagicMock()
        mol_0.residues = [residue]
        mol_1 = MagicMock()
        mol_1.residues = [residue]

        # entropy_manager returns different molecules by mol_id
        entropy_manager._get_molecule_container.side_effect = [mol_0, mol_1]

        # new_U_select_atom returns dummy selections twice per molecule call
        dummy_sel_1 = MagicMock()
        dummy_sel_2 = MagicMock()
        # For mol_0: light then heavy
        # For mol_1: light then heavy
        entropy_manager._run_manager.new_U_select_atom.side_effect = [
            dummy_sel_1,
            dummy_sel_2,
            dummy_sel_1,
            dummy_sel_2,
        ]

        # Mock compute_dihedral_conformations to return different states for each call
        state_1 = ["ua_state_1"]
        state_2 = ["ua_state_2"]
        level_manager.compute_dihedral_conformations = MagicMock(
            side_effect=[state_1, state_2]
        )

        groups = {0: [0, 1]}  # Group 0 contains molecule 0 and molecule 1
        levels = [["united_atom"], ["united_atom"]]
        start, end, step = 0, 10, 1
        number_frames = 10
        bin_width = 0.1

        states_ua, states_res = level_manager.build_conformational_states(
            entropy_manager,
            reduced_atom,
            levels,
            groups,
            start,
            end,
            step,
            number_frames,
            bin_width,
            ce,
        )

        assert states_ua[(0, 0)] == ["ua_state_1", "ua_state_2"]

        # Confirm compute_dihedral_conformations was called twice (once per molecule)
        assert level_manager.compute_dihedral_conformations.call_count == 2

    def test_build_conformational_states_residue_level_accumulates_states(self):
        """
        Test that the 'build_conformational_states' method correctly accumulates
        residue level conformational states for multiple molecules within the
        same group.

        When called with multiple molecules assigned to the same group at residue level,
        the method should concatenate the returned states into a single flat list.

        Verifies:
        - The states_res list contains concatenated residue states from all molecules.
        - The states_ua dictionary remains empty for residue level.
        - compute_dihedral_conformations is called once per molecule.
        """
        level_manager = LevelManager()
        entropy_manager = MagicMock()
        reduced_atom = MagicMock()
        ce = MagicMock()

        # Setup molecule with no residues
        mol = MagicMock()
        mol.residues = []
        entropy_manager._get_molecule_container.return_value = mol

        # Setup return values for compute_dihedral_conformations
        states_1 = ["res_state1"]
        states_2 = ["res_state2"]
        level_manager.compute_dihedral_conformations = MagicMock(
            side_effect=[states_1, states_2]
        )

        # Setup inputs with 2 molecules in same group
        groups = {0: [0, 1]}  # Both mol 0 and mol 1 are in group 0
        levels = [["residue"], ["residue"]]
        start, end, step = 0, 10, 1
        number_frames = 10
        bin_width = 0.1

        # Run
        states_ua, states_res = level_manager.build_conformational_states(
            entropy_manager,
            reduced_atom,
            levels,
            groups,
            start,
            end,
            step,
            number_frames,
            bin_width,
            ce,
        )

        # Confirm accumulation occurred
        assert states_ua == {}
        assert states_res[0] == ["res_state1", "res_state2"]
        assert states_res == [["res_state1", "res_state2"]]

        # Assert both calls to compute_dihedral_conformations happened
        assert level_manager.compute_dihedral_conformations.call_count == 2
