from unittest.mock import MagicMock, patch

import numpy as np

from CodeEntropy.levels import LevelManager
from CodeEntropy.mda_universe_operations import UniverseOperations
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

        universe_operations = UniverseOperations()

        # Import the class and call the method
        level_manager = LevelManager(universe_operations)
        number_molecules, levels = level_manager.select_levels(data_container)

        # Assertions
        self.assertEqual(number_molecules, 2)
        self.assertEqual(
            levels, [["united_atom", "residue"], ["united_atom", "residue", "polymer"]]
        )

    def test_get_matrices(self):
        """
        Atomic unit test for LevelManager.get_matrices:
        - AxesManager is mocked
        - No inertia / MDAnalysis math
        - Verifies block matrix construction and shape only
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        # Two beads
        bead1 = MagicMock()
        bead1.principal_axes.return_value = np.ones(3)

        bead2 = MagicMock()
        bead2.principal_axes.return_value = np.ones(3)

        level_manager.get_beads = MagicMock(return_value=[bead1, bead2])

        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )

        # Deterministic 3x3 submatrix for every (i,j) call
        I3 = np.identity(3)
        level_manager.create_submatrix = MagicMock(return_value=I3)

        data_container = MagicMock()
        data_container.atoms = MagicMock()
        data_container.atoms.principal_axes.return_value = np.ones(3)

        dummy_trans_axes = np.eye(3)
        dummy_rot_axes = np.eye(3)
        dummy_center = np.zeros(3)
        dummy_moi = np.eye(3)

        with patch("CodeEntropy.levels.AxesManager") as AxesManagerMock:
            axes = AxesManagerMock.return_value
            axes.get_residue_axes.return_value = (
                dummy_trans_axes,
                dummy_rot_axes,
                dummy_center,
                dummy_moi,
            )

            force_matrix, torque_matrix = level_manager.get_matrices(
                data_container=data_container,
                level="residue",
                highest_level=True,
                force_matrix=None,
                torque_matrix=None,
                force_partitioning=0.5,
                customised_axes=True,
            )

        # Shape: 2 beads × 3 dof => 6×6
        assert force_matrix.shape == (6, 6)
        assert torque_matrix.shape == (6, 6)

        # Expected block structure when every block is I3:
        expected = np.block([[I3, I3], [I3, I3]])
        np.testing.assert_array_equal(force_matrix, expected)
        np.testing.assert_array_equal(torque_matrix, expected)

        # Lightweight behavioral assertions
        level_manager.get_beads.assert_called_once_with(data_container, "residue")
        assert axes.get_residue_axes.call_count == 2

        # For 2 beads: (0,0), (0,1), (1,1) => 3 pairs;
        # each pair calls create_submatrix twice (force+torque)
        assert level_manager.create_submatrix.call_count == 6

    def test_get_matrices_force_shape_mismatch(self):
        """
        Test that get_matrices raises a ValueError when the provided force_matrix
        has a shape mismatch with the computed force block matrix.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        # Two beads -> force_block will be 6x6
        bead1 = MagicMock()
        bead1.principal_axes.return_value = np.ones(3)

        bead2 = MagicMock()
        bead2.principal_axes.return_value = np.ones(3)

        level_manager.get_beads = MagicMock(return_value=[bead1, bead2])

        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )

        level_manager.create_submatrix = MagicMock(return_value=np.identity(3))

        data_container = MagicMock()
        data_container.atoms = MagicMock()
        data_container.atoms.principal_axes.return_value = np.ones(3)

        bad_force_matrix = np.zeros((3, 3))
        correct_torque_matrix = np.zeros((6, 6))

        dummy_trans_axes = np.eye(3)
        dummy_rot_axes = np.eye(3)
        dummy_center = np.zeros(3)
        dummy_moi = np.eye(3)

        with patch("CodeEntropy.levels.AxesManager") as AxesManagerMock:
            axes = AxesManagerMock.return_value
            axes.get_residue_axes.return_value = (
                dummy_trans_axes,
                dummy_rot_axes,
                dummy_center,
                dummy_moi,
            )

            with self.assertRaises(ValueError) as context:
                level_manager.get_matrices(
                    data_container=data_container,
                    level="residue",
                    highest_level=True,
                    force_matrix=bad_force_matrix,
                    torque_matrix=correct_torque_matrix,
                    force_partitioning=0.5,
                    customised_axes=True,
                )

        assert "force matrix shape" in str(context.exception)

    def test_get_matrices_torque_shape_mismatch(self):
        """
        Test that get_matrices raises a ValueError when the provided torque_matrix
        has a shape mismatch with the computed torque block matrix.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        bead1 = MagicMock()
        bead1.principal_axes.return_value = np.ones(3)

        bead2 = MagicMock()
        bead2.principal_axes.return_value = np.ones(3)

        level_manager.get_beads = MagicMock(return_value=[bead1, bead2])

        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )
        level_manager.create_submatrix = MagicMock(return_value=np.identity(3))

        data_container = MagicMock()
        data_container.atoms = MagicMock()
        data_container.atoms.principal_axes.return_value = np.ones(3)

        correct_force_matrix = np.zeros((6, 6))
        bad_torque_matrix = np.zeros((3, 3))  # Incorrect shape (should be 6x6)

        # Mock AxesManager return tuple to satisfy unpacking
        dummy_trans_axes = np.eye(3)
        dummy_rot_axes = np.eye(3)
        dummy_center = np.zeros(3)
        dummy_moi = np.eye(3)

        with patch("CodeEntropy.levels.AxesManager") as AxesManagerMock:
            axes = AxesManagerMock.return_value
            axes.get_residue_axes.return_value = (
                dummy_trans_axes,
                dummy_rot_axes,
                dummy_center,
                dummy_moi,
            )

            with self.assertRaises(ValueError) as context:
                level_manager.get_matrices(
                    data_container=data_container,
                    level="residue",
                    highest_level=True,
                    force_matrix=correct_force_matrix,
                    torque_matrix=bad_torque_matrix,
                    force_partitioning=0.5,
                    customised_axes=True,
                )

        assert "torque matrix shape" in str(context.exception)

    def test_get_matrices_torque_consistency(self):
        """
        Test that get_matrices returns consistent force and torque matrices
        when called multiple times with the same inputs.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        bead1 = MagicMock()
        bead1.principal_axes.return_value = np.ones(3)

        bead2 = MagicMock()
        bead2.principal_axes.return_value = np.ones(3)

        level_manager.get_beads = MagicMock(return_value=[bead1, bead2])

        level_manager.get_weighted_forces = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        level_manager.get_weighted_torques = MagicMock(
            return_value=np.array([0.5, 1.5, 2.5])
        )
        level_manager.create_submatrix = MagicMock(return_value=np.identity(3))

        data_container = MagicMock()
        data_container.atoms = MagicMock()
        data_container.atoms.principal_axes.return_value = np.ones(3)

        initial_force_matrix = np.zeros((6, 6))
        initial_torque_matrix = np.zeros((6, 6))

        # Mock AxesManager return tuple (unpacked by get_matrices)
        dummy_trans_axes = np.eye(3)
        dummy_rot_axes = np.eye(3)
        dummy_center = np.zeros(3)
        dummy_moi = np.eye(3)

        with patch("CodeEntropy.levels.AxesManager") as AxesManagerMock:
            axes = AxesManagerMock.return_value
            axes.get_residue_axes.return_value = (
                dummy_trans_axes,
                dummy_rot_axes,
                dummy_center,
                dummy_moi,
            )

            force_matrix_1, torque_matrix_1 = level_manager.get_matrices(
                data_container=data_container,
                level="residue",
                highest_level=True,
                force_matrix=initial_force_matrix.copy(),
                torque_matrix=initial_torque_matrix.copy(),
                force_partitioning=0.5,
                customised_axes=True,
            )

            force_matrix_2, torque_matrix_2 = level_manager.get_matrices(
                data_container=data_container,
                level="residue",
                highest_level=True,
                force_matrix=initial_force_matrix.copy(),
                torque_matrix=initial_torque_matrix.copy(),
                force_partitioning=0.5,
                customised_axes=True,
            )

        np.testing.assert_array_equal(force_matrix_1, force_matrix_2)
        np.testing.assert_array_equal(torque_matrix_1, torque_matrix_2)

        assert force_matrix_1.shape == (6, 6)
        assert torque_matrix_1.shape == (6, 6)

    def test_get_beads_polymer_level(self):
        """
        Test `get_beads` for 'polymer' level.
        Should return a single atom group representing the whole system.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

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
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

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
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

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

    def test_get_beads_hydrogen_molecule(self):
        """
        Test `get_beads` for 'united_atom' level.
        Should return one bead for molecule with no heavy atoms.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        data_container = MagicMock()
        heavy_atoms = []
        data_container.select_atoms.side_effect = [
            heavy_atoms,
            "hydrogen",
        ]

        result = level_manager.get_beads(data_container, level="united_atom")

        self.assertEqual(len(result), 1)
        self.assertEqual(result, ["hydrogen"])
        self.assertEqual(
            data_container.select_atoms.call_count, 2
        )  # 1 for heavy_atoms + 1 beads

    def test_get_weighted_force_with_partitioning(self):
        """
        Test correct weighted force calculation with partitioning enabled.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.total_mass.return_value = 4.0

        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.force = np.array([2.0, 0.0, 0.0])

        trans_axes = np.identity(3)

        result = level_manager.get_weighted_forces(
            data_container, bead, trans_axes, highest_level=True, force_partitioning=0.5
        )

        expected = (0.5 * np.array([2.0, 0.0, 0.0])) / np.sqrt(4.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_weighted_force_without_partitioning(self):
        """
        Test correct weighted force calculation with partitioning disabled.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        atom = MagicMock()
        atom.index = 0

        bead = MagicMock()
        bead.atoms = [atom]
        bead.total_mass.return_value = 1.0

        data_container = MagicMock()
        data_container.atoms.__getitem__.return_value.force = np.array([3.0, 0.0, 0.0])

        trans_axes = np.identity(3)

        result = level_manager.get_weighted_forces(
            data_container,
            bead,
            trans_axes,
            highest_level=False,
            force_partitioning=0.5,
        )

        expected = np.array([3.0, 0.0, 0.0]) / np.sqrt(1.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_weighted_forces_zero_mass_raises_value_error(self):
        """
        Test that a zero mass raises a ValueError.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

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
                data_container,
                bead,
                trans_axes,
                highest_level=True,
                force_partitioning=0.5,
            )

    def test_get_weighted_forces_negative_mass_raises_value_error(self):
        """
        Test that a negative mass raises a ValueError.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

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
                data_container,
                bead,
                trans_axes,
                highest_level=True,
                force_partitioning=0.5,
            )

    def test_get_weighted_torques_weighted_torque_basic(self):
        """
        Test basic torque calculation with non-zero moment of inertia and torques.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        # Bead with one "atom"
        bead = MagicMock()
        bead.positions = np.array([[1.0, 0.0, 0.0]])  # r
        bead.forces = np.array([[0.0, 1.0, 0.0]])  # F

        rot_axes = np.identity(3)
        center = np.array([0.0, 0.0, 0.0])
        force_partitioning = 0.5
        moment_of_inertia = np.array([1.0, 1.0, 1.0])

        result = level_manager.get_weighted_torques(
            bead=bead,
            rot_axes=rot_axes,
            center=center,
            force_partitioning=force_partitioning,
            moment_of_inertia=moment_of_inertia,
        )

        expected = np.array([0.0, 0.0, 0.5])
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_get_weighted_torques_zero_torque_skips_division(self):
        """
        Test that zero torque components skip division and remain zero.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        bead = MagicMock()
        # All zeros => r x F = 0
        bead.positions = np.array([[0.0, 0.0, 0.0]])
        bead.forces = np.array([[0.0, 0.0, 0.0]])

        rot_axes = np.identity(3)
        center = np.array([0.0, 0.0, 0.0])
        force_partitioning = 0.5

        # Use non-zero MOI so that "skip division" is only due to zero torque
        moment_of_inertia = np.array([1.0, 2.0, 3.0])

        result = level_manager.get_weighted_torques(
            bead=bead,
            rot_axes=rot_axes,
            center=center,
            force_partitioning=force_partitioning,
            moment_of_inertia=moment_of_inertia,
        )

        expected = np.zeros(3)
        np.testing.assert_array_equal(result, expected)

    def test_get_weighted_torques_zero_moi(self):
        """
        Should set torque to 0 when moment of inertia is zero in a dimension
        and torque in that dimension is non-zero.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        bead = MagicMock()
        # r = (1,0,0), F = (0,1,0) => torque = (0,0,1)
        bead.positions = np.array([[1.0, 0.0, 0.0]])
        bead.forces = np.array([[0.0, 1.0, 0.0]])

        rot_axes = np.identity(3)
        center = np.array([0.0, 0.0, 0.0])
        force_partitioning = 0.5

        # MOI is zero in z dimension (index 2)
        moment_of_inertia = np.array([1.0, 1.0, 0.0])

        torque = level_manager.get_weighted_torques(
            bead=bead,
            rot_axes=rot_axes,
            center=center,
            force_partitioning=force_partitioning,
            moment_of_inertia=moment_of_inertia,
        )

        # x and y torques are zero; z torque is non-zero
        # but MOI_z==0 => weighted z should be 0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(torque, expected)

    def test_get_weighted_torques_negative_moi_sets_zero(self):
        """
        Negative moment of inertia components should be skipped and set to 0
        even if the corresponding torque component is non-zero.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        bead = MagicMock()
        # r=(1,0,0), F=(0,1,0) => raw torque in z is non-zero
        bead.positions = np.array([[1.0, 0.0, 0.0]])
        bead.forces = np.array([[0.0, 1.0, 0.0]])

        rot_axes = np.identity(3)
        center = np.array([0.0, 0.0, 0.0])
        force_partitioning = 0.5

        # Negative MOI in z dimension
        moment_of_inertia = np.array([1.0, 1.0, -1.0])

        result = level_manager.get_weighted_torques(
            bead=bead,
            rot_axes=rot_axes,
            center=center,
            force_partitioning=force_partitioning,
            moment_of_inertia=moment_of_inertia,
        )

        # z torque would be non-zero, but negative MOI => z component forced to 0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_create_submatrix_basic_outer_product(self):
        """
        Test with known vectors to verify correct outer product.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        data_i = np.array([1, 0, 0])
        data_j = np.array([0, 1, 0])

        expected = np.outer(data_i, data_j)
        result = level_manager.create_submatrix(data_i, data_j)

        np.testing.assert_array_equal(result, expected)

    def test_create_submatrix_zero_vectors_returns_zero_matrix(self):
        """
        Test that all-zero input vectors return a zero matrix.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        data_i = np.zeros(3)
        data_j = np.zeros(3)
        result = level_manager.create_submatrix(data_i, data_j)

        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_create_submatrix_single_frame(self):
        """
        Test that one frame should return the outer product of the single pair of
        vectors.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        vec_i = np.array([1, 2, 3])
        vec_j = np.array([4, 5, 6])
        expected = np.outer(vec_i, vec_j)

        result = level_manager.create_submatrix([vec_i], [vec_j])
        np.testing.assert_array_almost_equal(result, expected)

    def test_create_submatrix_symmetric_result_when_data_equal(self):
        """
        Test that if data_i == data_j, the result is symmetric.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        data = np.array([1, 2, 3])
        result = level_manager.create_submatrix(data, data)

        self.assertTrue(np.allclose(result, result.T))  # Check symmetry

    def test_build_covariance_matrices_atomic(self):
        """
        Test `build_covariance_matrices` to ensure it correctly orchestrates
        calls and returns dictionaries with the expected structure.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        entropy_manager = MagicMock()

        # Fake atom with minimal attributes
        atom = MagicMock()
        atom.resname = "RES"
        atom.resid = 1
        atom.segid = "A"

        fake_mol = MagicMock()
        fake_mol.atoms = [atom]

        universe_operations.get_molecule_container = MagicMock(return_value=fake_mol)

        timestep1 = MagicMock()
        timestep1.frame = 0
        timestep2 = MagicMock()
        timestep2.frame = 1

        reduced_atom = MagicMock()
        reduced_atom.trajectory.__getitem__.return_value = [timestep1, timestep2]

        groups = {"ua": ["mol1", "mol2"]}
        levels = {"mol1": ["level1", "level2"], "mol2": ["level1"]}

        level_manager.update_force_torque_matrices = MagicMock()

        force_matrices, torque_matrices, *_ = level_manager.build_covariance_matrices(
            entropy_manager=entropy_manager,
            reduced_atom=reduced_atom,
            levels=levels,
            groups=groups,
            start=0,
            end=2,
            step=1,
            number_frames=2,
            force_partitioning=0.5,
            combined_forcetorque=False,
            customised_axes=True,
        )

        self.assertIsInstance(force_matrices, dict)
        self.assertIsInstance(torque_matrices, dict)
        self.assertSetEqual(set(force_matrices.keys()), {"ua", "res", "poly"})
        self.assertSetEqual(set(torque_matrices.keys()), {"ua", "res", "poly"})

        self.assertIsInstance(force_matrices["res"], list)
        self.assertIsInstance(force_matrices["poly"], list)
        self.assertEqual(len(force_matrices["res"]), len(groups))
        self.assertEqual(len(force_matrices["poly"]), len(groups))

        self.assertEqual(universe_operations.get_molecule_container.call_count, 4)
        self.assertEqual(level_manager.update_force_torque_matrices.call_count, 6)

    def test_update_force_torque_matrices_united_atom(self):
        """
        Test that update_force_torque_matrices() correctly initializes force and torque
        matrices for the 'united_atom' level.

        Ensures:
        - The matrices are initialized for each UA group key.
        - Frame counts are incremented correctly.
        """
        universe_operations = UniverseOperations()
        universe_operations.new_U_select_atom = MagicMock()

        level_manager = LevelManager(universe_operations)

        entropy_manager = MagicMock()
        run_manager = MagicMock()
        entropy_manager._run_manager = run_manager

        mock_res = MagicMock()
        mock_res.trajectory = MagicMock()
        mock_res.trajectory.__getitem__.return_value = None

        universe_operations.new_U_select_atom.return_value = mock_res

        mock_residue1 = MagicMock()
        mock_residue1.atoms.indices = [0, 2]
        mock_residue2 = MagicMock()
        mock_residue2.atoms.indices = [3, 5]

        mol = MagicMock()
        mol.residues = [mock_residue1, mock_residue2]

        f_mat = np.array([[1]])
        t_mat = np.array([[2]])
        level_manager.get_matrices = MagicMock(return_value=(f_mat, t_mat))

        force_avg = {"ua": {}, "res": [None], "poly": [None]}
        torque_avg = {"ua": {}, "res": [None], "poly": [None]}
        forcetorque_avg = {"ua": {}, "res": [None], "poly": [None]}
        frame_counts = {"ua": {}, "res": [None], "poly": [None]}

        level_manager.update_force_torque_matrices(
            entropy_manager=entropy_manager,
            mol=mol,
            group_id=0,
            level="united_atom",
            level_list=["residue", "united_atom"],
            time_index=0,
            num_frames=10,
            force_avg=force_avg,
            torque_avg=torque_avg,
            forcetorque_avg=forcetorque_avg,
            frame_counts=frame_counts,
            force_partitioning=0.5,
            combined_forcetorque=False,
            customised_axes=True,
        )

        assert (0, 0) in force_avg["ua"]
        assert (0, 1) in force_avg["ua"]
        assert (0, 0) in torque_avg["ua"]
        assert (0, 1) in torque_avg["ua"]

        np.testing.assert_array_equal(force_avg["ua"][(0, 0)], f_mat)
        np.testing.assert_array_equal(force_avg["ua"][(0, 1)], f_mat)
        np.testing.assert_array_equal(torque_avg["ua"][(0, 0)], t_mat)
        np.testing.assert_array_equal(torque_avg["ua"][(0, 1)], t_mat)

        assert frame_counts["ua"][(0, 0)] == 1
        assert frame_counts["ua"][(0, 1)] == 1

        assert forcetorque_avg["ua"] == {}

    def test_update_force_torque_matrices_united_atom_increment(self):
        """
        Test that update_force_torque_matrices() correctly updates (increments)
        existing force and torque matrices for the 'united_atom' level.

        Confirms correct incremental averaging behavior.
        """
        universe_operations = UniverseOperations()
        universe_operations.new_U_select_atom = MagicMock()

        level_manager = LevelManager(universe_operations)

        entropy_manager = MagicMock()
        mol = MagicMock()

        residue = MagicMock()
        residue.atoms.indices = [0, 1]
        mol.residues = [residue]
        mol.trajectory = MagicMock()
        mol.trajectory.__getitem__.return_value = None

        selected_atoms = MagicMock()
        selected_atoms.trajectory = MagicMock()
        selected_atoms.trajectory.__getitem__.return_value = None
        universe_operations.new_U_select_atom.return_value = selected_atoms

        f_mat_1 = np.array([[1.0]])
        t_mat_1 = np.array([[2.0]])

        f_mat_2 = np.array([[3.0]])
        t_mat_2 = np.array([[4.0]])

        level_manager.get_matrices = MagicMock(
            side_effect=[(f_mat_1, t_mat_1), (f_mat_2, t_mat_2)]
        )

        force_avg = {"ua": {}, "res": [None], "poly": [None]}
        torque_avg = {"ua": {}, "res": [None], "poly": [None]}
        forcetorque_avg = {"ua": {}, "res": [None], "poly": [None]}
        frame_counts = {"ua": {}, "res": [None], "poly": [None]}

        level_manager.update_force_torque_matrices(
            entropy_manager=entropy_manager,
            mol=mol,
            group_id=0,
            level="united_atom",
            level_list=["residue", "united_atom"],
            time_index=0,
            num_frames=10,
            force_avg=force_avg,
            torque_avg=torque_avg,
            forcetorque_avg=forcetorque_avg,
            frame_counts=frame_counts,
            force_partitioning=0.5,
            combined_forcetorque=False,
            customised_axes=True,
        )

        # Second update
        level_manager.update_force_torque_matrices(
            entropy_manager=entropy_manager,
            mol=mol,
            group_id=0,
            level="united_atom",
            level_list=["residue", "united_atom"],
            time_index=1,
            num_frames=10,
            force_avg=force_avg,
            torque_avg=torque_avg,
            forcetorque_avg=forcetorque_avg,
            frame_counts=frame_counts,
            force_partitioning=0.5,
            combined_forcetorque=False,
            customised_axes=True,
        )

        expected_force = f_mat_1 + (f_mat_2 - f_mat_1) / 2
        expected_torque = t_mat_1 + (t_mat_2 - t_mat_1) / 2

        np.testing.assert_array_almost_equal(force_avg["ua"][(0, 0)], expected_force)
        np.testing.assert_array_almost_equal(torque_avg["ua"][(0, 0)], expected_torque)
        assert frame_counts["ua"][(0, 0)] == 2

        assert forcetorque_avg["ua"] == {}

    def test_update_force_torque_matrices_residue(self):
        """
        Test that `update_force_torque_matrices` correctly updates force and torque
        matrices for the 'residue' level, assigning whole-molecule matrices and
        incrementing frame counts.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

        entropy_manager = MagicMock()
        mol = MagicMock()
        mol.trajectory.__getitem__.return_value = None

        f_mat_mock = np.array([[1]])
        t_mat_mock = np.array([[2]])
        level_manager.get_matrices = MagicMock(return_value=(f_mat_mock, t_mat_mock))

        force_avg = {"ua": {}, "res": [None], "poly": [None]}
        torque_avg = {"ua": {}, "res": [None], "poly": [None]}
        forcetorque_avg = {"ua": {}, "res": [None], "poly": [None]}
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
            forcetorque_avg=forcetorque_avg,
            frame_counts=frame_counts,
            force_partitioning=0.5,
            combined_forcetorque=False,
            customised_axes=True,
        )

        np.testing.assert_array_equal(force_avg["res"][0], f_mat_mock)
        np.testing.assert_array_equal(torque_avg["res"][0], t_mat_mock)
        assert frame_counts["res"][0] == 1

        assert forcetorque_avg["res"][0] is None

    def test_update_force_torque_matrices_incremental_average(self):
        """
        Test that `update_force_torque_matrices` correctly applies the incremental
        mean formula when updating force and torque matrices over multiple frames.

        Ensures that float precision is maintained and no casting errors occur.
        """
        universe_operations = UniverseOperations()
        level_manager = LevelManager(universe_operations)

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
        forcetorque_avg = {"ua": {}, "res": [None], "poly": [None]}
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
            forcetorque_avg=forcetorque_avg,
            frame_counts=frame_counts,
            force_partitioning=0.5,
            combined_forcetorque=False,
            customised_axes=True,
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
            forcetorque_avg=forcetorque_avg,
            frame_counts=frame_counts,
            force_partitioning=0.5,
            combined_forcetorque=False,
            customised_axes=True,
        )

        expected_force = f_mat_1 + (f_mat_2 - f_mat_1) / 2
        expected_torque = t_mat_1 + (t_mat_2 - t_mat_1) / 2

        np.testing.assert_array_almost_equal(force_avg["res"][0], expected_force)
        np.testing.assert_array_almost_equal(torque_avg["res"][0], expected_torque)

        assert frame_counts["res"][0] == 2
        assert forcetorque_avg["res"][0] is None

    def test_filter_zero_rows_columns_no_zeros(self):
        """
        Test that matrix with no zero-only rows or columns should return unchanged.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        matrix = np.array([[1, 2], [3, 4]])
        result = level_manager.filter_zero_rows_columns(matrix)
        np.testing.assert_array_equal(result, matrix)

    def test_filter_zero_rows_columns_remove_rows_and_columns(self):
        """
        Test that matrix with zero-only rows and columns should return reduced matrix.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        matrix = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
        expected = np.array([[5]])
        result = level_manager.filter_zero_rows_columns(matrix)
        np.testing.assert_array_equal(result, expected)

    def test_filter_zero_rows_columns_all_zeros(self):
        """
        Test that matrix with all zeros should return an empty matrix.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        matrix = np.zeros((3, 3))
        result = level_manager.filter_zero_rows_columns(matrix)
        self.assertEqual(result.size, 0)
        self.assertEqual(result.shape, (0, 0))

    def test_filter_zero_rows_columns_partial_zero_removal(self):
        """
        Matrix with zeros in specific rows/columns should remove only those.
        """
        universe_operations = UniverseOperations()

        level_manager = LevelManager(universe_operations)

        matrix = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]])
        expected = np.array([[1, 2, 3]])
        result = level_manager.filter_zero_rows_columns(matrix)
        np.testing.assert_array_equal(result, expected)
