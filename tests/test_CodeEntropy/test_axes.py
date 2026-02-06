from unittest.mock import MagicMock, patch

import numpy as np

from CodeEntropy.axes import AxesManager
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestAxesManager(BaseTestCase):
    def setUp(self):
        super().setUp()

    def test_get_residue_axes_no_bonds_custom_axes_branch(self):
        """
        Tests that: atom_set empty (len == 0) -> custom axes branch
        """
        axes_manager = AxesManager()
        data_container = MagicMock()

        atom_set = MagicMock()
        atom_set.__len__.return_value = 0

        residue = MagicMock()

        data_container.select_atoms.side_effect = [atom_set, residue]

        center = np.array([1.0, 2.0, 3.0])
        residue.atoms.center_of_mass.return_value = center

        UAs = MagicMock()
        UAs.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        residue.select_atoms.return_value = UAs

        UA_masses = [12.0, 14.0]
        axes_manager.get_UA_masses = MagicMock(return_value=UA_masses)

        moi_tensor = np.eye(3) * 5.0
        axes_manager.get_moment_of_inertia_tensor = MagicMock(return_value=moi_tensor)

        rot_axes = np.eye(3)
        moi = np.array([10.0, 9.0, 8.0])
        axes_manager.get_custom_principal_axes = MagicMock(return_value=(rot_axes, moi))

        trans_axes_out, rot_axes_out, center_out, moi_out = (
            axes_manager.get_residue_axes(
                data_container=data_container,
                index=5,
            )
        )

        calls = data_container.select_atoms.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == "(resindex 4 or resindex 6) and bonded resid 5"
        assert calls[1].args[0] == "resindex 5"

        residue.select_atoms.assert_called_once_with("mass 2 to 999")

        axes_manager.get_UA_masses.assert_called_once_with(residue)

        axes_manager.get_moment_of_inertia_tensor.assert_called_once()
        tensor_args, tensor_kwargs = axes_manager.get_moment_of_inertia_tensor.call_args
        np.testing.assert_array_equal(tensor_args[0], center)
        np.testing.assert_array_equal(tensor_args[1], UAs.positions)
        assert tensor_args[2] == UA_masses
        assert tensor_kwargs == {}

        axes_manager.get_custom_principal_axes.assert_called_once_with(moi_tensor)

        np.testing.assert_array_equal(trans_axes_out, rot_axes)
        np.testing.assert_array_equal(rot_axes_out, rot_axes)
        np.testing.assert_array_equal(center_out, center)
        np.testing.assert_array_equal(moi_out, moi)

    def test_get_residue_axes_bonded_default_axes_branch(self):
        """
        Tests that: atom_set non-empty (len != 0) -> default/bonded branch
        """
        axes_manager = AxesManager()
        data_container = MagicMock()
        data_container.atoms = MagicMock()

        atom_set = MagicMock()
        atom_set.__len__.return_value = 2

        residue = MagicMock()
        data_container.select_atoms.side_effect = [atom_set, residue]

        trans_axes_expected = np.eye(3) * 2
        data_container.atoms.principal_axes.return_value = trans_axes_expected

        rot_axes_expected = np.eye(3) * 3
        residue.principal_axes.return_value = rot_axes_expected

        moi_tensor = np.eye(3)
        residue.moment_of_inertia.return_value = moi_tensor

        center_expected = np.array([9.0, 8.0, 7.0])
        residue.atoms.center_of_mass.return_value = center_expected
        residue.center_of_mass.return_value = center_expected

        with (
            patch("CodeEntropy.axes.make_whole", autospec=True),
            patch.object(
                AxesManager,
                "get_vanilla_axes",
                return_value=(rot_axes_expected, np.array([3.0, 2.0, 1.0])),
            ),
        ):
            trans_axes_out, rot_axes_out, center_out, moi_out = (
                axes_manager.get_residue_axes(
                    data_container=data_container,
                    index=5,
                )
            )

        np.testing.assert_allclose(trans_axes_out, trans_axes_expected)
        np.testing.assert_allclose(rot_axes_out, rot_axes_expected)
        np.testing.assert_allclose(center_out, center_expected)
        np.testing.assert_allclose(moi_out, np.array([3.0, 2.0, 1.0]))

    def test_get_UA_axes_returns_expected_outputs(self):
        """
        Tests that: `get_UA_axes` returns expected UA axes.
        """
        axes = AxesManager()

        dc = MagicMock()
        dc.atoms = MagicMock()
        dc.dimensions = np.array([1.0, 2.0, 3.0, 90.0, 90.0, 90.0])
        dc.atoms.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

        uas = MagicMock()
        uas.positions = np.zeros((2, 3))

        a0 = MagicMock()
        a0.index = 7
        a1 = MagicMock()
        a1.index = 9
        heavy_atoms = [a0, a1]

        heavy_ag = MagicMock()
        heavy_ag.positions = np.array([[9.9, 8.8, 7.7]])
        heavy_ag.__getitem__.return_value = MagicMock()

        dc.select_atoms.side_effect = [uas, heavy_atoms, heavy_ag]

        axes.get_UA_masses = MagicMock(return_value=[1.0, 1.0])
        axes.get_moment_of_inertia_tensor = MagicMock(return_value=np.eye(3))

        trans_axes_expected = np.eye(3)
        axes.get_custom_principal_axes = MagicMock(
            return_value=(trans_axes_expected, np.array([1.0, 1.0, 1.0]))
        )

        rot_axes_expected = np.eye(3) * 2
        moi_expected = np.array([3.0, 2.0, 1.0])
        axes.get_bonded_axes = MagicMock(return_value=(rot_axes_expected, moi_expected))

        trans_axes, rot_axes, center, moi = axes.get_UA_axes(dc, index=1)

        np.testing.assert_array_equal(trans_axes, trans_axes_expected)
        np.testing.assert_array_equal(rot_axes, rot_axes_expected)
        np.testing.assert_array_equal(center, heavy_ag.positions[0])
        np.testing.assert_array_equal(moi, moi_expected)

        calls = [c.args[0] for c in dc.select_atoms.call_args_list]
        assert calls[0] == "mass 2 to 999"
        assert calls[1] == "prop mass > 1.1"
        assert calls[2] == "index 9"

    def test_get_bonded_axes_returns_none_for_light_atom(self):
        """
        Tests that: bonded axes return none for light atoms
        """
        axes = AxesManager()

        atom = MagicMock()
        atom.mass = 1.0
        system = MagicMock()

        out = axes.get_bonded_axes(
            system=system, atom=atom, dimensions=np.array([1.0, 2.0, 3.0])
        )
        assert out is None

    def test_get_bonded_axes_case2_one_heavy_zero_light(self):
        """
        Tests that: bonded return one heavy and zero light atoms
        """
        axes = AxesManager()

        system = MagicMock()
        atom = MagicMock()
        atom.mass = 12.0
        atom.index = 0
        atom.position = np.array([0.0, 0.0, 0.0])

        heavy0 = MagicMock()
        heavy0.position = np.array([1.0, 0.0, 0.0])

        heavy_bonded = [heavy0]
        light_bonded = []

        axes.find_bonded_atoms = MagicMock(return_value=(heavy_bonded, light_bonded))

        custom_axes = np.eye(3)
        axes.get_custom_axes = MagicMock(return_value=custom_axes)

        moi = np.array([1.0, 2.0, 3.0])
        axes.get_custom_moment_of_inertia = MagicMock(return_value=moi)

        flipped_axes = np.eye(3) * 2
        axes.get_flipped_axes = MagicMock(return_value=flipped_axes)

        out_axes, out_moi = axes.get_bonded_axes(
            system, atom, np.array([10.0, 10.0, 10.0])
        )

        np.testing.assert_array_equal(out_axes, flipped_axes)
        np.testing.assert_array_equal(out_moi, moi)

        axes.get_custom_axes.assert_called_once()
        args, _ = axes.get_custom_axes.call_args
        np.testing.assert_array_equal(args[0], atom.position)
        assert len(args[1]) == 1
        np.testing.assert_array_equal(args[1][0], heavy0.position)
        np.testing.assert_array_equal(args[2], np.zeros(3))
        np.testing.assert_array_equal(args[3], np.array([10.0, 10.0, 10.0]))

    def test_get_bonded_axes_case3_one_heavy_with_light(self):
        """
        Tests that: bonded axes return one heavy with one light atom
        """
        axes = AxesManager()

        system = MagicMock()
        atom = MagicMock()
        atom.mass = 12.0
        atom.index = 0
        atom.position = np.array([0.0, 0.0, 0.0])

        heavy0 = MagicMock()
        heavy0.position = np.array([1.0, 0.0, 0.0])

        light0 = MagicMock()
        light0.position = np.array([0.0, 1.0, 0.0])

        heavy_bonded = [heavy0]
        light_bonded = [light0]

        axes.find_bonded_atoms = MagicMock(return_value=(heavy_bonded, light_bonded))

        custom_axes = np.eye(3)
        axes.get_custom_axes = MagicMock(return_value=custom_axes)
        axes.get_custom_moment_of_inertia = MagicMock(
            return_value=np.array([1.0, 1.0, 1.0])
        )
        axes.get_flipped_axes = MagicMock(return_value=custom_axes)

        axes.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

        axes.get_custom_axes.assert_called_once()
        args, _ = axes.get_custom_axes.call_args

        np.testing.assert_array_equal(args[2], light0.position)

    def test_get_bonded_axes_case5_two_or_more_heavy(self):
        """
        Tests that: bonded axes return two or more heavy atoms
        """
        axes = AxesManager()

        system = MagicMock()
        atom = MagicMock()
        atom.mass = 12.0
        atom.index = 0
        atom.position = np.array([0.0, 0.0, 0.0])

        heavy0 = MagicMock()
        heavy0.position = np.array([1.0, 0.0, 0.0])
        heavy1 = MagicMock()
        heavy1.position = np.array([0.0, 1.0, 0.0])

        heavy_bonded = MagicMock()
        heavy_bonded.__len__.return_value = 2
        heavy_bonded.positions = np.array([heavy0.position, heavy1.position])

        heavy_bonded.__getitem__.side_effect = lambda i: [heavy0, heavy1][i]

        light_bonded = []

        axes.find_bonded_atoms = MagicMock(return_value=(heavy_bonded, light_bonded))

        custom_axes = np.eye(3)
        axes.get_custom_axes = MagicMock(return_value=custom_axes)
        axes.get_custom_moment_of_inertia = MagicMock(
            return_value=np.array([9.0, 9.0, 9.0])
        )
        axes.get_flipped_axes = MagicMock(return_value=custom_axes)

        axes.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

        axes.get_custom_axes.assert_called_once()
        args, _ = axes.get_custom_axes.call_args

        np.testing.assert_array_equal(args[1], heavy_bonded.positions)
        np.testing.assert_array_equal(args[2], heavy1.position)

    def test_find_bonded_atoms_splits_heavy_and_h(self):
        """
        Tests that: Bonded atoms split into heavy and hydrogen.
        """
        axes = AxesManager()

        system = MagicMock()
        bonded = MagicMock()
        heavy = MagicMock()
        hydrogens = MagicMock()

        system.select_atoms.return_value = bonded
        bonded.select_atoms.side_effect = [heavy, hydrogens]

        out_heavy, out_h = axes.find_bonded_atoms(5, system)

        system.select_atoms.assert_called_once_with("bonded index 5")
        assert bonded.select_atoms.call_args_list[0].args[0] == "mass 2 to 999"
        assert bonded.select_atoms.call_args_list[1].args[0] == "mass 1 to 1.1"
        assert out_heavy is heavy
        assert out_h is hydrogens

    def test_get_vector_wraps_pbc(self):
        """
        Tests that: The vector wraps across periodic boundary.
        """
        axes = AxesManager()

        a = np.array([9.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        dims = np.array([10.0, 10.0, 10.0])

        out = axes.get_vector(a, b, dims)
        np.testing.assert_array_equal(out, np.array([2.0, 0.0, 0.0]))

    def test_get_custom_axes_returns_unit_axes(self):
        """
        Tests that: `get_axes` returns normalized 3x3 axes.
        """
        axes = AxesManager()

        a = np.zeros(3)
        b_list = [np.array([1.0, 0.0, 0.0])]
        c = np.array([0.0, 1.0, 0.0])
        dims = np.array([100.0, 100.0, 100.0])

        out = axes.get_custom_axes(a, b_list, c, dims)

        assert out.shape == (3, 3)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(3))

    def test_get_custom_axes_uses_bc_vector_when_multiple_heavy_atoms(self):
        """
        Tests that: `get_custom_axes` uses c → b_list[0] vector when b_list has
        ≥ 2 atoms.
        """
        axes = AxesManager()

        a = np.zeros(3)
        b0 = np.array([1.0, 0.0, 0.0])
        b1 = np.array([0.0, 1.0, 0.0])
        b_list = [b0, b1]
        c = np.array([0.0, 0.0, 1.0])
        dimensions = np.array([10.0, 10.0, 10.0])

        # Track calls to get_vector
        axes.get_vector = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))

        axes.get_custom_axes(a, b_list, c, dimensions)

        # get_vector should be called
        calls = axes.get_vector.call_args_list

        # Last call must be (c, b_list[0], dimensions)
        last_args = calls[-1].args
        np.testing.assert_array_equal(last_args[0], c)
        np.testing.assert_array_equal(last_args[1], b0)
        np.testing.assert_array_equal(last_args[2], dimensions)

    def test_get_custom_moment_of_inertia_len2_zeroed(self):
        """
        Tests that: `get_custom_moment_of_inertia` zeroes one MOI component for
        two-atom UA.
        """
        axes = AxesManager()

        UA = MagicMock()
        UA.positions = np.array([[1, 0, 0], [0, 1, 0]])
        UA.masses = np.array([12.0, 1.0])
        UA.__len__.return_value = 2

        dimensions = np.array([100.0, 100.0, 100.0])

        moi = axes.get_custom_moment_of_inertia(UA, np.eye(3), np.zeros(3), dimensions)

        assert moi.shape == (3,)
        assert np.any(np.isclose(moi, 0.0))

    def test_get_flipped_axes_flips_negative_dot(self):
        """
        Tests that: `get_flipped_axes` flips axis when dot product is negative.
        """
        axes = AxesManager()

        UA = MagicMock()
        atom0 = MagicMock()
        atom0.position = np.zeros(3)
        UA.__getitem__.return_value = atom0

        axes.get_vector = MagicMock(return_value=np.array([-1.0, 0.0, 0.0]))

        custom_axes = np.eye(3)
        out = axes.get_flipped_axes(
            UA, custom_axes, np.zeros(3), np.array([10, 10, 10])
        )

        np.testing.assert_array_equal(out[0], np.array([-1.0, 0.0, 0.0]))

    def test_get_moment_of_inertia_tensor_simple(self):
        """
        Tests that: `get_moment_of_inertia` Computes inertia tensor correctly.
        """
        axes = AxesManager()

        center = np.zeros(3)
        pos = np.array([[1, 0, 0], [0, 1, 0]])
        masses = np.array([1.0, 1.0])
        dimensions = np.array([100.0, 100.0, 100.0])

        tensor = axes.get_moment_of_inertia_tensor(center, pos, masses, dimensions)

        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
        np.testing.assert_array_equal(tensor, expected)

    def test_get_custom_principal_axes_flips_z(self):
        """
        Tests that: `get_custom_principle_axes` ensures right-handed axes.
        """
        axes = AxesManager()

        with patch("CodeEntropy.axes.np.linalg.eig") as eig:
            eig.return_value = (
                np.array([3, 2, 1]),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            )

            axes_out, moi = axes.get_custom_principal_axes(np.eye(3))

        np.testing.assert_array_equal(axes_out[2], np.array([0, 0, 1]))

    def test_get_UA_masses_sums_hydrogens(self):
        """
        Tests that: `get_UA_masses` sums heavy atom with bonded hydrogens.
        """
        axes = AxesManager()

        heavy = MagicMock(mass=12.0, index=0)
        light = MagicMock(mass=1.0, index=1)

        mol = MagicMock()
        mol.__iter__.return_value = iter([heavy, light])

        bonded = MagicMock()
        H = MagicMock(mass=1.0)
        mol.select_atoms.return_value = bonded
        bonded.select_atoms.return_value = [H]

        out = axes.get_UA_masses(mol)

        assert out == [13.0]
