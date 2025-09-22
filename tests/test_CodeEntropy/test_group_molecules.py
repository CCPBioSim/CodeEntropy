import unittest
from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.group_molecules import GroupMolecules
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestGroupMolecules(BaseTestCase):
    """
    Unit tests for GroupMolecules.
    """

    def setUp(self):
        super().setUp()
        self.group_molecules = GroupMolecules()

    def test_by_none_returns_individual_groups(self):
        """
        Test _by_none returns each molecule in its own group when grouping is 'each'.
        """
        mock_universe = MagicMock()
        # Simulate universe.atoms.fragments has 3 molecules
        mock_universe.atoms.fragments = [MagicMock(), MagicMock(), MagicMock()]

        groups = self.group_molecules._by_none(mock_universe)
        expected = {0: [0], 1: [1], 2: [2]}
        self.assertEqual(groups, expected)

    def test_by_molecules_groups_by_chemical_type(self):
        """
        Test _by_molecules groups molecules with identical atom counts and names
        together.
        """
        mock_universe = MagicMock()

        fragment0 = MagicMock()
        fragment0.names = np.array(["H", "O", "H"])
        fragment1 = MagicMock()
        fragment1.names = np.array(["H", "O", "H"])
        fragment2 = MagicMock()
        fragment2.names = np.array(["C", "C", "H", "H"])

        mock_universe.atoms.fragments = [fragment0, fragment1, fragment2]

        groups = self.group_molecules._by_molecules(mock_universe)

        # Expect first two grouped, third separate
        self.assertIn(0, groups)
        self.assertIn(2, groups)
        self.assertCountEqual(groups[0], [0, 1])
        self.assertEqual(groups[2], [2])

    def test_grouping_molecules_dispatches_correctly(self):
        """
        Test grouping_molecules method dispatches to correct grouping strategy.
        """
        mock_universe = MagicMock()
        mock_universe.atoms.fragments = [MagicMock()]  # Just 1 molecule to keep simple

        # When grouping='each', calls _by_none
        groups = self.group_molecules.grouping_molecules(mock_universe, "each")
        self.assertEqual(groups, {0: [0]})

        # When grouping='molecules', calls _by_molecules (mock to test call)
        self.group_molecules._by_molecules = MagicMock(return_value={"mocked": [42]})
        groups = self.group_molecules.grouping_molecules(mock_universe, "molecules")
        self.group_molecules._by_molecules.assert_called_once_with(mock_universe)
        self.assertEqual(groups, {"mocked": [42]})

        # If grouping unknown, should return empty dict
        groups = self.group_molecules.grouping_molecules(mock_universe, "unknown")
        self.assertEqual(groups, {})


if __name__ == "__main__":
    unittest.main()
