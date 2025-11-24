from unittest.mock import MagicMock

from CodeEntropy.dihedral_tools import DihedralAnalysis
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestDihedralAnalysis(BaseTestCase):
    """
    Unit tests for DihedralAnalysis.
    """

    def setUp(self):
        super().setUp()
        self.analysis = DihedralAnalysis()

    def test_get_dihedrals_united_atom(self):
        """
        Test `_get_dihedrals` for 'united_atom' level.

        The function should:
        - read dihedrals from `data_container.dihedrals`
        - extract `.atoms` from each dihedral
        - return a list of atom groups

        Expected behavior:
        If dihedrals = [d1, d2, d3] and each dihedral has an `.atoms`
        attribute, then the returned list must be:
            [d1.atoms, d2.atoms, d3.atoms]
        """
        data_container = MagicMock()

        # Mock dihedral objects with `.atoms`
        d1 = MagicMock()
        d1.atoms = "atoms1"
        d2 = MagicMock()
        d2.atoms = "atoms2"
        d3 = MagicMock()
        d3.atoms = "atoms3"

        data_container.dihedrals = [d1, d2, d3]

        result = self.analysis._get_dihedrals(data_container, level="united_atom")

        self.assertEqual(result, ["atoms1", "atoms2", "atoms3"])

    def test_get_dihedrals_residue(self):
        """
        Test `_get_dihedrals` for 'residue' level with 5 residues.

        The implementation:
        - iterates over residues 4 → N
        - for each, selects 4 bonded atom groups
        - merges them using __add__ to form a single atom_group
        - appends to result list

        For 5 residues (0–4), two dihedral groups should be created.
        Expected:
        - result of length 2
        - each item equal to the merged mock atom group
        """
        data_container = MagicMock()
        data_container.residues = [0, 1, 2, 3, 4]

        mock_atom_group = MagicMock()
        mock_atom_group.__add__.return_value = mock_atom_group

        # Every MDAnalysis selection returns the same mock atom group
        data_container.select_atoms.return_value = mock_atom_group

        result = self.analysis._get_dihedrals(data_container, level="residue")

        self.assertEqual(len(result), 2)
        self.assertTrue(all(r is mock_atom_group for r in result))

    def test_get_dihedrals_no_residue(self):
        """
        Test `_get_dihedrals` for 'residue' level when fewer than
        4 residues exist (here: 3 residues).

        Expected:
        - The function returns an empty list.
        """
        data_container = MagicMock()
        data_container.residues = [0, 1, 2]  # Only 3 residues → too few

        result = self.analysis._get_dihedrals(data_container, level="residue")

        self.assertEqual(result, [])
