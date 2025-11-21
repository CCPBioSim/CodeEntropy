from unittest.mock import MagicMock

from CodeEntropy.dihedral_tools import DihedralAnalysis
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestDihedralAnalysis(BaseTestCase):
    """
    Unit tests for DihedralAnalysis.
    """

    def setUp(self):
        super().setUp()

    def test_get_dihedrals_united_atom(self):
        """
        Test `get_dihedrals` for 'united_atom' level.
        Ensures it returns the dihedrals directly from the data container.
        """
        data_container = MagicMock()
        mock_dihedrals = ["d1", "d2", "d3"]
        data_container.dihedrals = mock_dihedrals

        result = DihedralAnalysis._get_dihedrals(data_container, level="united_atom")
        self.assertEqual(result, mock_dihedrals)

    def test_get_dihedrals_residue(self):
        """
        Test `get_dihedrals` for 'residue' level with 5 residues.
        Mocks bonded atom selections and verifies that dihedrals are constructed.
        """
        data_container = MagicMock()
        data_container.residues = [0, 1, 2, 3, 4]  # 5 residues

        # Mock select_atoms to return atom groups with .dihedral
        mock_dihedral = MagicMock()
        mock_atom_group = MagicMock()
        mock_atom_group.__add__.return_value = mock_atom_group
        mock_atom_group.dihedral = mock_dihedral
        data_container.select_atoms.return_value = mock_atom_group

        result = DihedralAnalysis._get_dihedrals(data_container, level="residue")

        # Should create 2 dihedrals for 5 residues (residues 0–3 and 1–4)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(d == mock_dihedral for d in result))

    def test_get_dihedrals_no_residue(self):
        """
        Test `get_dihedrals` for 'residue' level with 3 residues.
        Mocks bonded atom selections and verifies that dihedrals are constructed.
        """

        data_container = MagicMock()
        data_container.residues = [0, 1, 2]  # 3 residues

        # Mock select_atoms to return atom groups with .dihedral
        mock_dihedral = MagicMock()
        mock_atom_group = MagicMock()
        mock_atom_group.__add__.return_value = mock_atom_group
        mock_atom_group.dihedral = mock_dihedral
        data_container.select_atoms.return_value = mock_atom_group

        result = DihedralAnalysis._get_dihedrals(data_container, level="residue")

        # Should result in no residue dihedrals
        self.assertEqual(result, [])
