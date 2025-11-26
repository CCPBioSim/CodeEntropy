from unittest.mock import MagicMock, patch

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

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_identify_peaks_empty_dihedrals(self, Dihedral_patch):
        """
        Test `_identify_peaks` returns an empty list when the
        input dihedral list is empty.

        Expected:
            - No angle extraction occurs.
            - No histograms computed.
            - Return value is an empty list.
        """
        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        peaks = analysis._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals=[],
            bin_width=10,
            start=0,
            end=360,
            step=1,
        )

        assert peaks == []

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_identify_peaks_negative_angles_become_positive(self, Dihedral_patch):
        """
        Test that negative dihedral angles are converted into the
        0–360° range before histogramming.

        Scenario:
            - A single dihedral produces a single angle: -15°.
            - This should be converted to +345°.
            - With 90° bins, it falls into the final bin → one peak.

        Expected:
            - One peak detected.
            - Peak center lies between 300° and 360°.
        """
        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        R = MagicMock()
        R.results.angles = [[-15]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0]
        universe_operations.get_molecule_container.return_value = mol

        peaks = analysis._identify_peaks(
            MagicMock(),
            [0],
            dihedrals=[MagicMock()],
            bin_width=90,
            start=0,
            end=360,
            step=1,
        )

        assert len(peaks) == 1
        assert len(peaks[0]) == 1
        assert 300 <= peaks[0][0] <= 360

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_identify_peaks_internal_peak_detection(self, Dihedral_patch):
        """
        Test the detection of a peak located in a middle histogram bin.

        Scenario:
            - Angles fall into bin #1 (45°, 50°, 55°).
            - Bin 1 has higher population than its neighbors.

        Expected:
            - Exactly one peak is detected.
        """
        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        R = MagicMock()
        R.results.angles = [[45], [50], [55]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0, 1, 2]
        universe_operations.get_molecule_container.return_value = mol

        peaks = analysis._identify_peaks(
            MagicMock(),
            [0],
            dihedrals=[MagicMock()],
            bin_width=90,
            start=0,
            end=360,
            step=1,
        )

        assert len(peaks[0]) == 1

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_identify_peaks_circular_boundary(self, Dihedral_patch):
        """
        Test that `_identify_peaks` handles circular histogram boundaries
        correctly when identifying peaks in the last bin.

        Setup:
            - All angles are near 350°, falling into the final bin.

        Expected:
            - The final bin is correctly identified as a peak.
        """
        ops = MagicMock()
        analysis = DihedralAnalysis(ops)

        R = MagicMock()
        R.results.angles = [[350], [355], [349]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0, 1, 2]
        ops.get_molecule_container.return_value = mol

        peaks = analysis._identify_peaks(
            MagicMock(),
            [0],
            dihedrals=[MagicMock()],
            bin_width=90,
            start=0,
            end=360,
            step=1,
        )

        assert len(peaks[0]) == 1

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_identify_peaks_circular_last_bin(self, Dihedral_patch):
        """
        Test peak detection for circular histogram boundaries, where the
        last bin compares against the first bin.

        Scenario:
            - All angles near 350° fall into the final bin.
            - Final bin should be considered a peak if it exceeds both
              previous and first bins.

        Expected:
            - One peak detected in the last bin.
        """
        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        R = MagicMock()
        R.results.angles = [[350], [355], [349]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0, 1, 2]
        universe_operations.get_molecule_container.return_value = mol

        peaks = analysis._identify_peaks(
            MagicMock(),
            [0],
            dihedrals=[MagicMock()],
            bin_width=90,
            start=0,
            end=360,
            step=1,
        )

        assert len(peaks[0]) == 1

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_assign_states_negative_angle_conversion(self, Dihedral_patch):
        """
        Test `_assign_states` converts negative angles correctly and assigns
        the dihedral to the nearest peak.

        Scenario:
            - Angle returned = -10° → converted to 350°.
            - Peak list contains [350].

        Expected:
            - Assigned state is "0".
        """
        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        R = MagicMock()
        R.results.angles = [[-10]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0]
        universe_operations.get_molecule_container.return_value = mol

        states = analysis._assign_states(
            MagicMock(),
            [0],
            dihedrals=[MagicMock()],
            peaks=[[350]],
            start=0,
            end=360,
            step=1,
        )

        assert states == ["0"]

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_assign_states_closest_peak_selection(self, Dihedral_patch):
        """
        Test that `_assign_states` selects the peak nearest to each dihedral
        angle.

        Setup:
            - Angle = 30°.
            - Peaks = [20, 100].
            - Nearest peak = 20 (index 0).

        Expected:
            - Returned state is ["0"].
        """
        ops = MagicMock()
        analysis = DihedralAnalysis(ops)

        R = MagicMock()
        R.results.angles = [[30]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0]
        ops.get_molecule_container.return_value = mol

        states = analysis._assign_states(
            MagicMock(),
            [0],
            dihedrals=[MagicMock()],
            peaks=[[20, 100]],
            start=0,
            end=360,
            step=1,
        )

        assert states == ["0"]

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_assign_states_closest_peak(self, Dihedral_patch):
        """
        Test assignment to the correct peak based on minimum angular distance.

        Scenario:
            - Angle = 30°.
            - Peaks = [20, 100].
            - Closest peak is 20° → index 0.

        Expected:
            - Returned state is "0".
        """
        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        R = MagicMock()
        R.results.angles = [[30]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0]
        universe_operations.get_molecule_container.return_value = mol

        states = analysis._assign_states(
            MagicMock(),
            [0],
            dihedrals=[MagicMock()],
            peaks=[[20, 100]],
            start=0,
            end=360,
            step=1,
        )

        assert states == ["0"]

    @patch("CodeEntropy.dihedral_tools.Dihedral")
    def test_assign_states_multiple_dihedrals(self, Dihedral_patch):
        """
        Test concatenation of state labels across multiple dihedrals.

        Scenario:
            - Two dihedrals, one frame:
                dihedral 0 → 10° → closest peak 0
                dihedral 1 → 200° → closest peak 180 (index 0)
            - Resulting frame state: "00".

        Expected:
            - Returned list: ["00"].
        """
        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        R = MagicMock()
        R.results.angles = [[10, 200]]
        Dihedral_patch.return_value.run.return_value = R

        mol = MagicMock()
        mol.trajectory = [0]
        universe_operations.get_molecule_container.return_value = mol

        peaks = [[0, 180], [180, 300]]

        states = analysis._assign_states(
            MagicMock(),
            [0],
            dihedrals=[MagicMock(), MagicMock()],
            peaks=peaks,
            start=0,
            end=360,
            step=1,
        )

        assert states == ["00"]

    def test_assign_states_multiple_molecules(self):
        """
        Test that `_assign_states` generates different conformational state
        labels for different molecules when their dihedral angle trajectories
        differ.

        Molecule 0 is mocked to produce an angle near peak 0.
        Molecule 1 is mocked to produce an angle near peak 1.

        Expected:
            The returned state list reflects these differences as
            ["0", "1"].
        """

        universe_operations = MagicMock()
        analysis = DihedralAnalysis(universe_operations)

        mol1 = MagicMock()
        mol1.trajectory = [0]

        mol2 = MagicMock()
        mol2.trajectory = [0]

        universe_operations.get_molecule_container.side_effect = [mol1, mol2]

        # Two different R objects
        R1 = MagicMock()
        R1.results.angles = [[10]]  # peak index 0

        R2 = MagicMock()
        R2.results.angles = [[200]]  # peak index 1

        peaks = [[0, 180]]

        # Patch where Dihedral is *used*
        with patch("CodeEntropy.dihedral_tools.Dihedral") as Dihedral_patch:
            instance = Dihedral_patch.return_value
            instance.run.side_effect = [R1, R2]

            states = analysis._assign_states(
                MagicMock(),
                molecules=[0, 1],
                dihedrals=[MagicMock()],
                peaks=peaks,
                start=0,
                end=360,
                step=1,
            )

        assert states == ["0", "1"]

    def test_build_states_united_atom_no_dihedrals(self):
        """
        Test that UA-level state building produces empty state lists when no
        dihedrals are found for any residue.
        """
        ops = MagicMock()
        analysis = DihedralAnalysis(ops)

        mol = MagicMock()
        mol.residues = [MagicMock()]
        ops.get_molecule_container.return_value = mol
        ops.new_U_select_atom.return_value = MagicMock()

        analysis._get_dihedrals = MagicMock(return_value=[])
        analysis._identify_peaks = MagicMock(return_value=[])
        analysis._assign_states = MagicMock(return_value=[])

        groups = {0: [0]}
        levels = {0: ["united_atom"]}

        states_ua, states_res = analysis.build_conformational_states(
            MagicMock(), levels, groups, start=0, end=360, step=1, bin_width=10
        )

        assert states_ua[(0, 0)] == []

    def test_build_states_united_atom_accumulate(self):
        """
        Test that UA-level state building assigns states independently to each
        residue and accumulates them correctly.
        """
        ops = MagicMock()
        analysis = DihedralAnalysis(ops)

        mol = MagicMock()
        mol.residues = [MagicMock(), MagicMock()]
        ops.get_molecule_container.return_value = mol
        ops.new_U_select_atom.return_value = MagicMock()

        analysis._get_dihedrals = MagicMock(return_value=[1])
        analysis._identify_peaks = MagicMock(return_value=[[10]])
        analysis._assign_states = MagicMock(return_value=["A"])

        groups = {0: [0]}
        levels = {0: ["united_atom"]}

        states_ua, _ = analysis.build_conformational_states(
            MagicMock(), levels, groups, start=0, end=360, step=1, bin_width=10
        )

        assert states_ua[(0, 0)] == ["A"]
        assert states_ua[(0, 1)] == ["A"]

    def test_build_states_residue_no_dihedrals(self):
        """
        Test that residue-level state building returns an empty list when
        `_get_dihedrals` reports no available dihedral groups.
        """
        ops = MagicMock()
        analysis = DihedralAnalysis(ops)

        mol = MagicMock()
        mol.residues = [MagicMock()]
        ops.get_molecule_container.return_value = mol

        analysis._get_dihedrals = MagicMock(return_value=[])
        analysis._identify_peaks = MagicMock(return_value=[])
        analysis._assign_states = MagicMock(return_value=[])

        groups = {0: [0]}
        levels = {0: ["residue"]}

        _, states_res = analysis.build_conformational_states(
            MagicMock(), levels, groups, start=0, end=360, step=1, bin_width=10
        )

        assert states_res[0] == []

    def test_build_states_residue_accumulate(self):
        """
        Test that residue-level state building delegates all molecules in a group
        to a single `_assign_states` call, and stores its returned list directly.

        Expected:
            _assign_states returns ["A", "B"], so states_res[0] == ["A", "B"].
        """
        ops = MagicMock()
        analysis = DihedralAnalysis(ops)

        mol1 = MagicMock()
        mol1.residues = [MagicMock()]
        mol2 = MagicMock()
        mol2.residues = [MagicMock()]

        ops.get_molecule_container.side_effect = [mol1, mol2]

        analysis._get_dihedrals = MagicMock(return_value=[1])
        analysis._identify_peaks = MagicMock(return_value=[[10]])

        # One call for the whole group → one return value
        analysis._assign_states = MagicMock(return_value=["A", "B"])

        groups = {0: [0, 1]}
        levels = {0: ["residue"], 1: ["residue"]}

        _, states_res = analysis.build_conformational_states(
            MagicMock(), levels, groups, start=0, end=360, step=1, bin_width=10
        )

        assert states_res[0] == ["A", "B"]
