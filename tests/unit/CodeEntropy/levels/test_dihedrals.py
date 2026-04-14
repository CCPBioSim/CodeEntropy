import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from CodeEntropy.levels.dihedrals import ConformationStateBuilder


class _AddableAG:
    def __init__(self, name: str):
        self.name = name

    def __add__(self, other: "_AddableAG") -> "_AddableAG":
        return _AddableAG(f"({self.name}+{other.name})")


class _FakeProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *args, **kwargs):
        return 1

    def advance(self, *args, **kwargs):
        return None


@contextlib.contextmanager
def _fake_progress_bar(*_args, **_kwargs):
    yield _FakeProgress()


def test_select_heavy_residue_builds_two_selections():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([10, 11, 12], dtype=int)

    uops.select_atoms.side_effect = ["res_container", "heavy_only"]

    out = dt._select_heavy_residue(mol, res_id=0)

    assert out == "heavy_only"
    assert uops.select_atoms.call_count == 2
    uops.select_atoms.assert_any_call(mol, "index 10:12")
    uops.select_atoms.assert_any_call("res_container", "prop mass > 1.1")


def test_get_dihedrals_united_atom_collects_atoms_from_dihedral_objects():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    d0 = MagicMock()
    d0.atoms = "A0"
    d1 = MagicMock()
    d1.atoms = "A1"

    container = MagicMock()
    container.dihedrals = [d0, d1]

    assert dt._get_dihedrals(container, level="united_atom") == ["A0", "A1"]


def test_get_dihedrals_residue_returns_empty_when_less_than_4_residues():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    mol = MagicMock()
    mol.residues = [MagicMock(), MagicMock(), MagicMock()]
    mol.select_atoms = MagicMock()

    assert dt._get_dihedrals(mol, level="residue") == []
    mol.select_atoms.assert_not_called()


def test_get_dihedrals_residue_builds_one_dihedral_when_4_residues():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    mol = MagicMock()
    mol.residues = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    mol.select_atoms = MagicMock(
        side_effect=[
            _AddableAG("a1"),
            _AddableAG("a2"),
            _AddableAG("a3"),
            _AddableAG("a4"),
        ]
    )

    out = dt._get_dihedrals(mol, level="residue")

    assert len(out) == 1
    assert isinstance(out[0], _AddableAG)
    assert mol.select_atoms.call_count == 4


def test_identify_peaks_sets_empty_outputs_when_no_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    peaks_ua, peaks_res = dt._identify_peaks(
        data_container=MagicMock(),
        molecules=[0],
        bin_width=30.0,
        level_list=["united_atom", "residue"],
    )

    assert peaks_ua == []
    assert peaks_res == []


def test_identify_peaks_wraps_negative_angles_and_calls_find_histogram_peaks():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    dihedral = MagicMock()
    angles = np.array([[-10.0], [10.0]], dtype=float)

    dt._get_dihedrals = MagicMock(return_value=dihedral)
    dt._process_dihedral_phi = MagicMock(return_value=angles)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with (
        patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral),
        patch.object(dt, "_process_histogram", return_value=[15.0]) as peaks_spy,
    ):
        out = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            bin_width=10.0,
            level_list=["residue"],
        )

    assert out == ([], [15.0])
    peaks_spy.assert_called_once()


def test_find_histogram_peaks_hits_interior_and_wraparound_last_bin():
    popul = [0, 2, 0, 3]
    bin_value = [10.0, 20.0, 30.0, 40.0]
    assert ConformationStateBuilder._find_histogram_peaks(popul, bin_value) == [
        20.0,
        40.0,
    ]


def test_assign_states_initialises_then_extends_for_multiple_molecules():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    dihedrals = ["D0"]
    angles = np.array([[5.0], [15.0]], dtype=float)
    peaks = [[5.0, 15.0]]
    states_ua = []
    states_res = []
    flexible_ua = []
    flexible_res = []

    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        dt._assign_states(
            data_container=MagicMock(),
            group_id=0,
            molecules=[0, 1],
            level_list=["residue"],
            peaks_ua=[],
            peaks_res=peaks,
            states_ua=states_ua,
            states_res=states_res,
            flexible_ua=flexible_ua,
            flexible_res=flexible_res,
        )

    assert states_res[0] == ["0", "1", "0", "1"]
    assert flexible_res[0] == 1


def test_build_conformational_states_runs_group_and_skips_empty_group(monkeypatch):
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    groups = {0: [], 1: [7]}
    levels = {7: ["residue"]}

    uops.extract_fragment.return_value = MagicMock(trajectory=[0])

    states_ua, states_res, flex_ua, flex_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        bin_width=30.0,
    )

    assert states_ua == {}
    assert len(states_res) == 3
    assert flex_ua == {}
    assert flex_res[0] == 0


def test_identify_peaks_handles_multiple_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    dihedrals = (["D0", "D1"],)
    angles = np.array(
        [
            [-10.0, 10.0],
            [20.0, -20.0],
        ],
        dtype=float,
    )

    dt._get_dihedrals = MagicMock(return_value=dihedrals)
    dt._process_dihedral_phi = MagicMock(return_value=angles)
    dt._process_histogram = MagicMock(return_value=[1, 2])

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        out = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            bin_width=30.0,
            level_list=["united_atom", "residue"],
        )

    assert len(out) == 2


def test_assign_states_filters_out_empty_state_strings_when_no_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1, 2]
    uops.extract_fragment.return_value = mol

    dihedrals = []
    states_ua = []
    states_res = []
    flexible_ua = []
    flexible_res = []

    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=[]))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        dt._assign_states(
            data_container=MagicMock(),
            group_id=0,
            molecules=[0],
            level_list=["residue"],
            peaks_ua=[],
            peaks_res=[],
            states_ua=states_ua,
            states_res=states_res,
            flexible_ua=flexible_ua,
            flexible_res=flexible_res,
        )

    assert states_res[0] == []
    assert flexible_res[0] == 0


def test_identify_peaks_multiple_molecules_real_histogram():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol0 = MagicMock()
    mol0.trajectory = [0, 1]
    mol1 = MagicMock()
    mol1.trajectory = [0, 1]

    uops.extract_fragment.side_effect = [mol0, mol0, mol1]

    dihedrals = ["D0"]
    angles = np.array([[10.0], [20.0]], dtype=float)
    phi_values = {}
    phi_values[0] = np.array([[10.0], [20.0]], dtype=float)

    dt._get_dihedrals = MagicMock(return_value=dihedrals)
    dt._process_dihedral_phi = MagicMock(return_value=phi_values)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with (
        patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral),
        patch(
            "CodeEntropy.levels.dihedrals.ConformationStateBuilder._process_dihedral_phi",
            dt._process_dihedral_phi,
        ),
    ):
        peaks_ua, peaks_res = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0, 1],
            bin_width=90.0,
            level_list=["residue"],
        )

    assert len(peaks_ua) == 0
    assert len(peaks_res) == 1


def test_assign_states_wraps_negative_angles():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    angles = np.array([[-10.0], [10.0]], dtype=float)
    peaks = [[10.0, 350.0]]
    dihedrals = ["D0"]
    states_ua = []
    states_res = []
    flexible_ua = []
    flexible_res = []

    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        dt._assign_states(
            data_container=MagicMock(),
            group_id=0,
            molecules=[0, 1],
            level_list=["residue"],
            peaks_ua=[],
            peaks_res=peaks,
            states_ua=states_ua,
            states_res=states_res,
            flexible_ua=flexible_ua,
            flexible_res=flexible_res,
        )

    assert states_res[0] == ["1", "0", "1", "0"]
    assert flexible_res[0] == 1


def test_build_conformational_states_with_progress_handles_no_groups():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    progress = MagicMock()
    progress.add_task.return_value = 123

    states_ua, states_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels={},
        groups={},  # empty
        bin_width=30.0,
        progress=progress,
    )

    assert states_ua == {}
    assert states_res == []
    progress.add_task.assert_called_once()
    progress.update.assert_called_once_with(123, title="No groups")
    progress.advance.assert_called_once_with(123)


def test_build_conformational_states_with_progress_skips_empty_molecule_group():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    progress = MagicMock()
    progress.add_task.return_value = 5

    groups = {0: []}
    levels = {}

    states_ua, states_res, flex_ua, flex_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        bin_width=30.0,
        progress=progress,
    )

    assert states_ua == {}
    assert len(states_res) == 1
    assert flex_ua == {}
    assert flex_res == []
    progress.update.assert_called_with(5, title="Group 0 (empty)")
    progress.advance.assert_called_with(5)


def test_build_conformational_states_with_progress_updates_title_per_group(monkeypatch):
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    progress = MagicMock()
    progress.add_task.return_value = 9

    groups = {1: [7]}
    levels = {7: ["residue"]}

    uops.extract_fragment.return_value = MagicMock(trajectory=[0])

    dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        bin_width=30.0,
        progress=progress,
    )

    progress.update.assert_any_call(9, title="Group 1")
    progress.advance.assert_called_with(9)
