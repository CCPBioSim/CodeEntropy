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


def test_collect_dihedrals_for_group_handles_both_levels():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    mol = MagicMock()
    mol.residues = [MagicMock(), MagicMock()]

    with (
        patch.object(
            dt, "_select_heavy_residue", side_effect=["heavy0", "heavy1"]
        ) as sel_spy,
        patch.object(
            dt, "_get_dihedrals", side_effect=[["ua0"], ["ua1"], ["res_d0"]]
        ) as get_spy,
    ):
        ua, res = dt._collect_dihedrals_for_group(
            mol=mol, level_list=["united_atom", "residue"]
        )

    assert ua == [["ua0"], ["ua1"]]
    assert res == ["res_d0"]
    assert sel_spy.call_count == 2
    assert get_spy.call_count == 3


def test_collect_peaks_for_group_sets_empty_outputs_when_no_dihedrals():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    dihedrals_ua = [[], []]
    dihedrals_res = []

    with patch.object(dt, "_identify_peaks") as identify_spy:
        peaks_ua, peaks_res = dt._collect_peaks_for_group(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals_ua=dihedrals_ua,
            dihedrals_res=dihedrals_res,
            bin_width=30.0,
            start=0,
            end=10,
            step=1,
            level_list=["united_atom", "residue"],
        )

    assert peaks_ua == [[], []]
    assert peaks_res == []
    identify_spy.assert_not_called()


def test_identify_peaks_wraps_negative_angles_and_calls_find_histogram_peaks():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    angles = np.array([[-10.0], [10.0]], dtype=float)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with (
        patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral),
        patch.object(dt, "_find_histogram_peaks", return_value=[15.0]) as peaks_spy,
    ):
        out = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals=[MagicMock()],
            bin_width=180.0,
            start=0,
            end=2,
            step=1,
        )

    assert out == [[15.0]]
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

    angles = np.array([[5.0], [15.0]], dtype=float)
    peaks = [[5.0, 15.0]]

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        states = dt._assign_states(
            data_container=MagicMock(),
            molecules=[0, 1],
            dihedrals=["D0"],
            peaks=peaks,
            start=0,
            end=2,
            step=1,
        )

    assert states == ["0", "1", "0", "1"]


def test_assign_states_for_group_sets_empty_lists_and_delegates_for_nonempty():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    states_ua = {}
    states_res = [None, None]

    with patch.object(dt, "_assign_states", return_value=["x"]) as assign_spy:
        dt._assign_states_for_group(
            data_container=MagicMock(),
            group_id=1,
            molecules=[99],
            dihedrals_ua=[[], ["UA"]],
            peaks_ua=[[], [["p"]]],
            dihedrals_res=[],
            peaks_res=[],
            start=0,
            end=2,
            step=1,
            level_list=["united_atom", "residue"],
            states_ua=states_ua,
            states_res=states_res,
        )

    assert states_ua[(1, 0)] == []
    assert states_ua[(1, 1)] == ["x"]
    assert states_res[1] == []
    assert assign_spy.call_count == 1


def test_build_conformational_states_runs_group_and_skips_empty_group(monkeypatch):
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)
    monkeypatch.setattr(dt, "_progress_bar", _fake_progress_bar)

    groups = {0: [], 1: [7]}
    levels = {7: ["residue"]}

    uops.extract_fragment.return_value = MagicMock(trajectory=[0])

    monkeypatch.setattr(dt, "_collect_dihedrals_for_group", lambda **kw: ([], []))
    monkeypatch.setattr(dt, "_collect_peaks_for_group", lambda **kw: ([], []))
    monkeypatch.setattr(dt, "_assign_states_for_group", lambda **kw: None)

    states_ua, states_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        start=0,
        end=1,
        step=1,
        bin_width=30.0,
    )

    assert states_ua == {}
    assert len(states_res) == 2


def test_count_total_items_counts_all_levels_across_grouped_molecules():
    levels = {10: ["residue"], 11: ["united_atom", "residue"]}
    groups = {0: [10], 1: [11]}
    assert (
        ConformationStateBuilder._count_total_items(levels=levels, groups=groups) == 3
    )


def test_progress_bar_constructs_rich_progress_instance():
    prog = ConformationStateBuilder._progress_bar(total_items=1)
    assert hasattr(prog, "add_task")


def test_identify_peaks_handles_multiple_dihedrals_and_calls_histogram_each_time():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    angles = np.array(
        [
            [-10.0, 10.0],
            [20.0, -20.0],
        ],
        dtype=float,
    )

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with (
        patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral),
        patch(
            "CodeEntropy.levels.dihedrals.np.histogram", wraps=np.histogram
        ) as hist_spy,
    ):
        out = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals=["D0", "D1"],
            bin_width=180.0,
            start=0,
            end=2,
            step=1,
        )

    assert len(out) == 2
    assert hist_spy.call_count == 2


def test_assign_states_filters_out_empty_state_strings_when_no_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1, 2]
    uops.extract_fragment.return_value = mol

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=[]))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        out = dt._assign_states(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals=[],
            peaks=[],
            start=0,
            end=3,
            step=1,
        )

    assert out == []


def test_identify_peaks_multiple_molecules_real_histogram():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol0 = MagicMock()
    mol0.trajectory = [0, 1]
    mol1 = MagicMock()
    mol1.trajectory = [0, 1]

    uops.extract_fragment.side_effect = [mol0, mol1]

    angles = np.array([[10.0], [20.0]], dtype=float)

    class _FakeDihedral:
        def __init__(self, _):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        peaks = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0, 1],
            dihedrals=["D0"],
            bin_width=90.0,
            start=0,
            end=2,
            step=1,
        )

    assert len(peaks) == 1


def test_identify_peaks_real_histogram_without_spy():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    angles = np.array([[10.0], [20.0]], dtype=float)

    class _FakeDihedral:
        def __init__(self, _):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        peaks = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals=["D0"],
            bin_width=90.0,
            start=0,
            end=2,
            step=1,
        )

    assert isinstance(peaks, list)


def test_assign_states_for_group_residue_nonempty_calls_assign_states():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    states_ua = {}
    states_res = [None, None]

    with patch.object(dt, "_assign_states", return_value=["A"]) as spy:
        dt._assign_states_for_group(
            data_container=MagicMock(),
            group_id=1,
            molecules=[0],
            dihedrals_ua=[[]],
            peaks_ua=[[]],
            dihedrals_res=["D"],
            peaks_res=[["p"]],
            start=0,
            end=1,
            step=1,
            level_list=["residue"],
            states_ua=states_ua,
            states_res=states_res,
        )

    assert states_res[1] == ["A"]
    spy.assert_called_once()


def test_assign_states_first_empty_then_extend():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol0 = MagicMock()
    mol0.trajectory = []
    mol1 = MagicMock()
    mol1.trajectory = [0]

    uops.extract_fragment.side_effect = [mol0, mol1]

    angles = np.array([[10.0]], dtype=float)

    class _FakeDihedral:
        def __init__(self, _):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        states = dt._assign_states(
            data_container=MagicMock(),
            molecules=[0, 1],
            dihedrals=["D0"],
            peaks=[[10.0]],
            start=0,
            end=1,
            step=1,
        )

    assert states == ["0"]


def test_collect_peaks_for_group_calls_identify_peaks_for_ua_and_residue():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    dihedrals_ua = [["UA_D0"]]
    dihedrals_res = ["RES_D0"]

    with patch.object(
        dt,
        "_identify_peaks",
        side_effect=[[["ua_peak"]], [["res_peak"]]],
    ) as identify_spy:
        peaks_ua, peaks_res = dt._collect_peaks_for_group(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals_ua=dihedrals_ua,
            dihedrals_res=dihedrals_res,
            bin_width=30.0,
            start=0,
            end=10,
            step=1,
            level_list=["united_atom", "residue"],
        )

    assert peaks_ua == [[["ua_peak"]]]
    assert peaks_res == [["res_peak"]]
    assert identify_spy.call_count == 2


def test_assign_states_wraps_negative_angles():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.trajectory = [0, 1]
    uops.extract_fragment.return_value = mol

    angles = np.array([[-10.0], [10.0]], dtype=float)
    peaks = [[10.0, 350.0]]

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        states = dt._assign_states(
            data_container=MagicMock(),
            molecules=[0],
            dihedrals=["D0"],
            peaks=peaks,
            start=0,
            end=2,
            step=1,
        )

    assert states == ["1", "0"]


def test_build_conformational_states_with_progress_handles_no_groups():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    progress = MagicMock()
    progress.add_task.return_value = 123

    states_ua, states_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels={},
        groups={},  # empty
        start=0,
        end=1,
        step=1,
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

    states_ua, states_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        start=0,
        end=1,
        step=1,
        bin_width=30.0,
        progress=progress,
    )

    assert states_ua == {}
    assert len(states_res) == 1
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

    monkeypatch.setattr(dt, "_collect_dihedrals_for_group", lambda **kw: ([], []))
    monkeypatch.setattr(dt, "_collect_peaks_for_group", lambda **kw: ([], []))
    monkeypatch.setattr(dt, "_assign_states_for_group", lambda **kw: None)

    dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        start=0,
        end=1,
        step=1,
        bin_width=30.0,
        progress=progress,
    )

    progress.update.assert_any_call(9, title="Group 1")
    progress.advance.assert_called_with(9)
