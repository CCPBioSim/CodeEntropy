from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from CodeEntropy.levels.dihedrals import (
    ConformationStateBuilder,
    ConformationStateData,
    DihedralAngleData,
    DihedralPeakData,
)
from CodeEntropy.trajectory.frames import FrameSelection


class _AddableAG:
    """Minimal addable AtomGroup test double."""

    def __init__(self, name: str):
        """Initialize the fake AtomGroup.

        Args:
            name: Human-readable identifier used in composed names.
        """
        self.name = name

    def __add__(self, other: _AddableAG) -> _AddableAG:
        """Return a composed fake AtomGroup.

        Args:
            other: Fake AtomGroup to combine with this object.

        Returns:
            New fake AtomGroup containing a composed name.
        """
        return _AddableAG(f"({self.name}+{other.name})")


def _make_frame_selection(
    start: int = 0,
    stop: int = 2,
    step: int = 1,
) -> FrameSelection:
    """Build a FrameSelection for dihedral unit tests.

    Args:
        start: Inclusive source-frame start.
        stop: Exclusive source-frame stop.
        step: Source-frame step.

    Returns:
        FrameSelection covering the requested bounds.
    """
    return FrameSelection.from_bounds(start=start, stop=stop, step=step)


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


def test_collect_dihedral_angle_data_sets_empty_outputs_when_no_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=[])

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    angle_data = dt._collect_dihedral_angle_data(
        data_container=MagicMock(),
        molecules=[0],
        level_list=["united_atom", "residue"],
        frame_selection=frame_selection,
    )

    assert angle_data.num_residues == 1
    assert angle_data.num_dihedrals_ua == [0]
    assert angle_data.num_dihedrals_res == 0
    assert angle_data.phi_ua == {0: []}
    assert angle_data.phi_res == []


def test_collect_dihedral_angle_data_wraps_negative_angles():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    dihedrals = ["D0"]
    angles = np.array([[-10.0], [10.0]], dtype=float)

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        angle_data = dt._collect_dihedral_angle_data(
            data_container=MagicMock(),
            molecules=[0],
            level_list=["united_atom", "residue"],
            frame_selection=frame_selection,
        )

    assert angle_data.phi_ua[0][0] == [350.0, 10.0]
    assert angle_data.phi_res[0] == [350.0, 10.0]


def test_build_peak_data_returns_empty_outputs_when_no_angles():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    angle_data = DihedralAngleData(
        num_residues=1,
        num_dihedrals_ua=[0],
        num_dihedrals_res=0,
        phi_ua={0: []},
        phi_res=[],
    )

    peak_data = dt._build_peak_data(
        angle_data=angle_data,
        level_list=["united_atom", "residue"],
        bin_width=30.0,
    )

    assert peak_data == DihedralPeakData(peaks_ua=[[]], peaks_res=[])


def test_build_peak_data_calls_process_histogram_for_ua_and_residue():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    angle_data = DihedralAngleData(
        num_residues=1,
        num_dihedrals_ua=[1],
        num_dihedrals_res=1,
        phi_ua={0: {0: [10.0, 20.0]}},
        phi_res={0: [30.0, 40.0]},
    )

    dt._process_histogram = MagicMock(side_effect=[["ua_peak"], ["res_peak"]])

    peak_data = dt._build_peak_data(
        angle_data=angle_data,
        level_list=["united_atom", "residue"],
        bin_width=30.0,
    )

    assert peak_data.peaks_ua == [["ua_peak"]]
    assert peak_data.peaks_res == ["res_peak"]
    assert dt._process_histogram.call_args_list == [
        call(
            num_dihedrals=1,
            phi_values={0: [10.0, 20.0]},
            bin_width=30.0,
        ),
        call(
            num_dihedrals=1,
            phi_values={0: [30.0, 40.0]},
            bin_width=30.0,
        ),
    ]


def test_identify_peaks_delegates_to_angle_collection_and_peak_building():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    angle_data = DihedralAngleData(
        num_residues=1,
        num_dihedrals_ua=[1],
        num_dihedrals_res=1,
        phi_ua={0: {0: [10.0]}},
        phi_res={0: [10.0]},
    )
    peak_data = DihedralPeakData(peaks_ua=[[[10.0]]], peaks_res=[[10.0]])

    dt._collect_dihedral_angle_data = MagicMock(return_value=angle_data)
    dt._build_peak_data = MagicMock(return_value=peak_data)

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    peaks_ua, peaks_res = dt._identify_peaks(
        data_container="universe",
        molecules=[0],
        bin_width=30.0,
        level_list=["united_atom", "residue"],
        frame_selection=frame_selection,
    )

    assert peaks_ua == peak_data.peaks_ua
    assert peaks_res == peak_data.peaks_res
    dt._collect_dihedral_angle_data.assert_called_once_with(
        data_container="universe",
        molecules=[0],
        level_list=["united_atom", "residue"],
        frame_selection=frame_selection,
    )
    dt._build_peak_data.assert_called_once_with(
        angle_data=angle_data,
        level_list=["united_atom", "residue"],
        bin_width=30.0,
    )


def test_identify_peaks_wraps_negative_angles_and_calls_process_histogram():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    dihedrals = ["D0"]
    angles = np.array([[-10.0], [10.0]], dtype=float)

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with (
        patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral),
        patch.object(dt, "_process_histogram", return_value=[15.0]) as peaks_spy,
    ):
        out_ua, out_res = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            bin_width=10.0,
            level_list=["united_atom", "residue"],
            frame_selection=frame_selection,
        )

    assert out_ua == [[15.0]]
    assert out_res == [15.0]
    assert peaks_spy.call_count == 2


def test_find_histogram_peaks_hits_interior_and_wraparound_last_bin():
    popul = [0, 2, 0, 3]
    bin_value = [10.0, 20.0, 30.0, 40.0]
    assert ConformationStateBuilder._find_histogram_peaks(popul, bin_value) == [
        20.0,
        40.0,
    ]


def test_calculate_group_state_data_initialises_then_extends_for_multiple_molecules():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    dihedrals = ["D0"]
    angles = np.array([[5.0], [15.0]], dtype=float)
    peaks = [[5.0, 15.0]]

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        state_data = dt._calculate_group_state_data(
            data_container=MagicMock(),
            group_id=0,
            molecules=[0, 1],
            level_list=["united_atom", "residue"],
            peaks_ua=[peaks],
            peaks_res=peaks,
            frame_selection=frame_selection,
        )

    assert state_data.states_ua_updates[(0, 0)] == ["0", "1", "0", "1"]
    assert state_data.flexible_ua_updates[(0, 0)] == 1
    assert state_data.state_res == ["0", "1", "0", "1"]
    assert state_data.flex_res == 1


def test_merge_group_state_data_initialises_final_accumulators():
    states_ua = {}
    states_res = []
    flexible_ua = {}
    flexible_res = []

    state_data = ConformationStateData(
        state_res=["0", "1"],
        flex_res=1,
        states_ua_updates={(0, 0): ["0", "1"]},
        flexible_ua_updates={(0, 0): 1},
    )

    ConformationStateBuilder._merge_group_state_data(
        state_data=state_data,
        states_ua=states_ua,
        states_res=states_res,
        flexible_ua=flexible_ua,
        flexible_res=flexible_res,
    )

    assert states_ua == {(0, 0): ["0", "1"]}
    assert states_res == [["0", "1"]]
    assert flexible_ua == {(0, 0): 1}
    assert flexible_res == [1]


def test_merge_group_state_data_extends_existing_ua_states():
    states_ua = {(0, 0): ["0"]}
    states_res = []
    flexible_ua = {(0, 0): 1}
    flexible_res = []

    state_data = ConformationStateData(
        state_res=["1"],
        flex_res=0,
        states_ua_updates={(0, 0): ["1"], (0, 1): ["2"]},
        flexible_ua_updates={(0, 0): 2, (0, 1): 0},
    )

    ConformationStateBuilder._merge_group_state_data(
        state_data=state_data,
        states_ua=states_ua,
        states_res=states_res,
        flexible_ua=flexible_ua,
        flexible_res=flexible_res,
    )

    assert states_ua[(0, 0)] == ["0", "1"]
    assert states_ua[(0, 1)] == ["2"]
    assert flexible_ua[(0, 0)] == 2
    assert flexible_ua[(0, 1)] == 0
    assert states_res == [["1"]]
    assert flexible_res == [0]


def test_assign_states_delegates_to_calculation_and_merge():
    dt = ConformationStateBuilder(universe_operations=MagicMock())

    state_data = ConformationStateData(
        state_res=["0"],
        flex_res=1,
        states_ua_updates={(0, 0): ["0"]},
        flexible_ua_updates={(0, 0): 1},
    )
    dt._calculate_group_state_data = MagicMock(return_value=state_data)
    dt._merge_group_state_data = MagicMock()

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)
    states_ua = {}
    states_res = []
    flexible_ua = {}
    flexible_res = []

    dt._assign_states(
        data_container="universe",
        group_id=0,
        molecules=[0],
        level_list=["united_atom"],
        peaks_ua=[[[10.0]]],
        peaks_res=[],
        states_ua=states_ua,
        states_res=states_res,
        flexible_ua=flexible_ua,
        flexible_res=flexible_res,
        frame_selection=frame_selection,
    )

    dt._calculate_group_state_data.assert_called_once_with(
        data_container="universe",
        group_id=0,
        molecules=[0],
        level_list=["united_atom"],
        peaks_ua=[[[10.0]]],
        peaks_res=[],
        frame_selection=frame_selection,
    )
    dt._merge_group_state_data.assert_called_once_with(
        state_data=state_data,
        states_ua=states_ua,
        states_res=states_res,
        flexible_ua=flexible_ua,
        flexible_res=flexible_res,
    )


def test_assign_states_initialises_then_extends_for_multiple_molecules():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    dihedrals = ["D0"]
    angles = np.array([[5.0], [15.0]], dtype=float)
    peaks = [[5.0, 15.0]]

    states_ua = {}
    states_res = []
    flexible_ua = {}
    flexible_res = []

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        dt._assign_states(
            data_container=MagicMock(),
            group_id=0,
            molecules=[0, 1],
            level_list=["united_atom", "residue"],
            peaks_ua=[peaks],
            peaks_res=peaks,
            states_ua=states_ua,
            states_res=states_res,
            flexible_ua=flexible_ua,
            flexible_res=flexible_res,
            frame_selection=frame_selection,
        )

    assert states_ua[(0, 0)] == ["0", "1", "0", "1"]
    assert flexible_ua[(0, 0)] == 1
    assert states_res[0] == ["0", "1", "0", "1"]
    assert flexible_res[0] == 1


def test_build_conformational_states_runs_group_and_skips_empty_group():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    groups = {0: [], 1: [7]}
    levels = {7: ["residue"]}

    dt._identify_peaks = MagicMock(return_value=([], []))
    dt._assign_states = MagicMock()

    frame_selection = _make_frame_selection(start=0, stop=1, step=1)

    states_ua, states_res, flex_ua, flex_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        bin_width=30.0,
        frame_selection=frame_selection,
    )

    assert states_ua == {}
    assert states_res == [[], []]
    assert flex_ua == {}
    assert flex_res == []

    dt._identify_peaks.assert_called_once()
    dt._assign_states.assert_called_once()
    assert dt._identify_peaks.call_args.kwargs["frame_selection"] is frame_selection
    assert dt._assign_states.call_args.kwargs["frame_selection"] is frame_selection


def test_identify_peaks_handles_multiple_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    dihedrals = ["D0", "D1"]
    angles = np.array(
        [
            [-10.0, 10.0],
            [20.0, -20.0],
        ],
        dtype=float,
    )

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=dihedrals)
    dt._process_histogram = MagicMock(return_value=[1, 2])

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        out_ua, out_res = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0],
            bin_width=30.0,
            level_list=["united_atom", "residue"],
            frame_selection=frame_selection,
        )

    assert out_ua == [[1, 2]]
    assert out_res == [1, 2]
    assert dt._process_histogram.call_count == 2


def test_collect_dihedral_angle_data_initialises_phi_res_dict_before_processing():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    uops.extract_fragment.return_value = mol

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    dihedrals = ["D0"]
    dihedral_results = MagicMock()
    processed_phi = {0: [10.0, 20.0]}

    dt._get_dihedrals = MagicMock(return_value=dihedrals)
    dt._run_dihedrals = MagicMock(return_value=dihedral_results)
    dt._process_dihedral_phi = MagicMock(return_value=processed_phi)

    angle_data = dt._collect_dihedral_angle_data(
        data_container=MagicMock(),
        molecules=[0],
        level_list=["residue"],
        frame_selection=frame_selection,
    )

    assert angle_data.num_residues == 4
    assert angle_data.phi_res == processed_phi
    assert angle_data.num_dihedrals_res == 1

    dt._process_dihedral_phi.assert_called_once_with(
        dihedral_results=dihedral_results,
        num_dihedrals=1,
        number_frames=2,
        phi_values={},
    )


def test_identify_peaks_initialises_phi_res_dict_before_processing_residue_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    uops.extract_fragment.return_value = mol

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    dihedrals = ["D0"]
    dihedral_results = MagicMock()
    processed_phi = {0: [10.0, 20.0]}

    dt._get_dihedrals = MagicMock(return_value=dihedrals)
    dt._run_dihedrals = MagicMock(return_value=dihedral_results)
    dt._process_dihedral_phi = MagicMock(return_value=processed_phi)
    dt._process_histogram = MagicMock(return_value=[[15.0]])

    peaks_ua, peaks_res = dt._identify_peaks(
        data_container=MagicMock(),
        molecules=[0],
        bin_width=30.0,
        level_list=["residue"],
        frame_selection=frame_selection,
    )

    assert peaks_ua == [[], [], [], []]
    assert peaks_res == [[15.0]]

    dt._process_dihedral_phi.assert_called_once_with(
        dihedral_results=dihedral_results,
        num_dihedrals=1,
        number_frames=2,
        phi_values={},
    )


def test_assign_states_filters_out_empty_state_strings_when_no_dihedrals():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    states_ua = {}
    states_res = []
    flexible_ua = {}
    flexible_res = []

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=[])

    frame_selection = _make_frame_selection(start=0, stop=3, step=1)

    dt._assign_states(
        data_container=MagicMock(),
        group_id=0,
        molecules=[0],
        level_list=["united_atom", "residue"],
        peaks_ua=[],
        peaks_res=[],
        states_ua=states_ua,
        states_res=states_res,
        flexible_ua=flexible_ua,
        flexible_res=flexible_res,
        frame_selection=frame_selection,
    )

    assert states_ua[(0, 0)] == []
    assert flexible_ua[(0, 0)] == 0
    assert states_res[0] == []
    assert flexible_res[0] == 0


def test_identify_peaks_multiple_molecules_real_histogram():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol0 = MagicMock()
    mol0.residues = [MagicMock()]
    mol0.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)

    mol1 = MagicMock()
    mol1.residues = [MagicMock()]
    mol1.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)

    uops.extract_fragment.side_effect = [mol0, mol0, mol1]

    dihedrals = ["D0"]
    angles = np.array([[10.0], [20.0]], dtype=float)

    dt._select_heavy_residue = MagicMock(return_value=mol0)
    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        peaks_ua, peaks_res = dt._identify_peaks(
            data_container=MagicMock(),
            molecules=[0, 1],
            bin_width=90.0,
            level_list=["united_atom", "residue"],
            frame_selection=frame_selection,
        )

    assert len(peaks_ua) == 1
    assert len(peaks_res) == 1


def test_assign_states_wraps_negative_angles():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    mol = MagicMock()
    mol.residues = [MagicMock()]
    mol.residues[0].atoms.indices = np.array([0, 1, 2, 3], dtype=int)
    uops.extract_fragment.return_value = mol

    angles = np.array([[-10.0], [10.0]], dtype=float)
    peaks = [[10.0, 350.0]]
    dihedrals = ["D0"]

    states_ua = {}
    states_res = []
    flexible_ua = {}
    flexible_res = []

    dt._select_heavy_residue = MagicMock(return_value=mol)
    dt._get_dihedrals = MagicMock(return_value=dihedrals)

    class _FakeDihedral:
        def __init__(self, _dihedrals):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(angles=angles))

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with patch("CodeEntropy.levels.dihedrals.Dihedral", _FakeDihedral):
        dt._assign_states(
            data_container=MagicMock(),
            group_id=0,
            molecules=[0, 1],
            level_list=["united_atom", "residue"],
            peaks_ua=[peaks],
            peaks_res=peaks,
            states_ua=states_ua,
            states_res=states_res,
            flexible_ua=flexible_ua,
            flexible_res=flexible_res,
            frame_selection=frame_selection,
        )

    assert states_ua[(0, 0)] == ["1", "0", "1", "0"]
    assert flexible_ua[(0, 0)] == 1
    assert states_res[0] == ["1", "0", "1", "0"]
    assert flexible_res[0] == 1


def test_build_conformational_states_with_progress_handles_no_groups():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    progress = MagicMock()
    progress.add_task.return_value = 123

    frame_selection = _make_frame_selection(start=0, stop=1, step=1)

    states_ua, states_res, flex_ua, flex_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels={},
        groups={},
        bin_width=30.0,
        frame_selection=frame_selection,
        progress=progress,
    )

    assert states_ua == {}
    assert states_res == []
    assert flex_ua == {}
    assert flex_res == []

    progress.add_task.assert_called_once()
    progress.update.assert_called_once_with(123, title="No groups")
    progress.advance.assert_called_once_with(123)


def test_build_conformational_states_with_progress_skips_empty_molecule_group():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    progress = MagicMock()
    progress.add_task.return_value = 5

    frame_selection = _make_frame_selection(start=0, stop=1, step=1)

    states_ua, states_res, flex_ua, flex_res = dt.build_conformational_states(
        data_container=MagicMock(),
        levels={},
        groups={0: []},
        bin_width=30.0,
        frame_selection=frame_selection,
        progress=progress,
    )

    assert states_ua == {}
    assert states_res == [[]]
    assert flex_ua == {}
    assert flex_res == []

    progress.update.assert_called_with(5, title="Group 0 (empty)")
    progress.advance.assert_called_with(5)


def test_build_conformational_states_with_progress_updates_title_per_group():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    progress = MagicMock()
    progress.add_task.return_value = 9

    groups = {1: [7]}
    levels = {7: ["residue"]}

    dt._identify_peaks = MagicMock(return_value=([], []))
    dt._assign_states = MagicMock()

    frame_selection = _make_frame_selection(start=0, stop=1, step=1)

    dt.build_conformational_states(
        data_container=MagicMock(),
        levels=levels,
        groups=groups,
        bin_width=30.0,
        frame_selection=frame_selection,
        progress=progress,
    )

    progress.update.assert_any_call(9, title="Group 1")
    progress.advance.assert_called_with(9)
    assert dt._identify_peaks.call_args.kwargs["frame_selection"] is frame_selection
    assert dt._assign_states.call_args.kwargs["frame_selection"] is frame_selection


def test_process_dihedral_phi():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    dihedral_results = MagicMock()
    dihedral_results.results.angles = [[0, 1, 2], [3, 4, 5]]
    num_dihedrals = 3
    number_frames = 2
    phi_values = {}

    phi_values = dt._process_dihedral_phi(
        dihedral_results, num_dihedrals, number_frames, phi_values
    )

    assert len(phi_values) == 3
    assert phi_values[0] == [0, 3]


def test_process_dihedral_phi_negative():
    uops = MagicMock()
    dt = ConformationStateBuilder(universe_operations=uops)

    dihedral_results = MagicMock()
    dihedral_results.results.angles = [[0, 1, 2], [-3, 4, 5]]
    num_dihedrals = 3
    number_frames = 2
    phi_values = {}

    phi_values = dt._process_dihedral_phi(
        dihedral_results, num_dihedrals, number_frames, phi_values
    )

    assert len(phi_values) == 3
    assert phi_values[0] == [0, 357]


def test_run_dihedrals_raises_when_no_dihedrals():
    dt = ConformationStateBuilder(universe_operations=MagicMock())
    frame_selection = _make_frame_selection(start=0, stop=2, step=1)

    with pytest.raises(
        ValueError, match="Cannot run Dihedral analysis with no dihedrals"
    ):
        dt._run_dihedrals(
            dihedrals=[],
            frame_selection=frame_selection,
        )


def test_analysis_run_bounds_raises_when_frame_selection_empty():
    frame_selection = FrameSelection(indices=())

    with pytest.raises(ValueError, match="Frame selection is empty"):
        ConformationStateBuilder._analysis_run_bounds(frame_selection)
