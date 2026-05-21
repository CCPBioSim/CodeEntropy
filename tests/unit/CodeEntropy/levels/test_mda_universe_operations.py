from unittest.mock import MagicMock

import numpy as np
import pytest
from MDAnalysis.exceptions import NoDataError

from CodeEntropy.levels.mda import UniverseOperations


class _FakeAF:
    """Fake AnalysisFromFunction that avoids MDAnalysis trajectory requirements."""

    def __init__(self, func, atomgroup):
        self._func = func
        self._ag = atomgroup
        self.results = {}

    def run(self):
        self.results["timeseries"] = self._func(self._ag)
        return self


class _FakeTrajectory:
    def __init__(self, n_frames: int):
        self.n_frames = n_frames
        self.seen = []

    def __len__(self):
        return self.n_frames

    def __getitem__(self, frame_index):
        self.seen.append(frame_index)
        return frame_index


def test_extract_timeseries_unknown_kind_raises():
    ops = UniverseOperations()

    universe = MagicMock()
    universe.trajectory = _FakeTrajectory(0)

    ag = MagicMock()
    ag.universe = universe

    with pytest.raises(ValueError, match="Unknown timeseries kind"):
        ops._extract_timeseries(ag, kind="nope")


def test_extract_force_timeseries_fallback_to_positions_when_no_forces():
    ops = UniverseOperations()
    ag_force = MagicMock()

    def _extract(atomgroup, *, kind):
        if kind == "forces":
            raise NoDataError("no forces")
        return np.ones((2, 3, 3))

    ops._extract_timeseries = MagicMock(side_effect=_extract)

    out = ops._extract_force_timeseries_with_fallback(
        ag_force, fallback_to_positions_if_no_forces=True
    )
    assert out.shape == (2, 3, 3)


def test_extract_force_timeseries_raises_when_no_fallback():
    ops = UniverseOperations()
    ops._extract_timeseries = MagicMock(side_effect=NoDataError("no forces"))

    with pytest.raises(NoDataError):
        ops._extract_force_timeseries_with_fallback(
            MagicMock(), fallback_to_positions_if_no_forces=False
        )


def test_select_frames_defaults_start_end_and_slices(monkeypatch):
    ops = UniverseOperations()

    u = MagicMock()
    u.trajectory = _FakeTrajectory(10)
    u.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])

    select_atom = MagicMock()
    select_atom.universe = u
    select_atom.positions = np.zeros((2, 3))
    select_atom.forces = np.ones((2, 3))
    u.select_atoms.return_value = select_atom

    merged = MagicMock()
    merged.load_new = MagicMock()
    monkeypatch.setattr("CodeEntropy.levels.mda.mda.Merge", lambda ag: merged)

    out = ops.select_frames(u, start=None, end=None, step=2)

    assert out is merged
    u.select_atoms.assert_called_once_with("all", updating=True)
    assert u.trajectory.seen == [0, 2, 4, 6, 8]
    merged.load_new.assert_called_once()


def test_merge_forces_scales_kcal(monkeypatch):
    ops = UniverseOperations()

    u = MagicMock()
    u.select_atoms.return_value = MagicMock()
    u_force = MagicMock()
    u_force.select_atoms.return_value = MagicMock()

    monkeypatch.setattr(
        "CodeEntropy.levels.mda.mda.Universe", MagicMock(side_effect=[u, u_force])
    )

    ops._extract_timeseries = MagicMock(
        side_effect=[
            np.zeros((2, 2, 3)),  # coordinates
            np.zeros((2, 6)),  # dimensions
        ]
    )

    forces = np.ones((2, 2, 3), dtype=float)
    ops._extract_force_timeseries_with_fallback = MagicMock(return_value=forces)

    merged = MagicMock()
    merged.load_new = MagicMock()
    monkeypatch.setattr("CodeEntropy.levels.mda.mda.Merge", lambda ag: merged)

    out = ops.merge_forces(
        tprfile="tpr",
        trrfile="trr",
        forcefile="force.trr",
        fileformat=None,
        kcal=True,
    )

    assert out is merged
    assert np.allclose(forces, np.ones((2, 2, 3)) * 4.184)


def test_convert_lammps_transforms_forces_and_energies(monkeypatch):
    ops = UniverseOperations()

    mock_universe = MagicMock()
    transformations_captured = []

    def capture_universe(*args, **kwargs):
        if "transformations" in kwargs:
            transformations_captured.extend(kwargs["transformations"])
        return mock_universe

    monkeypatch.setattr("CodeEntropy.levels.mda.mda.Universe", capture_universe)

    ops.convert_lammps("tpr", "trr", "LAMMPSDUMP")

    ts = MagicMock()
    ts.forces = np.array([[1.0, 2.0, 3.0]], dtype=float)
    ts.data = {"c_5": np.array([1.0]), "c_7": np.array([2.0])}

    transformations_captured[0](ts)

    assert np.allclose(ts.forces, np.array([[1.0, 2.0, 3.0]], dtype=float) * 4.184)
    assert np.allclose(ts.data["c_5"], np.array([1.0], dtype=float) * 4.184)
    assert np.allclose(ts.data["c_7"], np.array([2.0], dtype=float) * 4.184)


def test_convert_lammps_fallback_on_keyerror(monkeypatch):
    ops = UniverseOperations()

    transformations_captured = []
    call_count = [0]

    def mock_universe(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise KeyError("c_5")
        if "transformations" in kwargs:
            transformations_captured.extend(kwargs["transformations"])
        return MagicMock()

    monkeypatch.setattr("CodeEntropy.levels.mda.mda.Universe", mock_universe)

    ops.convert_lammps("tpr", "trr", "LAMMPSDUMP")

    ts = MagicMock()
    ts.forces = np.array([[1.0, 2.0, 3.0]], dtype=float)

    transformations_captured[0](ts)

    assert np.allclose(ts.forces, np.array([[1.0, 2.0, 3.0]], dtype=float) * 4.184)
    assert call_count[0] == 2


def test_select_atoms_builds_merged_universe_and_loads_timeseries(monkeypatch):
    ops = UniverseOperations()

    u = MagicMock()
    u.trajectory = _FakeTrajectory(2)

    sel = MagicMock()
    sel.universe = u
    sel.positions = np.zeros((3, 3))
    sel.forces = np.ones((3, 3))

    u.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    u.select_atoms.return_value = sel

    merged = MagicMock()
    merged.load_new = MagicMock()

    monkeypatch.setattr("CodeEntropy.levels.mda.mda.Merge", lambda ag: merged)

    out = ops.select_atoms(u, "name CA")

    assert out is merged
    u.select_atoms.assert_called_once_with("name CA", updating=True)
    merged.load_new.assert_called_once()

    coordinates = merged.load_new.call_args.args[0]
    assert coordinates.shape == (2, 3, 3)


def test_extract_fragment_selects_by_resindices(monkeypatch):
    u = MagicMock()
    frag0 = MagicMock()
    frag0.indices = np.array([10, 11, 12], dtype=int)

    u.atoms.fragments = [frag0]

    uops = UniverseOperations()

    select_spy = MagicMock(return_value="FRAG")
    monkeypatch.setattr(uops, "select_atoms", select_spy)

    out = uops.extract_fragment(u, molecule_id=0)

    assert out == "FRAG"
    select_spy.assert_called_once_with(u, "index 10:12")


def test_extract_timeseries_kind_positions_returns_xyz_array():
    uops = UniverseOperations()

    universe = MagicMock()
    universe.trajectory = _FakeTrajectory(2)

    ag = MagicMock()
    ag.universe = universe
    ag.positions = np.array([[1.0, 2.0, 3.0]], dtype=float)

    out = uops._extract_timeseries(atomgroup=ag, kind="positions")

    assert out.shape == (2, 1, 3)
    assert np.allclose(out[0], np.array([[1.0, 2.0, 3.0]]))
    assert universe.trajectory.seen == [0, 1]


def test_extract_timeseries_invalid_kind_raises_value_error():
    uops = UniverseOperations()

    universe = MagicMock()
    universe.trajectory = _FakeTrajectory(0)

    ag = MagicMock()
    ag.universe = universe

    with pytest.raises(ValueError, match="Unknown timeseries kind"):
        uops._extract_timeseries(atomgroup=ag, kind="not-a-kind")


def test_extract_timeseries_forces_branch_uses_forces_copy():
    uops = UniverseOperations()

    universe = MagicMock()
    universe.trajectory = _FakeTrajectory(2)

    ag = MagicMock()
    ag.universe = universe
    ag.forces = np.array([[1.0, 2.0, 3.0]], dtype=float)

    out = uops._extract_timeseries(ag, kind="forces")

    assert out.shape == (2, 1, 3)
    assert np.allclose(out[0], ag.forces)
    assert universe.trajectory.seen == [0, 1]


def test_extract_timeseries_dimensions_branch_uses_dimensions_copy():
    uops = UniverseOperations()

    universe = MagicMock()
    universe.trajectory = _FakeTrajectory(2)
    universe.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90], dtype=float)

    ag = MagicMock()
    ag.universe = universe

    out = uops._extract_timeseries(ag, kind="dimensions")

    assert out.shape == (2, 6)
    assert np.allclose(out[0], universe.dimensions)
    assert universe.trajectory.seen == [0, 1]
