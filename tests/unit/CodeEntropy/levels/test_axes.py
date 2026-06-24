from unittest.mock import MagicMock

import numpy as np
import pytest

from CodeEntropy.levels.axes import AxesCalculator
from CodeEntropy.levels.nodes.axes_topology import UAAxesTopology


class _FakeAtom:
    def __init__(self, index: int, mass: float, position):
        self.index = int(index)
        self.mass = float(mass)
        self.position = np.asarray(position, dtype=float)

    def __add__(self, other):
        # atom + atomgroup => atomgroup
        if isinstance(other, _FakeAtomGroup):
            return _FakeAtomGroup([self] + list(other._atoms))
        if isinstance(other, _FakeAtom):
            return _FakeAtomGroup([self, other])
        raise TypeError(f"Unsupported add: _FakeAtom + {type(other)}")


class _FakeAtomGroup:
    def __init__(self, atoms, positions=None, select_map=None):
        self._atoms = list(atoms)
        self._select_map = dict(select_map or {})

        if positions is None:
            if self._atoms:
                self.positions = np.vstack([a.position for a in self._atoms]).astype(
                    float
                )
            else:
                self.positions = np.zeros((0, 3), dtype=float)
        else:
            self.positions = np.asarray(positions, dtype=float)

    def __len__(self):
        return len(self._atoms)

    def __iter__(self):
        return iter(self._atoms)

    def __getitem__(self, idx):
        return self._atoms[idx]

    @property
    def masses(self):
        return np.asarray([a.mass for a in self._atoms], dtype=float)

    def select_atoms(self, query: str):
        return self._select_map.get(query, _FakeAtomGroup([]))

    def __add__(self, other):
        if isinstance(other, _FakeAtomGroup):
            return _FakeAtomGroup(self._atoms + other._atoms)
        if isinstance(other, _FakeAtom):
            return _FakeAtomGroup(self._atoms + [other])
        raise TypeError(f"Unsupported add: _FakeAtomGroup + {type(other)}")


def _atom(index=0, mass=12.0, pos=(0.0, 0.0, 0.0), resindex=0):
    a = MagicMock()
    a.index = index
    a.mass = mass
    a.position = np.array(pos, dtype=float)
    a.resindex = resindex
    return a


def test_get_residue_axes_empty_residue_raises():
    ax = AxesCalculator()
    u = MagicMock()
    u.select_atoms.return_value = []

    with pytest.raises(ValueError):
        ax.get_residue_axes(u, index=5)


def test_get_residue_axes_no_bonds_uses_custom_principal_axes(monkeypatch):
    ax = AxesCalculator()

    # residue selection: non-empty, has heavy atoms and positions
    residue = MagicMock()
    residue.__len__.return_value = 1
    residue.atoms.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
    residue.select_atoms.return_value = MagicMock(positions=np.zeros((2, 3)))
    residue.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])

    # atom_set empty => "no bonds to other residues" branch
    def _select_atoms(q):
        if q.startswith("(resindex"):
            return []
        if q.startswith("resindex "):
            return residue
        return []

    u.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(ax, "get_UA_masses", lambda mol: [10.0, 12.0])
    monkeypatch.setattr(ax, "get_moment_of_inertia_tensor", lambda **kwargs: np.eye(3))
    monkeypatch.setattr(
        ax,
        "get_custom_principal_axes",
        lambda moi: (np.eye(3), np.array([3.0, 2.0, 1.0])),
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=7)

    assert np.allclose(trans, np.eye(3))
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(center, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(moi, np.array([3.0, 2.0, 1.0]))


def test_get_residue_axes_with_bonds_uses_vanilla_axes(monkeypatch):
    ax = AxesCalculator()

    residue = MagicMock()
    residue.__len__.return_value = 1
    residue.atoms.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])
    residue.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])

    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    u.atoms.principal_axes.return_value = np.eye(3)

    # atom_set non-empty => bonded branch
    def _select_atoms(q):
        if q.startswith("(resindex"):
            return [1]  # non-empty
        if q.startswith("resindex "):
            return residue
        return []

    u.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    monkeypatch.setattr(
        ax, "get_vanilla_axes", lambda mol: (np.eye(3) * 2, np.array([9.0, 8.0, 7.0]))
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=10)

    assert np.allclose(trans, np.eye(3))
    assert np.allclose(rot, np.eye(3) * 2)
    assert np.allclose(moi, np.array([9.0, 8.0, 7.0]))


def test_get_UA_axes_uses_principal_axes_when_single_heavy(monkeypatch):
    ax = AxesCalculator()

    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    u.atoms.principal_axes.return_value = np.eye(3)

    # heavy_atoms length <= 1 => principal_axes path
    heavy_atom = MagicMock(index=5)
    heavy_atoms = [heavy_atom]

    def _sel(q):
        if q == "prop mass > 1.1":
            return heavy_atoms
        if q.startswith("index "):
            # return atom group with positions
            ag = MagicMock()
            ag.positions = np.array([[4.0, 0.0, 0.0]])
            ag.__getitem__.return_value = MagicMock(
                mass=12.0, position=np.array([4.0, 0.0, 0.0]), index=5
            )
            return ag
        return []

    u.select_atoms.side_effect = _sel

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), np.array([1.0, 2.0, 3.0])),
    )

    trans, rot, center, moi = ax.get_UA_axes(u, index=0)

    assert np.allclose(trans, np.eye(3))
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(center, np.array([4.0, 0.0, 0.0]))
    assert np.allclose(moi, np.array([1.0, 2.0, 3.0]))


def test_get_UA_axes_raises_when_bonded_axes_fail(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])

    heavy_atom = MagicMock(index=5)
    heavy_atoms = [heavy_atom]

    def _sel(q):
        if q == "prop mass > 1.1":
            return heavy_atoms
        if q.startswith("index "):
            ag = MagicMock()
            ag.positions = np.array([[1.0, 1.0, 1.0]])
            ag.__getitem__.return_value = MagicMock(
                mass=12.0, position=np.array([1.0, 1.0, 1.0]), index=5
            )
            return ag
        return []

    u.select_atoms.side_effect = _sel
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    monkeypatch.setattr(ax, "get_bonded_axes", lambda **kwargs: (None, None))

    with pytest.raises(ValueError):
        ax.get_UA_axes(u, index=0)


def test_get_custom_axes_degenerate_axis1_raises():
    ax = AxesCalculator()
    a = np.zeros(3)
    b_list = [np.zeros(3)]
    with pytest.raises(ValueError):
        ax.get_custom_axes(
            a=a, b_list=b_list, c=np.zeros(3), dimensions=np.array([10.0, 10.0, 10.0])
        )


def test_get_custom_axes_normalizes_and_uses_bc_when_multiple_b(monkeypatch):
    ax = AxesCalculator()
    a = np.array([0.0, 0.0, 0.0])
    b_list = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    c = np.array([0.0, 1.0, 0.0])

    axes = ax.get_custom_axes(
        a=a, b_list=b_list, c=c, dimensions=np.array([10.0, 10.0, 10.0])
    )
    assert axes.shape == (3, 3)
    # axes rows must be unit length
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)


def test_get_custom_moment_of_inertia_two_atom_sets_smallest_to_zero():
    ax = AxesCalculator()

    a0 = _FakeAtom(0, 12.0, [0.0, 0.0, 0.0])
    a1 = _FakeAtom(1, 1.0, [1.0, 0.0, 0.0])
    ua = _FakeAtomGroup([a0, a1])

    axes = np.eye(3)
    moi = ax.get_custom_moment_of_inertia(
        UA=ua,
        custom_rotation_axes=axes,
        center_of_mass=np.array([0.0, 0.0, 0.0]),
        dimensions=np.array([10.0, 10.0, 10.0]),
    )
    assert moi.shape == (3,)
    assert np.isclose(np.min(moi), 0.0)


def test_get_flipped_axes_flips_negative_dot():
    ax = AxesCalculator()

    a0 = _FakeAtom(0, 12.0, [0.0, 0.0, 0.0])
    ua = _FakeAtomGroup([a0])

    # axis0 points opposite to rr_axis -> should flip
    custom_axes = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    flipped = ax.get_flipped_axes(
        UA=ua,
        custom_axes=custom_axes,
        center_of_mass=np.array([1.0, 0.0, 0.0]),
        dimensions=np.array([10.0, 10.0, 10.0]),
    )
    assert np.allclose(flipped[0], np.array([1.0, 0.0, 0.0]))


def test_get_custom_principal_axes_flips_z_when_left_handed():
    ax = AxesCalculator()
    moi = np.eye(3)
    axes, vals = ax.get_custom_principal_axes(moi)
    assert axes.shape == (3, 3)
    assert vals.shape == (3,)


def test_get_UA_masses_sums_bonded_hydrogens():
    ax = AxesCalculator()

    heavy = _FakeAtom(index=0, mass=12.0, position=[0, 0, 0])
    h1 = _FakeAtom(index=1, mass=1.0, position=[1, 0, 0])
    h2 = _FakeAtom(index=2, mass=1.0, position=[0, 1, 0])

    bonded_atoms = _FakeAtomGroup(
        [h1, h2], select_map={"mass 1 to 1.1": _FakeAtomGroup([h1, h2])}
    )

    mol = _FakeAtomGroup(
        [heavy, h1, h2],
        select_map={
            "bonded index 0": bonded_atoms,
        },
    )

    masses = ax.get_UA_masses(mol)
    assert masses == [14.0]


def test_get_vanilla_axes_sorts_eigenvalues_desc_by_abs(monkeypatch):
    ax = AxesCalculator()
    mol = MagicMock()
    moi_tensor = np.diag([1.0, -10.0, 3.0])
    mol.moment_of_inertia.return_value = moi_tensor
    mol.principal_axes.return_value = np.eye(3)
    mol.atoms = MagicMock()

    # avoid real MDAnalysis unwrap
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)

    axes, moments = ax.get_vanilla_axes(mol)

    assert axes.shape == (3, 3)
    # sorted by abs descending => -10, 3, 1
    assert np.allclose(moments, np.array([-10.0, 3.0, 1.0]))


def test_find_bonded_atoms_selects_heavy_and_hydrogen_groups():
    ax = AxesCalculator()

    bonded = MagicMock()
    heavy = MagicMock()
    hyd = MagicMock()
    bonded.select_atoms.side_effect = [heavy, hyd]

    system = MagicMock()
    system.select_atoms.return_value = bonded

    out_heavy, out_h = ax.find_bonded_atoms(atom_idx=7, system=system)

    system.select_atoms.assert_called_once_with("bonded index 7")
    bonded.select_atoms.assert_any_call("mass 2 to 999")
    bonded.select_atoms.assert_any_call("mass 1 to 1.1")
    assert out_heavy is heavy
    assert out_h is hyd


def test_get_bonded_axes_non_heavy_returns_none():
    ax = AxesCalculator()
    system = MagicMock()
    atom = _atom(index=1, mass=1.0)

    out_axes, out_moi = ax.get_bonded_axes(
        system, atom, dimensions=np.array([10.0, 10.0, 10.0])
    )
    assert out_axes is None
    assert out_moi is None


def test_get_bonded_axes_case1_uses_vanilla_axes_and_returns_flipped(monkeypatch):
    ax = AxesCalculator()
    system = MagicMock()
    atom = _atom(index=1, mass=12.0, pos=(1, 0, 0))

    heavy = _FakeAtomGroup([])  # len == 0 -> case1
    hyd = _FakeAtomGroup([_atom(index=2, mass=1.0)])
    monkeypatch.setattr(ax, "find_bonded_atoms", lambda _idx, _sys: (heavy, hyd))

    monkeypatch.setattr(
        ax, "get_vanilla_axes", lambda _ag: (np.eye(3) * 7, np.array([1.0, 2.0, 3.0]))
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes * -1)

    out_axes, out_moi = ax.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

    assert np.allclose(out_axes, -np.eye(3) * 7)
    assert np.allclose(out_moi, np.array([1.0, 2.0, 3.0]))


def test_get_bonded_axes_case2_one_heavy_no_h_calls_get_custom_axes_and_custom_moi(
    monkeypatch,
):
    ax = AxesCalculator()
    system = MagicMock()
    atom = _atom(index=1, mass=12.0, pos=(0, 0, 0))

    heavy = _FakeAtomGroup(
        [_atom(index=3, mass=12.0, pos=(1, 0, 0))], positions=np.array([[1, 0, 0]])
    )
    hyd = _FakeAtomGroup([])

    monkeypatch.setattr(ax, "find_bonded_atoms", lambda _idx, _sys: (heavy, hyd))
    monkeypatch.setattr(ax, "get_custom_axes", lambda **kwargs: np.eye(3))
    monkeypatch.setattr(
        ax, "get_custom_moment_of_inertia", lambda **kwargs: np.array([9.0, 8.0, 7.0])
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes)

    out_axes, out_moi = ax.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

    assert out_axes.shape == (3, 3)
    assert np.allclose(out_moi, np.array([9.0, 8.0, 7.0]))


def test_get_bonded_axes_case3_one_heavy_with_h_calls_get_custom_axes(monkeypatch):
    ax = AxesCalculator()
    system = MagicMock()
    atom = _atom(index=1, mass=12.0, pos=(0, 0, 0))

    heavy = _FakeAtomGroup(
        [_atom(index=3, mass=12.0, pos=(1, 0, 0))], positions=np.array([[1, 0, 0]])
    )
    hyd = _FakeAtomGroup([_atom(index=4, mass=1.0, pos=(0, 1, 0))])

    monkeypatch.setattr(ax, "find_bonded_atoms", lambda _idx, _sys: (heavy, hyd))
    called = {"n": 0}

    def _custom_axes(**kwargs):
        called["n"] += 1
        return np.eye(3) * 2

    monkeypatch.setattr(ax, "get_custom_axes", _custom_axes)
    monkeypatch.setattr(
        ax, "get_custom_moment_of_inertia", lambda **kwargs: np.array([1.0, 1.0, 1.0])
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes)

    out_axes, out_moi = ax.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

    assert called["n"] == 1
    assert np.allclose(out_axes, np.eye(3) * 2)
    assert np.allclose(out_moi, np.array([1.0, 1.0, 1.0]))


def test_get_bonded_axes_case5_two_heavy_calls_get_custom_axes(monkeypatch):
    ax = AxesCalculator()
    system = MagicMock()
    atom = _atom(index=1, mass=12.0, pos=(0, 0, 0))

    heavy_atoms = [
        _atom(index=3, mass=12.0, pos=(1, 0, 0)),
        _atom(index=5, mass=12.0, pos=(0, 1, 0)),
    ]
    heavy = _FakeAtomGroup(heavy_atoms, positions=np.array([[1, 0, 0], [0, 1, 0]]))
    heavy.positions = np.array([[1, 0, 0], [0, 1, 0]])
    hyd = _FakeAtomGroup([])

    monkeypatch.setattr(ax, "find_bonded_atoms", lambda _idx, _sys: (heavy, hyd))
    monkeypatch.setattr(ax, "get_custom_axes", lambda **kwargs: np.eye(3) * 3)
    monkeypatch.setattr(
        ax, "get_custom_moment_of_inertia", lambda **kwargs: np.array([2.0, 2.0, 2.0])
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes)

    out_axes, out_moi = ax.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

    assert np.allclose(out_axes, np.eye(3) * 3)
    assert np.allclose(out_moi, np.array([2.0, 2.0, 2.0]))


def test_get_residue_axes_no_bonds_custom_path(monkeypatch):
    ax = AxesCalculator()

    residue = MagicMock()
    residue.__len__.return_value = 1
    residue.atoms.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
    residue.select_atoms.return_value = MagicMock(positions=np.zeros((2, 3)))
    residue.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])

    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])

    def _select_atoms(q):
        if q.startswith("(resindex"):
            return []  # no bonds
        if q.startswith("resindex "):
            return residue
        return []

    u.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(ax, "get_UA_masses", lambda mol: [10.0, 12.0])
    monkeypatch.setattr(ax, "get_moment_of_inertia_tensor", lambda **kwargs: np.eye(3))
    monkeypatch.setattr(
        ax,
        "get_custom_principal_axes",
        lambda moi: (np.eye(3), np.array([3.0, 2.0, 1.0])),
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=7)

    assert trans.shape == (3, 3)
    assert rot.shape == (3, 3)
    assert np.allclose(center, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(moi, np.array([3.0, 2.0, 1.0]))


def test_get_residue_axes_with_bonds_vanilla_path(monkeypatch):
    ax = AxesCalculator()

    residue = MagicMock()
    residue.__len__.return_value = 1
    residue.atoms.principal_axes.return_value = np.eye(3) * 2
    residue.atoms.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])
    residue.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])

    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    u.atoms.principal_axes.return_value = np.eye(3) * 2

    def _select_atoms(q):
        if q.startswith("(resindex"):
            return [1]
        if q.startswith("resindex "):
            return residue
        return []

    u.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    monkeypatch.setattr(
        ax, "get_vanilla_axes", lambda mol: (np.eye(3) * 2, np.array([9.0, 8.0, 7.0]))
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=10)

    assert np.allclose(trans, np.eye(3) * 2)
    assert np.allclose(rot, np.eye(3) * 2)
    assert np.allclose(center, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(moi, np.array([9.0, 8.0, 7.0]))


def test_get_vector_wraps_periodic_boundaries():
    ac = AxesCalculator()
    dims = np.array([10.0, 10.0, 10.0])

    a = np.array([9.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    out = ac.get_vector(a, b, dims)
    np.testing.assert_allclose(out, np.array([2.0, 0.0, 0.0]))


def test_get_custom_axes_raises_when_axis1_degenerate():
    ac = AxesCalculator()
    a = np.zeros(3)
    b_list = [np.zeros(3), np.zeros(3)]
    c = np.ones(3)
    dims = np.array([10.0, 10.0, 10.0])
    with pytest.raises(ValueError):
        ac.get_custom_axes(a=a, b_list=b_list, c=c, dimensions=dims)


def test_get_custom_axes_raises_when_normalization_degenerate():
    ac = AxesCalculator()
    dims = np.array([10.0, 10.0, 10.0])

    a = np.zeros(3)
    b_list = [np.array([1.0, 0.0, 0.0])]
    c = np.array([2.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        ac.get_custom_axes(a=a, b_list=b_list, c=c, dimensions=dims)


def test_get_custom_principal_axes_flips_z_for_handedness():
    ac = AxesCalculator()

    moi = np.diag([3.0, 2.0, 1.0])
    axes, vals = ac.get_custom_principal_axes(moi)

    assert axes.shape == (3, 3)
    assert vals.shape == (3,)

    cross_xy = np.cross(axes[0], axes[1])
    assert float(np.dot(cross_xy, axes[2])) > 0.0


def test_get_moment_of_inertia_tensor_shape_and_symmetry():
    ac = AxesCalculator()
    dims = np.array([10.0, 10.0, 10.0])
    com = np.zeros(3)
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    masses = [1.0, 3.0]

    moi = ac.get_moment_of_inertia_tensor(com, positions, masses, dims)

    assert moi.shape == (3, 3)
    np.testing.assert_allclose(moi, moi.T)


def test_get_custom_moment_of_inertia_len2_zeros_smallest_component():
    ac = AxesCalculator()
    dims = np.array([10.0, 10.0, 10.0])

    UA = MagicMock()
    UA.positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    UA.masses = [12.0, 1.0]
    UA.__len__.return_value = 2

    axes = np.eye(3)
    com = np.zeros(3)

    moi = ac.get_custom_moment_of_inertia(UA, axes, com, dims)

    assert moi.shape == (3,)
    assert np.isclose(np.min(moi), 0.0)


def test_get_UA_axes_multiple_heavy_atoms_uses_custom_principal_axes(monkeypatch):
    ax = AxesCalculator()

    heavy_atoms = _FakeAtomGroup(
        [
            _FakeAtom(0, 12.0, [0, 0, 0]),
            _FakeAtom(1, 12.0, [1, 0, 0]),
        ],
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
    )

    system_atom = _FakeAtom(index=0, mass=12.0, position=[0, 0, 0])
    heavy_atom_selection = _FakeAtomGroup(
        [system_atom], positions=np.array([[0, 0, 0]], dtype=float)
    )

    class _Atoms:
        def center_of_mass(self, *args, **kwargs):
            return np.array([0.0, 0.0, 0.0], dtype=float)

        def __getitem__(self, idx):
            return system_atom

    data_container = MagicMock()
    data_container.atoms = _Atoms()
    data_container.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90], dtype=float)

    def _select_atoms(q):
        if q == "prop mass > 1.1":
            return heavy_atoms
        if q.startswith("index "):
            return heavy_atom_selection
        return _FakeAtomGroup([])

    data_container.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), np.array([1.0, 1.0, 1.0])),
    )
    monkeypatch.setattr(ax, "get_UA_masses", lambda _ag: [12.0, 12.0])

    got_tensor = MagicMock(return_value=np.eye(3))
    monkeypatch.setattr(ax, "get_moment_of_inertia_tensor", got_tensor)

    got_custom_axes = MagicMock(return_value=(np.eye(3), np.array([3.0, 2.0, 1.0])))
    monkeypatch.setattr(ax, "get_custom_principal_axes", got_custom_axes)

    trans_axes, rot_axes, center, moi = ax.get_UA_axes(data_container, index=0)

    assert trans_axes.shape == (3, 3)
    assert rot_axes.shape == (3, 3)
    assert np.allclose(center, np.array([0.0, 0.0, 0.0]))
    assert moi.shape == (3,)
    got_tensor.assert_called_once()
    got_custom_axes.assert_called_once()


def test_get_bonded_axes_returns_none_none_if_custom_axes_none(monkeypatch):
    ax = AxesCalculator()

    atom = _FakeAtom(index=7, mass=12.0, position=[0, 0, 0])
    system = MagicMock()
    dimensions = np.array([10.0, 10.0, 10.0], dtype=float)

    heavy_bonded = _FakeAtomGroup(
        [_FakeAtom(8, 12.0, [1, 0, 0])],
        positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
    )
    light_bonded = _FakeAtomGroup([], positions=np.zeros((0, 3), dtype=float))

    monkeypatch.setattr(
        ax, "find_bonded_atoms", lambda _idx, _sys: (heavy_bonded, light_bonded)
    )

    monkeypatch.setattr(ax, "get_custom_axes", lambda **kwargs: None)

    custom_axes, moi = ax.get_bonded_axes(
        system=system, atom=atom, dimensions=dimensions
    )

    assert custom_axes is None
    assert moi is None


class _FakeIndexedAtoms:
    """Container supporting ``u.atoms[index]`` and ``u.atoms[index_array]``."""

    def __init__(self, atom_map):
        self._atom_map = dict(atom_map)

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            return _FakeAtomGroup([self._atom_map[int(i)] for i in index])
        if isinstance(index, (list, tuple)):
            return _FakeAtomGroup([self._atom_map[int(i)] for i in index])
        return self._atom_map[int(index)]


class _FakeUniverse:
    """Small universe-like object with indexed atoms and dimensions."""

    def __init__(self, atom_map, dimensions=None):
        self.atoms = _FakeIndexedAtoms(atom_map)
        self.dimensions = np.asarray(
            dimensions
            if dimensions is not None
            else [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            dtype=float,
        )


def _ua_topology(
    *,
    heavy_atom_index=1,
    ua_atom_indices=(1,),
    ua_all_atom_indices=(1,),
    bonded_heavy_indices=(),
    bonded_light_indices=(),
    residue_heavy_indices=(1,),
    residue_ua_masses=(12.0,),
):
    """Build a small cached UA topology fixture."""
    return UAAxesTopology(
        heavy_atom_index=int(heavy_atom_index),
        ua_atom_indices=np.asarray(ua_atom_indices, dtype=int),
        ua_all_atom_indices=np.asarray(ua_all_atom_indices, dtype=int),
        bonded_heavy_indices=np.asarray(bonded_heavy_indices, dtype=int),
        bonded_light_indices=np.asarray(bonded_light_indices, dtype=int),
        residue_heavy_indices=np.asarray(residue_heavy_indices, dtype=int),
        residue_ua_masses=np.asarray(residue_ua_masses, dtype=float),
    )


def test_get_UA_axes_from_topology_multiple_heavy_uses_cached_indices_and_box(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [1.0, 2.0, 3.0])
    other_heavy = _FakeAtom(3, 14.0, [4.0, 5.0, 6.0])
    universe = _FakeUniverse({1: heavy_atom, 3: other_heavy})
    residue_atoms = MagicMock()
    residue_atoms.center_of_mass.return_value = np.array([9.0, 8.0, 7.0])

    topology = _ua_topology(
        heavy_atom_index=1,
        residue_heavy_indices=(1, 3),
        residue_ua_masses=(13.0, 14.0),
    )

    get_tensor = MagicMock(return_value=np.eye(3))
    get_principal = MagicMock(return_value=(np.eye(3) * 2.0, np.array([3.0, 2.0, 1.0])))
    get_bonded = MagicMock(return_value=(np.eye(3) * 4.0, np.array([1.0, 1.0, 1.0])))

    monkeypatch.setattr(ax, "get_moment_of_inertia_tensor", get_tensor)
    monkeypatch.setattr(ax, "get_custom_principal_axes", get_principal)
    monkeypatch.setattr(ax, "get_bonded_axes_from_topology", get_bonded)

    box = np.array([20.0, 30.0, 40.0])
    trans_axes, rot_axes, center, moi = ax.get_UA_axes_from_topology(
        u=universe,
        residue_atoms=residue_atoms,
        topology=topology,
        box=box,
    )

    np.testing.assert_allclose(trans_axes, np.eye(3) * 2.0)
    np.testing.assert_allclose(rot_axes, np.eye(3) * 4.0)
    np.testing.assert_allclose(center, heavy_atom.position)
    np.testing.assert_allclose(moi, np.array([1.0, 1.0, 1.0]))

    get_tensor.assert_called_once()
    tensor_kwargs = get_tensor.call_args.kwargs
    np.testing.assert_allclose(
        tensor_kwargs["center_of_mass"], np.array([9.0, 8.0, 7.0])
    )
    np.testing.assert_allclose(
        tensor_kwargs["positions"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    np.testing.assert_allclose(tensor_kwargs["masses"], np.array([13.0, 14.0]))
    np.testing.assert_allclose(tensor_kwargs["dimensions"], box)

    get_principal.assert_called_once()
    np.testing.assert_allclose(get_principal.call_args.args[0], np.eye(3))
    get_bonded.assert_called_once_with(
        u=universe,
        heavy_atom=heavy_atom,
        topology=topology,
        dimensions=box,
    )


def test_get_UA_axes_from_topology_single_heavy_uses_residue_principal_axes(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [1.0, 0.0, 0.0])
    universe = _FakeUniverse(
        {1: heavy_atom}, dimensions=[11.0, 12.0, 13.0, 90.0, 90.0, 90.0]
    )
    residue_atoms = MagicMock()
    residue_atoms.principal_axes.return_value = np.eye(3) * 5.0

    topology = _ua_topology(heavy_atom_index=1, residue_heavy_indices=(1,))

    make_whole = MagicMock()
    get_bonded = MagicMock(return_value=(np.eye(3) * 6.0, np.array([6.0, 5.0, 4.0])))

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", make_whole)
    monkeypatch.setattr(ax, "get_bonded_axes_from_topology", get_bonded)

    trans_axes, rot_axes, center, moi = ax.get_UA_axes_from_topology(
        u=universe,
        residue_atoms=residue_atoms,
        topology=topology,
        box=None,
    )

    make_whole.assert_called_once_with(residue_atoms)
    residue_atoms.principal_axes.assert_called_once()
    np.testing.assert_allclose(trans_axes, np.eye(3) * 5.0)
    np.testing.assert_allclose(rot_axes, np.eye(3) * 6.0)
    np.testing.assert_allclose(center, heavy_atom.position)
    np.testing.assert_allclose(moi, np.array([6.0, 5.0, 4.0]))

    called_kwargs = get_bonded.call_args.kwargs
    np.testing.assert_allclose(
        called_kwargs["dimensions"], np.array([11.0, 12.0, 13.0])
    )


def test_get_UA_axes_from_topology_raises_when_cached_bonded_axes_fail(monkeypatch):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [1.0, 0.0, 0.0])
    universe = _FakeUniverse({1: heavy_atom})
    residue_atoms = MagicMock()
    residue_atoms.principal_axes.return_value = np.eye(3)
    topology = _ua_topology(heavy_atom_index=1, residue_heavy_indices=(1,))

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    monkeypatch.setattr(
        ax, "get_bonded_axes_from_topology", lambda **kwargs: (None, None)
    )

    with pytest.raises(ValueError, match="cached UA bead"):
        ax.get_UA_axes_from_topology(
            u=universe,
            residue_atoms=residue_atoms,
            topology=topology,
            box=None,
        )


def test_get_bonded_axes_from_topology_non_heavy_returns_none_none():
    ax = AxesCalculator()
    light_atom = _FakeAtom(1, 1.0, [0.0, 0.0, 0.0])

    custom_axes, moi = ax.get_bonded_axes_from_topology(
        u=MagicMock(),
        heavy_atom=light_atom,
        topology=_ua_topology(heavy_atom_index=1),
        dimensions=np.array([10.0, 10.0, 10.0]),
    )

    assert custom_axes is None
    assert moi is None


def test_get_bonded_axes_from_topology_no_bonded_heavy_uses_vanilla_axes(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [0.0, 0.0, 0.0])
    hydrogen = _FakeAtom(2, 1.0, [1.0, 0.0, 0.0])
    universe = _FakeUniverse({1: heavy_atom, 2: hydrogen})
    topology = _ua_topology(
        heavy_atom_index=1,
        ua_atom_indices=(1, 2),
        ua_all_atom_indices=(1, 2),
        bonded_heavy_indices=(),
        bonded_light_indices=(2,),
    )

    get_vanilla = MagicMock(return_value=(np.eye(3) * 7.0, np.array([7.0, 8.0, 9.0])))
    get_custom_moi = MagicMock()
    get_flipped = MagicMock(return_value=np.eye(3) * -7.0)

    monkeypatch.setattr(ax, "get_vanilla_axes", get_vanilla)
    monkeypatch.setattr(ax, "get_custom_moment_of_inertia", get_custom_moi)
    monkeypatch.setattr(ax, "get_flipped_axes", get_flipped)

    custom_axes, moi = ax.get_bonded_axes_from_topology(
        u=universe,
        heavy_atom=heavy_atom,
        topology=topology,
        dimensions=np.array([10.0, 10.0, 10.0]),
    )

    np.testing.assert_allclose(custom_axes, np.eye(3) * -7.0)
    np.testing.assert_allclose(moi, np.array([7.0, 8.0, 9.0]))
    get_vanilla.assert_called_once()
    get_custom_moi.assert_not_called()
    get_flipped.assert_called_once()


def test_get_bonded_axes_from_topology_one_heavy_no_light_uses_custom_axes(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [0.0, 0.0, 0.0])
    bonded_heavy = _FakeAtom(3, 12.0, [1.0, 0.0, 0.0])
    universe = _FakeUniverse({1: heavy_atom, 3: bonded_heavy})
    topology = _ua_topology(
        heavy_atom_index=1,
        ua_atom_indices=(1,),
        ua_all_atom_indices=(1, 3),
        bonded_heavy_indices=(3,),
        bonded_light_indices=(),
    )

    get_custom_axes = MagicMock(return_value=np.eye(3) * 2.0)
    get_custom_moi = MagicMock(return_value=np.array([2.0, 3.0, 4.0]))
    get_flipped = MagicMock(return_value=np.eye(3) * 3.0)

    monkeypatch.setattr(ax, "get_custom_axes", get_custom_axes)
    monkeypatch.setattr(ax, "get_custom_moment_of_inertia", get_custom_moi)
    monkeypatch.setattr(ax, "get_flipped_axes", get_flipped)

    custom_axes, moi = ax.get_bonded_axes_from_topology(
        u=universe,
        heavy_atom=heavy_atom,
        topology=topology,
        dimensions=np.array([10.0, 10.0, 10.0]),
    )

    np.testing.assert_allclose(custom_axes, np.eye(3) * 3.0)
    np.testing.assert_allclose(moi, np.array([2.0, 3.0, 4.0]))

    kwargs = get_custom_axes.call_args.kwargs
    np.testing.assert_allclose(kwargs["a"], heavy_atom.position)
    np.testing.assert_allclose(kwargs["b_list"][0], bonded_heavy.position)
    np.testing.assert_allclose(kwargs["c"], np.zeros(3))
    get_custom_moi.assert_called_once()
    get_flipped.assert_called_once()


def test_get_bonded_axes_from_topology_one_heavy_with_light_uses_light_as_c(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [0.0, 0.0, 0.0])
    bonded_heavy = _FakeAtom(3, 12.0, [1.0, 0.0, 0.0])
    bonded_light = _FakeAtom(2, 1.0, [0.0, 1.0, 0.0])
    universe = _FakeUniverse({1: heavy_atom, 2: bonded_light, 3: bonded_heavy})
    topology = _ua_topology(
        heavy_atom_index=1,
        ua_atom_indices=(1, 2),
        ua_all_atom_indices=(1, 3, 2),
        bonded_heavy_indices=(3,),
        bonded_light_indices=(2,),
    )

    get_custom_axes = MagicMock(return_value=np.eye(3))
    monkeypatch.setattr(ax, "get_custom_axes", get_custom_axes)
    monkeypatch.setattr(
        ax,
        "get_custom_moment_of_inertia",
        lambda **kwargs: np.array([1.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes)

    custom_axes, moi = ax.get_bonded_axes_from_topology(
        u=universe,
        heavy_atom=heavy_atom,
        topology=topology,
        dimensions=np.array([10.0, 10.0, 10.0]),
    )

    np.testing.assert_allclose(custom_axes, np.eye(3))
    np.testing.assert_allclose(moi, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(
        get_custom_axes.call_args.kwargs["c"], bonded_light.position
    )


def test_get_bonded_axes_from_topology_two_heavy_uses_heavy_positions_as_b_list(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [0.0, 0.0, 0.0])
    bonded_heavy_0 = _FakeAtom(3, 12.0, [1.0, 0.0, 0.0])
    bonded_heavy_1 = _FakeAtom(4, 12.0, [0.0, 1.0, 0.0])
    universe = _FakeUniverse(
        {
            1: heavy_atom,
            3: bonded_heavy_0,
            4: bonded_heavy_1,
        }
    )
    topology = _ua_topology(
        heavy_atom_index=1,
        ua_atom_indices=(1,),
        ua_all_atom_indices=(1, 3, 4),
        bonded_heavy_indices=(3, 4),
        bonded_light_indices=(),
    )

    get_custom_axes = MagicMock(return_value=np.eye(3) * 4.0)
    monkeypatch.setattr(ax, "get_custom_axes", get_custom_axes)
    monkeypatch.setattr(
        ax,
        "get_custom_moment_of_inertia",
        lambda **kwargs: np.array([4.0, 5.0, 6.0]),
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes)

    custom_axes, moi = ax.get_bonded_axes_from_topology(
        u=universe,
        heavy_atom=heavy_atom,
        topology=topology,
        dimensions=np.array([10.0, 10.0, 10.0]),
    )

    np.testing.assert_allclose(custom_axes, np.eye(3) * 4.0)
    np.testing.assert_allclose(moi, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(
        get_custom_axes.call_args.kwargs["b_list"],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    np.testing.assert_allclose(
        get_custom_axes.call_args.kwargs["c"],
        bonded_heavy_1.position,
    )


def test_get_bonded_axes_from_topology_returns_none_when_custom_axes_none(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [0.0, 0.0, 0.0])
    bonded_heavy = _FakeAtom(3, 12.0, [1.0, 0.0, 0.0])
    universe = _FakeUniverse({1: heavy_atom, 3: bonded_heavy})
    topology = _ua_topology(
        heavy_atom_index=1,
        ua_atom_indices=(1,),
        ua_all_atom_indices=(1, 3),
        bonded_heavy_indices=(3,),
        bonded_light_indices=(),
    )

    get_custom_moi = MagicMock()
    get_flipped = MagicMock()

    monkeypatch.setattr(ax, "get_custom_axes", lambda **kwargs: None)
    monkeypatch.setattr(ax, "get_custom_moment_of_inertia", get_custom_moi)
    monkeypatch.setattr(ax, "get_flipped_axes", get_flipped)

    custom_axes, moi = ax.get_bonded_axes_from_topology(
        u=universe,
        heavy_atom=heavy_atom,
        topology=topology,
        dimensions=np.array([10.0, 10.0, 10.0]),
    )

    assert custom_axes is None
    assert moi is None
    get_custom_moi.assert_not_called()
    get_flipped.assert_not_called()
