from unittest.mock import MagicMock

import numpy as np
import pytest

from CodeEntropy.levels.axes import AxesCalculator


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

    trans, rot, center, moi = ax.get_UA_axes(u, index=0, res_position=None)

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
        ax.get_UA_axes(u, index=0, res_position=None)


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

    trans, rot, center, moi = ax.get_residue_axes(u, index=10, residue=residue)

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

    trans_axes, rot_axes, center, moi = ax.get_UA_axes(
        data_container, index=0, res_position=None
    )

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
