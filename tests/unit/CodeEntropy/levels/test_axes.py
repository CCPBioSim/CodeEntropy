from unittest.mock import MagicMock

import MDAnalysis as mda
import numpy as np
import pytest

from CodeEntropy.levels.axes import AxesCalculator
from CodeEntropy.levels.nodes.axes_topology import (
    BuildAxesTopologyNode,
    ResidueAxesTopology,
    UAAxesTopology,
)


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
        ax.get_residue_axes(u, index=5, relative_index=0)


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

    monkeypatch.setattr(
        ax, "get_molecule_axes", lambda molecule: (np.eye(3), np.array([3.0, 2.0, 1.0]))
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=7, relative_index=0)

    assert np.allclose(trans, np.eye(3))
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(center, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(moi, np.array([3.0, 2.0, 1.0]))


def test_get_residue_axes_one_residue_uses_principal_axes(monkeypatch):
    ax = AxesCalculator()

    residue = MagicMock()
    residue.__len__.return_value = 1
    residue.atoms.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])
    residue.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])

    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])

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
        ax, "get_molecule_axes", lambda molecule: (np.eye(3), np.array([9.0, 8.0, 7.0]))
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=10, relative_index=0)

    assert np.allclose(trans, np.eye(3))
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(moi, np.array([9.0, 8.0, 7.0]))


def test_get_UA_axes_uses_principal_axes_when_single_heavy(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    u.center_of_mass.return_value = np.array([[4.0, 0.0, 0.0]])

    # heavy_atoms length <= 1 => principal_axes path
    heavy_atom = MagicMock(index=5, position=np.array([4.0, 0.0, 0.0]))
    heavy_atoms = [heavy_atom]

    def _sel(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
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
        if q == "mass 2 to 999":
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
        ax.get_UA_axes(u, index=5, res_position=None)


def test_get_bonded_vector_axes_degenerate_axis1_raises():
    ax = AxesCalculator()
    a = np.zeros(3)
    b_list = [np.zeros(3)]
    with pytest.raises(ValueError):
        ax.get_bonded_vector_axes(
            a=a, b_list=b_list, c=np.zeros(3), dimensions=np.array([10.0, 10.0, 10.0])
        )


def test_get_bonded_vector_axes_normalizes_and_uses_bc_when_multiple_b(monkeypatch):
    ax = AxesCalculator()
    a = np.array([0.0, 0.0, 0.0])
    b_list = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    c = np.array([0.0, 1.0, 0.0])

    axes = ax.get_bonded_vector_axes(
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


def test_get_principal_axes_from_tensor_returns_axes_and_moments():
    ax = AxesCalculator()
    moi = np.eye(3)
    axes, vals = ax.get_principal_axes_from_tensor(moi)
    assert axes.shape == (3, 3)
    assert vals.shape == (3,)


def test_get_molecule_axes_sorts_eigenvalues_desc_by_abs(monkeypatch):
    ax = AxesCalculator()
    mol = MagicMock()
    moi_tensor = np.diag([1.0, -10.0, 3.0])
    mol.moment_of_inertia.return_value = moi_tensor
    mol.atoms = MagicMock()

    # avoid real MDAnalysis unwrap
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)

    axes, moments = ax.get_molecule_axes(mol)

    assert axes.shape == (3, 3)
    # sorted by abs descending => -10, 3, 1
    assert np.allclose(moments, np.array([-10.0, 3.0, 1.0]))


def test_get_principal_axes_sorts_by_descending_eigenvalue():
    ax = AxesCalculator()
    group = MagicMock()
    group.atoms.moment_of_inertia.return_value = np.diag([1.0, 3.0, 2.0])

    axes = ax.get_principal_axes_from_group(group)

    # eigenvalues 3, 2, 1 => rows along y, z, x (sign may differ)
    expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    assert np.allclose(np.abs(axes), expected)


def test_get_principal_axes_orthonormal_and_right_handed_for_near_degenerate():
    ax = AxesCalculator()
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    tensor = q @ np.diag([1.0, 1.0 + 1e-9, 5.0]) @ q.T
    group = MagicMock()
    group.atoms.moment_of_inertia.return_value = tensor

    axes = ax.get_principal_axes_from_group(group)

    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(axes), 1.0)
    assert np.isclose(abs(np.dot(axes[0], q[:, 2])), 1.0)


def _scrambled_eigh(seed):
    """Return an ``eigh`` that gives valid but arbitrary signs and degenerate bases."""
    rng = np.random.default_rng(seed)
    real_eigh = np.linalg.eigh

    def scrambled(matrix):
        values, vectors = real_eigh(matrix)
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                if np.isclose(values[i], values[j], rtol=1e-5):
                    angle = rng.uniform(0, 2 * np.pi)
                    c, s = np.cos(angle), np.sin(angle)
                    vi, vj = vectors[:, i].copy(), vectors[:, j].copy()
                    vectors[:, i] = c * vi + s * vj
                    vectors[:, j] = -s * vi + c * vj
        return values, vectors * rng.choice([-1.0, 1.0], size=n)

    return scrambled


def _example_tensors():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return {
        "asymmetric_top": q @ np.diag([5.0, 3.0, 1.0]) @ q.T,
        "linear": q @ np.diag([4.0, 4.0, 0.0]) @ q.T,
        "symmetric_top": q @ np.diag([5.0, 2.0, 2.0]) @ q.T,
        "spherical_top": 3.0 * np.eye(3),
    }


@pytest.mark.parametrize("name", list(_example_tensors()))
def test_get_principal_axes_from_tensor_orthonormal_right_handed_and_diagonalising(
    name,
):
    ax = AxesCalculator()
    tensor = _example_tensors()[name]

    axes, moments = ax.get_principal_axes_from_tensor(tensor)

    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(axes), 1.0)
    assert np.allclose(axes @ tensor @ axes.T, np.diag(moments), atol=1e-12)
    assert np.all(np.diff(np.abs(moments)) <= 1e-12)


@pytest.mark.parametrize("name", list(_example_tensors()))
def test_get_principal_axes_from_tensor_independent_of_eigensolver_choices(
    monkeypatch, name
):
    ax = AxesCalculator()
    tensor = _example_tensors()[name]
    expected_axes, expected_moments = ax.get_principal_axes_from_tensor(tensor)

    for seed in range(20):
        monkeypatch.setattr(np.linalg, "eigh", _scrambled_eigh(seed))
        axes, moments = ax.get_principal_axes_from_tensor(tensor)
        assert np.allclose(axes, expected_axes, atol=1e-9)
        assert np.allclose(moments, expected_moments)


@pytest.mark.parametrize("name", list(_example_tensors()))
def test_get_principal_axes_from_tensor_stable_under_rounding_level_noise(name):
    ax = AxesCalculator()
    tensor = _example_tensors()[name]
    expected_axes, _ = ax.get_principal_axes_from_tensor(tensor)
    rng = np.random.default_rng(2)

    for _ in range(50):
        noise = rng.normal(scale=1e-7, size=(3, 3))
        axes, _ = ax.get_principal_axes_from_tensor(tensor + (noise + noise.T) / 2)
        assert np.allclose(axes, expected_axes, atol=1e-4)


def test_get_principal_axes_from_tensor_keeps_unique_axis_of_linear_tensor_exact():
    ax = AxesCalculator()
    direction = np.array([1.0, -2.0, 0.5])
    direction /= np.linalg.norm(direction)
    tensor = 4.0 * (np.eye(3) - np.outer(direction, direction))

    axes, moments = ax.get_principal_axes_from_tensor(tensor)

    assert np.isclose(moments[2], 0.0, atol=1e-12)
    assert np.isclose(abs(axes[2] @ direction), 1.0, atol=1e-12)


def test_get_principal_axes_from_tensor_signs_first_two_axes_along_reference():
    ax = AxesCalculator()
    tensor = np.diag([5.0, 3.0, 1.0])

    axes, _ = ax.get_principal_axes_from_tensor(tensor)

    assert axes[0] @ np.array([1.0, 2.0, 3.0]) > 0
    assert axes[1] @ np.array([1.0, 2.0, 3.0]) > 0


def test_get_molecule_axes_moments_ignore_position_relative_to_box():
    """Shifting a molecule by a box vector must not change its moments."""
    universe = mda.Universe.empty(
        n_atoms=3, n_residues=1, atom_resindex=[0, 0, 0], trajectory=True
    )
    universe.add_TopologyAttr("masses", [16.0, 1.0, 1.0])
    universe.add_TopologyAttr("bonds", [(0, 1), (0, 2)])
    universe.dimensions = [50.0, 50.0, 50.0, 90.0, 90.0, 90.0]
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    ax = AxesCalculator()

    universe.atoms.positions = positions + 25.0
    _, moments = ax.get_molecule_axes(universe.atoms)
    universe.atoms.positions = positions + 25.0 + np.array([50.0, 0.0, 0.0])
    _, shifted_moments = ax.get_molecule_axes(universe.atoms)

    assert np.allclose(shifted_moments, moments, rtol=1e-3)
    assert moments.max() < 10.0


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


def test_get_bonded_axes_case1_uses_molecule_axes_and_returns_flipped(monkeypatch):
    ax = AxesCalculator()
    system = MagicMock()
    atom = _atom(index=1, mass=12.0, pos=(1, 0, 0))

    heavy = _FakeAtomGroup([])  # len == 0 -> case1
    hyd = _FakeAtomGroup([_atom(index=2, mass=1.0)])
    monkeypatch.setattr(ax, "find_bonded_atoms", lambda _idx, _sys: (heavy, hyd))

    monkeypatch.setattr(
        ax, "get_molecule_axes", lambda _ag: (np.eye(3) * 7, np.array([1.0, 2.0, 3.0]))
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes * -1)

    out_axes, out_moi = ax.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

    assert np.allclose(out_axes, -np.eye(3) * 7)
    assert np.allclose(out_moi, np.array([1.0, 2.0, 3.0]))


def test_get_bonded_axes_case2_one_heavy_no_h_calls_bonded_vector_axes_and_custom_moi(
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
    monkeypatch.setattr(ax, "get_bonded_vector_axes", lambda **kwargs: np.eye(3))
    monkeypatch.setattr(
        ax, "get_custom_moment_of_inertia", lambda **kwargs: np.array([9.0, 8.0, 7.0])
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes)

    out_axes, out_moi = ax.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

    assert out_axes.shape == (3, 3)
    assert np.allclose(out_moi, np.array([9.0, 8.0, 7.0]))


def test_get_bonded_axes_case3_one_heavy_with_two_h_calls_get_bonded_vector_axes(
    monkeypatch,
):
    ax = AxesCalculator()
    system = MagicMock()
    atom = _atom(index=1, mass=12.0, pos=(0, 0, 0))

    heavy = _FakeAtomGroup(
        [_atom(index=3, mass=12.0, pos=(1, 0, 0))], positions=np.array([[1, 0, 0]])
    )
    hyd = _FakeAtomGroup(
        [
            _atom(index=4, mass=1.0, pos=(0, 1, 0)),
            _atom(index=5, mass=1.0, pos=(0, 0, 1)),
        ]
    )

    monkeypatch.setattr(ax, "find_bonded_atoms", lambda _idx, _sys: (heavy, hyd))
    called = {"n": 0}

    def _custom_axes(**kwargs):
        called["n"] += 1
        return np.eye(3) * 2

    monkeypatch.setattr(ax, "get_bonded_vector_axes", _custom_axes)
    monkeypatch.setattr(
        ax, "get_custom_moment_of_inertia", lambda **kwargs: np.array([1.0, 1.0, 1.0])
    )
    monkeypatch.setattr(ax, "get_flipped_axes", lambda ua, axes, com, dims: axes)

    out_axes, out_moi = ax.get_bonded_axes(system, atom, np.array([10.0, 10.0, 10.0]))

    assert called["n"] == 1
    assert np.allclose(out_axes, np.eye(3) * 2)
    assert np.allclose(out_moi, np.array([1.0, 1.0, 1.0]))


def test_get_bonded_axes_case5_two_heavy_calls_get_bonded_vector_axes(monkeypatch):
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
    monkeypatch.setattr(ax, "get_bonded_vector_axes", lambda **kwargs: np.eye(3) * 3)
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

    monkeypatch.setattr(
        ax,
        "get_molecule_axes",
        lambda molecule: (2 * np.eye(3), np.array([3.0, 2.0, 1.0])),
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=7, relative_index=0)

    assert trans.shape == (3, 3)
    assert rot.shape == (3, 3)
    assert np.allclose(center, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(moi, np.array([3.0, 2.0, 1.0]))


def test_get_residue_axes_one_residue_principal_axes_path(monkeypatch):
    ax = AxesCalculator()

    residue = MagicMock()
    residue.__len__.return_value = 1
    residue.atoms.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])
    residue.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])

    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])

    def _select_atoms(q):
        if q.startswith("(resindex"):
            return [1]
        if q.startswith("resindex "):
            return residue
        return []

    u.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)

    monkeypatch.setattr(
        ax,
        "get_molecule_axes",
        lambda molecule: (2 * np.eye(3), np.array([9.0, 8.0, 7.0])),
    )

    trans, rot, center, moi = ax.get_residue_axes(u, index=10, relative_index=0)

    assert np.allclose(trans, 2 * np.eye(3))
    assert np.allclose(rot, 2 * np.eye(3))
    assert np.allclose(center, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(moi, np.array([9.0, 8.0, 7.0]))


def test_get_vector_wraps_periodic_boundaries():
    ac = AxesCalculator()
    dims = np.array([10.0, 10.0, 10.0])

    a = np.array([9.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    out = ac.get_vector(a, b, dims)
    np.testing.assert_allclose(out, np.array([2.0, 0.0, 0.0]))


def test_get_bonded_vector_axes_raises_when_axis1_degenerate():
    ac = AxesCalculator()
    a = np.zeros(3)
    b_list = [np.zeros(3), np.zeros(3)]
    c = np.ones(3)
    dims = np.array([10.0, 10.0, 10.0])
    with pytest.raises(ValueError):
        ac.get_bonded_vector_axes(a=a, b_list=b_list, c=c, dimensions=dims)


def test_get_bonded_vector_axes_raises_when_normalization_degenerate():
    ac = AxesCalculator()
    dims = np.array([10.0, 10.0, 10.0])

    a = np.zeros(3)
    b_list = [np.array([1.0, 0.0, 0.0])]
    c = np.array([2.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        ac.get_bonded_vector_axes(a=a, b_list=b_list, c=c, dimensions=dims)


def test_get_principal_axes_from_tensor_is_right_handed():
    ac = AxesCalculator()

    moi = np.diag([3.0, 2.0, 1.0])
    axes, vals = ac.get_principal_axes_from_tensor(moi)

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
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )

    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=[0, 0, 0]),
            _atom(index=1, mass=12.0, pos=[1, 0, 0]),
        ],
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
    )

    system_atom = _atom(index=0, mass=12.0, pos=[0, 0, 0])
    heavy_atom_selection = _FakeAtomGroup(
        [system_atom], positions=np.array([[0, 0, 0]], dtype=float)
    )

    class _Atoms:
        def center_of_mass(self, *args, **kwargs):
            return np.array([0.0, 0.0, 0.0], dtype=float)

        def __getitem__(self, idx):
            return system_atom

        def select_atoms(self, q):
            if q == "mass 2 to 999":
                return heavy_atoms
            if q.startswith("index "):
                return [heavy_atom_selection[0]]

    data_container = MagicMock()
    data_container.atoms = _Atoms()
    data_container.residues.__len__.return_value = 1
    data_container.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90], dtype=float)
    _FakeAtomGroup.atoms = heavy_atom_selection

    def _select_atoms(q):
        if q == "mass 2 to 999":
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

    trans_axes, rot_axes, center, moi = ax.get_UA_axes(
        data_container, index=0, res_position=None
    )

    assert trans_axes.shape == (3, 3)
    assert rot_axes.shape == (3, 3)
    assert np.allclose(center, np.array([0.0, 0.0, 0.0]))
    assert moi.shape == (3,)


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

    monkeypatch.setattr(ax, "get_bonded_vector_axes", lambda **kwargs: None)

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


def _residue_topology(
    *,
    residue_heavy_indices=(1,),
    residue_ua_masses=(12.0,),
    has_neighbor_bonds=False,
):
    """Build a small cached residue topology fixture."""
    return ResidueAxesTopology(
        residue_heavy_indices=np.asarray(residue_heavy_indices, dtype=int),
        residue_ua_masses=np.asarray(residue_ua_masses, dtype=float),
        has_neighbor_bonds=bool(has_neighbor_bonds),
    )


def test_get_residue_axes_from_topology_no_neighbor_bonds_uses_cached_indices(
    monkeypatch,
):
    ax = AxesCalculator()

    heavy_atom = _FakeAtom(1, 12.0, [1.0, 2.0, 3.0])
    other_heavy = _FakeAtom(3, 14.0, [4.0, 5.0, 6.0])
    universe = _FakeUniverse({1: heavy_atom, 3: other_heavy})
    mol = MagicMock()
    residue_atoms = MagicMock()
    residue_atoms.center_of_mass.return_value = np.array([9.0, 8.0, 7.0])
    topology = _residue_topology(
        residue_heavy_indices=(1, 3),
        residue_ua_masses=(13.0, 14.0),
        has_neighbor_bonds=False,
    )

    get_principal = MagicMock(return_value=(np.eye(3) * 2.0, np.array([3.0, 2.0, 1.0])))

    monkeypatch.setattr(ax, "get_molecule_axes", get_principal)

    trans_axes, rot_axes, center, moi = ax.get_residue_axes_from_topology(
        u=universe,
        mol=mol,
        residue_atoms=residue_atoms,
        topology=topology,
    )

    np.testing.assert_allclose(trans_axes, np.eye(3) * 2.0)
    np.testing.assert_allclose(rot_axes, np.eye(3) * 2.0)
    np.testing.assert_allclose(center, np.array([9.0, 8.0, 7.0]))
    np.testing.assert_allclose(moi, np.array([3.0, 2.0, 1.0]))

    get_principal.assert_called_once()


def test_get_residue_axes_from_topology_neighbor_bonds_uses_molecule_axes(
    monkeypatch,
):
    ax = AxesCalculator()

    universe = _FakeUniverse(
        {},
        dimensions=[11.0, 12.0, 13.0, 90.0, 90.0, 90.0],
    )
    mol = MagicMock()
    get_principal = MagicMock(return_value=np.eye(3) * 5.0)
    monkeypatch.setattr(ax, "get_principal_axes_from_group", get_principal)
    residue_atoms = MagicMock()
    residue_atoms.center_of_mass.return_value = np.array([1.0, 2.0, 3.0])
    topology = _residue_topology(has_neighbor_bonds=True)

    make_whole = MagicMock()
    get_molecule = MagicMock(return_value=(np.eye(3) * 6.0, np.array([6.0, 5.0, 4.0])))

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", make_whole)
    monkeypatch.setattr(ax, "get_molecule_axes", get_molecule)

    trans_axes, rot_axes, center, moi = ax.get_residue_axes_from_topology(
        u=universe,
        mol=mol,
        residue_atoms=residue_atoms,
        topology=topology,
    )

    make_whole.assert_called_once_with(mol.atoms)
    get_principal.assert_called_once()
    get_molecule.assert_called_once_with(residue_atoms)
    np.testing.assert_allclose(trans_axes, np.eye(3) * 5.0)
    np.testing.assert_allclose(rot_axes, np.eye(3) * 6.0)
    np.testing.assert_allclose(center, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(moi, np.array([6.0, 5.0, 4.0]))


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

    get_principal = MagicMock(return_value=(np.eye(3) * 2.0))
    get_bonded = MagicMock(return_value=(np.eye(3) * 4.0, np.array([1.0, 1.0, 1.0])))

    monkeypatch.setattr(ax, "get_principal_axes_from_group", get_principal)
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

    get_principal.assert_called_once()
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
    get_principal = MagicMock(return_value=np.eye(3) * 5.0)
    monkeypatch.setattr(ax, "get_principal_axes_from_group", get_principal)

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
    get_principal.assert_called_once()
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
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )
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


def test_get_bonded_axes_from_topology_no_bonded_heavy_uses_molecule_axes(
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

    get_molecule = MagicMock(return_value=(np.eye(3) * 7.0, np.array([7.0, 8.0, 9.0])))
    get_custom_moi = MagicMock()
    get_flipped = MagicMock(return_value=np.eye(3) * -7.0)

    monkeypatch.setattr(ax, "get_molecule_axes", get_molecule)
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
    get_molecule.assert_called_once()
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

    get_bonded_vector_axes = MagicMock(return_value=np.eye(3) * 2.0)
    get_custom_moi = MagicMock(return_value=np.array([2.0, 3.0, 4.0]))
    get_flipped = MagicMock(return_value=np.eye(3) * 3.0)

    monkeypatch.setattr(ax, "get_bonded_vector_axes", get_bonded_vector_axes)
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

    kwargs = get_bonded_vector_axes.call_args.kwargs
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
    second_light = _FakeAtom(4, 1.0, [0.0, 0.0, 1.0])
    universe = _FakeUniverse(
        {1: heavy_atom, 2: bonded_light, 3: bonded_heavy, 4: second_light}
    )
    topology = _ua_topology(
        heavy_atom_index=1,
        ua_atom_indices=(1, 2, 4),
        ua_all_atom_indices=(1, 3, 2, 4),
        bonded_heavy_indices=(3,),
        bonded_light_indices=(2, 4),
    )

    get_bonded_vector_axes = MagicMock(return_value=np.eye(3))
    monkeypatch.setattr(ax, "get_bonded_vector_axes", get_bonded_vector_axes)
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
        get_bonded_vector_axes.call_args.kwargs["c"], bonded_light.position
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

    get_bonded_vector_axes = MagicMock(return_value=np.eye(3) * 4.0)
    monkeypatch.setattr(ax, "get_bonded_vector_axes", get_bonded_vector_axes)
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
        get_bonded_vector_axes.call_args.kwargs["b_list"],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    np.testing.assert_allclose(
        get_bonded_vector_axes.call_args.kwargs["c"],
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

    monkeypatch.setattr(ax, "get_bonded_vector_axes", lambda **kwargs: None)
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


def test_get_residue_custom_axes(monkeypatch):
    ax = AxesCalculator()

    edge_atoms = _FakeAtomGroup(
        [_FakeAtom(8, 12.0, [1, 0, 0]), _FakeAtom(10, 14.0, [0, 0, 0])],
        positions=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
    )

    backbone_center = np.array([0.0, 1.0, 0.0])
    rot_center, rot_axes = ax.get_residue_custom_axes(
        [edge_atoms[0].position, edge_atoms[1].position], backbone_center
    )

    assert rot_center.shape == (3,)
    assert rot_axes.shape == (3, 3)


def test_get_residue_custom_axes_raises(monkeypatch):
    ax = AxesCalculator()
    edge_atoms = _FakeAtomGroup(
        [_FakeAtom(8, 12.0, [1, 0, 0]), _FakeAtom(10, 14.0, [0, 0, 0])],
        positions=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
    )
    backbone_center = np.array([2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        ax.get_residue_custom_axes(
            [edge_atoms[0].position, edge_atoms[1].position], backbone_center
        )


def test_get_custom_residue_moment_of_inertia(monkeypatch):
    ax = AxesCalculator()
    heavy_atoms = _FakeAtomGroup(
        [_FakeAtom(0, 12.0, [1, 2, 1]), _FakeAtom(1, 12.0, [2, 1, 1])],
        positions=np.array([[1, 2, 1], [2, 1, 1]], dtype=float),
    )
    dimensions = np.array([10.0, 10.0, 10.0], dtype=float)

    moi = ax.get_custom_residue_moment_of_inertia(
        center_of_mass=np.array([1, 1, 1]),
        positions=heavy_atoms.positions,
        masses=heavy_atoms.masses,
        custom_rot_axes=np.eye(3),
        dimensions=dimensions,
    )

    assert moi.shape == (3,)


def test_get_residue_bonded_axes_1_heavy_atom_backbone_2neighbours(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    residue = u.select_atoms("resindex 1")
    residue.__len__.return_value = 1

    edge_atom_set = _FakeAtomGroup(
        [
            _atom(index=8, mass=12.0, pos=[1, 0, 0]),
            _atom(index=10, mass=14.0, pos=[0, 0, 0]),
        ],
        positions=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
    )
    backbone_atom = _atom(index=9, mass=12.0, pos=[0, 1, 0])

    def _select_atoms(q):
        if q.endswith("(bonded resindex 0 or resindex 2)"):
            return edge_atom_set

    u.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )
    monkeypatch.setattr(ax, "get_chain", lambda residue, first, last: [backbone_atom])
    monkeypatch.setattr(
        ax,
        "get_custom_residue_moment_of_inertia",
        lambda center_of_mass, positions, masses, custom_rot_axes, dimensions: np.array(
            [1, 1, 1]
        ),
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_residue_axes(
        u, index=1, relative_index=0
    )

    assert len(edge_atom_set) == 2
    assert np.allclose(trans_axes, np.eye(3))
    assert rot_axes.shape == (3, 3)
    assert rot_center.shape == (3,)
    assert np.allclose(moi, np.array([1, 1, 1]))


def test_get_residue_bonded_axes_multiple_heavy_atoms_backbone_2neighbours(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    residue = u.select_atoms("resindex 1")
    residue.__len__.return_value = 1

    edge_atom_set = _FakeAtomGroup(
        [
            _atom(index=8, mass=12.0, pos=[1, 0, 0]),
            _atom(index=12, mass=14.0, pos=[0, 0, 0]),
        ],
        positions=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
    )
    backbone_atoms = [
        _atom(index=9, mass=12.0, pos=[0, 1, 0]),
        _atom(index=10, mass=14.0, pos=[0, 2, 0]),
    ]

    def _select_atoms(q):
        if q.endswith("(bonded resindex 0 or resindex 2)"):
            return edge_atom_set

    u.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )
    monkeypatch.setattr(
        ax, "get_chain", lambda residue, edge_atom_1, edge_atom_2: backbone_atoms
    )
    monkeypatch.setattr(
        ax,
        "get_custom_residue_moment_of_inertia",
        lambda center_of_mass, positions, masses, custom_rot_axes, dimensions: np.array(
            [1, 1, 1]
        ),
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_residue_axes(
        u, index=1, relative_index=0
    )

    #

    assert len(edge_atom_set) == 2
    assert np.allclose(trans_axes, np.eye(3))
    assert rot_axes.shape == (3, 3)
    assert rot_center.shape == (3,)
    assert np.allclose(moi, np.array([1, 1, 1]))


def test_get_residue_bonded_axes_terminal_resid(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    residue = u.select_atoms("resindex 0")
    residue.__len__.return_value = 3
    uas = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=[1, -1, 0]),
            _atom(index=1, mass=12.0, pos=[0, 1, 0]),
            _atom(index=2, mass=12.0, pos=[0, 0, 0]),
        ]
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            return uas
        if q.startswith("(mass 2 to 999) and bonded"):
            return [uas[1]]
        if q.startswith("resindex 0 and (bonded resindex"):
            return [uas[2]]

    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )
    u.atoms.select_atoms.side_effect = _select_atoms
    residue.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(
        ax,
        "get_custom_residue_moment_of_inertia",
        lambda center_of_mass, positions, masses, custom_rot_axes, dimensions: np.array(
            [1, 1, 1]
        ),
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_residue_axes(
        u, index=0, relative_index=0
    )

    assert np.allclose(trans_axes, np.eye(3))
    assert rot_axes.shape == (3, 3)
    assert rot_center.shape == (3,)
    assert np.allclose(moi, np.array([1, 1, 1]))


def test_get_ua_axes_bonded_axes_2neighbours_1_heavy_atom_backbone(monkeypatch):
    ax = AxesCalculator()
    residue_group = MagicMock()
    residue_group.__len__ = 3
    residue = residue_group.residues[1]

    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=3, mass=12.0, pos=(1, 0, 0)),
            _atom(index=5, mass=12.0, pos=(0, 1, 0)),
            _atom(index=7, mass=12.0, pos=(0, 0, 1)),
        ],
        positions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    )

    edge_atoms = _FakeAtomGroup(
        [
            _atom(index=3, mass=12.0, pos=(1, 0, 0)),
            _atom(index=7, mass=12.0, pos=(0, 0, 1)),
        ],
        positions=np.array([[1, 0, 0], [0, 0, 1]], dtype=float),
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return heavy_atoms
        if q.startswith("resindex "):
            return edge_atoms
        if q.startswith("index "):
            return [heavy_atoms[1]]

    residue_group.select_atoms.side_effect = _select_atoms
    residue.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(ax, "get_chain", lambda residue, first, last: [heavy_atoms[1]])
    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), np.array([1.0, 2.0, 3.0])),
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_UA_axes(
        data_container=residue_group, index=1, res_position=0
    )

    assert trans_axes.shape == (3, 3)
    assert rot_axes.shape == (3, 3)
    assert moi.shape == (3,)
    assert rot_center.shape == (3,)


def test_get_ua_axes_bonded_axes_2neighbours_multiple_heavy_atoms_backbone(monkeypatch):
    ax = AxesCalculator()
    residue_group = MagicMock()
    residue_group.__len__ = 3
    residue = residue_group.residues[1]

    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=3, mass=12.0, pos=(1, 0, 0)),
            _atom(index=5, mass=12.0, pos=(0, 1, 0)),
            _atom(index=7, mass=12.0, pos=(0, 0, 1)),
            _atom(index=9, mass=12.0, pos=(0, 1, 1)),
        ],
        positions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]], dtype=float),
    )

    edge_atoms = _FakeAtomGroup(
        [
            _atom(index=3, mass=12.0, pos=(1, 0, 0)),
            _atom(index=9, mass=12.0, pos=(0, 1, 1)),
        ],
        positions=np.array([[1, 0, 0], [0, 1, 1]], dtype=float),
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return heavy_atoms
        if q.startswith("resindex "):
            return edge_atoms
        if q.startswith("index "):
            return [heavy_atoms[1]]

    residue_group.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax, "get_chain", lambda residue, first, last: [heavy_atoms[1], heavy_atoms[2]]
    )
    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), np.array([1.0, 2.0, 3.0])),
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_UA_axes(
        data_container=residue_group, index=1, res_position=0
    )

    assert trans_axes.shape == (3, 3)
    assert rot_axes.shape == (3, 3)
    assert moi.shape == (3,)
    assert rot_center.shape == (3,)


def test_get_ua_axes_bonded_axes_first_resid(monkeypatch):
    ax = AxesCalculator()
    residue_group = MagicMock()
    residue_group.__len__ = 2
    residue = residue_group.residues[0]
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
            _atom(index=1, mass=12.0, pos=(0, 1, 0)),
            _atom(index=2, mass=12.0, pos=(0, 0, 1)),
        ],
    )

    residue.atoms[0] = heavy_atoms[0]
    residue.atoms[0].position = heavy_atoms[0].position
    edge_atom_set = [heavy_atoms[2]]
    bonded_atoms = [heavy_atoms[1]]

    def _select_atoms(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return heavy_atoms
        if q.startswith("index"):
            return [heavy_atoms[0]]
        if q.startswith("resindex "):
            return edge_atom_set
        if q.startswith("(mass 2 to 999) and bonded index "):
            return bonded_atoms

    residue_group.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), np.array([1.0, 1.0, 1.0])),
    )
    trans_axes, rot_axes, rot_center, moi = ax.get_UA_axes(
        data_container=residue_group, index=0, res_position=-1
    )

    assert trans_axes.shape == (3, 3)
    assert rot_axes.shape == (3, 3)
    assert moi.shape == (3,)
    assert rot_center.shape == (3,)


def test_get_ua_axes_bonded_axes_last_resid(monkeypatch):
    ax = AxesCalculator()
    residue_group = MagicMock()
    residue_group.__len__ = 2
    residue = residue_group.residues[0]
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
            _atom(index=1, mass=12.0, pos=(0, 1, 0)),
            _atom(index=2, mass=12.0, pos=(0, 0, 1)),
        ],
    )

    residue.atoms[0] = heavy_atoms[0]
    residue.atoms[0].position = heavy_atoms[0].position
    edge_atom_set = [heavy_atoms[0]]
    bonded_atoms = [heavy_atoms[1]]

    def _select_atoms(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return heavy_atoms
        if q.startswith("index"):
            return [heavy_atoms[0]]
        if q.startswith("resindex "):
            return edge_atom_set
        if q.startswith("(mass 2 to 999) and bonded index "):
            return bonded_atoms

    residue_group.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), np.array([1.0, 1.0, 1.0])),
    )
    trans_axes, rot_axes, rot_center, moi = ax.get_UA_axes(
        data_container=residue_group, index=0, res_position=1
    )

    assert trans_axes.shape == (3, 3)
    assert rot_axes.shape == (3, 3)
    assert moi.shape == (3,)
    assert rot_center.shape == (3,)


def test_get_chain(monkeypatch):
    ax = AxesCalculator()
    residue = MagicMock()
    residue.__len__ = 5
    heavy_atoms = [
        _atom(index=0, mass=12.0, pos=(1, 0, 0)),
        _atom(index=1, mass=12.0, pos=(0, 1, 0)),
        _atom(index=2, mass=12.0, pos=(0, 0, 1)),
        _atom(index=3, mass=12.0, pos=(0, 1, 1)),
        _atom(index=4, mass=12.0, pos=(1, 0, 1)),
    ]

    def _select_atoms(q):
        if q.endswith("bonded index 0"):
            return [heavy_atoms[1]]
        elif q.endswith("bonded index 1"):
            return [heavy_atoms[0], heavy_atoms[2]]
        elif q.endswith("bonded index 2"):
            return [heavy_atoms[1], heavy_atoms[3]]
        elif q.endswith("bonded index 3"):
            return [heavy_atoms[2], heavy_atoms[4]]
        elif q.endswith("bonded index 4"):
            return [heavy_atoms[3]]
        elif q.endswith("not index 0"):
            return heavy_atoms[1:]

    residue.atoms.select_atoms.side_effect = _select_atoms
    residue.atoms.indices = np.arange(5)

    chain = ax.get_chain(residue=residue, first=heavy_atoms[0], last=heavy_atoms[-1])

    assert len(chain) == 3


def test_get_UA_axes_raises_when_only_rot_axes_fail(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.atoms.atoms.moment_of_inertia.return_value = np.diag([3.0, 2.0, 1.0])
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    heavy_atoms = [
        _atom(index=0, mass=12.0, pos=(1, 0, 0)),
        _atom(index=1, mass=12.0, pos=(0, 1, 0)),
    ]
    u.residues = [heavy_atoms]

    def _sel(q):
        if q == "mass 2 to 999":
            return heavy_atoms
        if q.startswith("index"):
            return [heavy_atoms[0]]

    u.select_atoms.side_effect = _sel
    u.atoms.select_atoms.side_effect = _sel

    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    monkeypatch.setattr(ax, "get_bonded_axes", lambda **kwargs: (None, None))

    with pytest.raises(ValueError):
        ax.get_UA_axes(u, index=0, res_position=None)


def test_get_ua_axes_bonded_terminal_2_points(monkeypatch):
    ax = AxesCalculator()
    residue_group = MagicMock()
    residue_group.__len__ = 2
    residue = residue_group.residues[1]
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
            _atom(index=1, mass=12.0, pos=(0, 1, 0)),
        ],
    )
    edge_atom_set = [heavy_atoms[0]]
    bonded_atoms = [heavy_atoms[1]]

    def _select_atoms(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return heavy_atoms
        if q.startswith("index"):
            return [heavy_atoms[0]]
        if q.startswith("resindex "):
            return edge_atom_set
        if q.startswith("(mass 2 to 999) and bonded index "):
            return bonded_atoms

    residue_group.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), np.array([1.0, 1.0, 1.0])),
    )

    monkeypatch.setattr(
        ax, "get_bonded_vector_axes", lambda a, b_list, c, dimensions: 2 * np.eye(3)
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_UA_axes(
        data_container=residue_group, index=0, res_position=1
    )

    assert np.allclose(trans_axes, 2 * np.eye(3))
    assert np.allclose(rot_axes, np.eye(3))
    assert np.allclose(rot_center, [1, 0, 0])
    assert np.allclose(moi, np.array([1, 1, 1]))


def test_get_ua_axes_non_terminal_2_atoms(monkeypatch):
    ax = AxesCalculator()
    residue_group = MagicMock()
    residue_group.__len__ = 3
    residue = residue_group.residues[1]
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 1, 1)),
            _atom(index=1, mass=12.0, pos=(3, 3, 3)),
        ],
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return heavy_atoms
        if q.startswith("index"):
            return [heavy_atoms[0]]
        if q.startswith("resindex "):
            return heavy_atoms

    residue_group.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), 3 * np.eye(3)),
    )
    monkeypatch.setattr(
        ax, "get_bonded_vector_axes", lambda a, b_list, c, dimensions: 2 * np.eye(3)
    )
    monkeypatch.setattr(ax, "get_chain", lambda residue, first, last: [])
    trans_axes, rot_axes, rot_center, moi = ax.get_UA_axes(
        data_container=residue_group, index=0, res_position=0
    )

    assert np.allclose(trans_axes, 2 * np.eye(3))
    assert np.allclose(rot_axes, np.eye(3))
    assert np.allclose(rot_center, [1, 1, 1])
    assert np.allclose(moi, 3 * np.eye(3))


def test_get_residue_axes_non_terminal_2_atoms(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    residue = u.select_atoms("resindex 5")
    residue.__len__.return_value = 2
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )
    uas = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 1, 1)),
            _atom(index=1, mass=12.0, pos=(3, 3, 3)),
        ],
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            return uas
        if q.startswith("resindex 5 and (bonded resindex"):
            return uas

    u.atoms.select_atoms.side_effect = _select_atoms
    residue.select_atoms.side_effect = residue
    monkeypatch.setattr(ax, "get_chain", lambda residue, first, last: [])
    monkeypatch.setattr(
        ax, "get_bonded_vector_axes", lambda a, b_list, c, dimensions: 2 * np.eye(3)
    )
    monkeypatch.setattr(
        ax,
        "get_custom_residue_moment_of_inertia",
        lambda center_of_mass, positions, masses, custom_rot_axes, dimensions: np.array(
            [1, 1, 1]
        ),
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_residue_axes(
        data_container=u,
        index=5,
        relative_index=0,
    )

    assert np.allclose(trans_axes, np.eye(3))
    assert np.allclose(rot_axes, 2 * np.eye(3))
    assert np.allclose(rot_center, [2, 2, 2])
    assert np.allclose(moi, [1, 1, 1])


def test_get_residue_axes_terminal_2_atoms(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    residue = u.select_atoms("resindex 0")
    residue.__len__.return_value = 2
    uas = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
            _atom(index=1, mass=12.0, pos=(0, 1, 0)),
            _atom(index=2, mass=12.0, pos=(0, 0, 1)),
        ],
    )
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            return uas
        if q.startswith("resindex 0 and (bonded resindex"):
            # the edge atom
            return [uas[2]]
        if q.startswith("(mass 2 to 999) and bonded index "):
            return uas[0:2]

    u.atoms.select_atoms.side_effect = _select_atoms
    residue.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax, "get_bonded_vector_axes", lambda a, b_list, c, dimensions: 2 * np.eye(3)
    )
    monkeypatch.setattr(
        ax,
        "get_custom_residue_moment_of_inertia",
        lambda center_of_mass, positions, masses, custom_rot_axes, dimensions: np.array(
            [1, 1, 1]
        ),
    )
    trans_axes, rot_axes, rot_center, moi = ax.get_residue_axes(
        data_container=u,
        index=0,
        relative_index=0,
    )
    assert np.allclose(trans_axes, np.eye(3))
    assert np.allclose(rot_axes, 2 * np.eye(3))
    assert np.allclose(rot_center, [0, 0, 1])
    assert np.allclose(moi, [1, 1, 1])


def test_get_res_axes_terminal_1_atom(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    residue = u.select_atoms("resindex 0")
    residue.__len__.return_value = 1
    uas = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
        ],
    )
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            return uas
        if q.startswith("(mass 2 to 999) and bonded index "):
            return []
        if q.startswith("resindex "):
            return [uas[0]]

    u.atoms.select_atoms.side_effect = _select_atoms
    residue.select_atoms.side_effect = _select_atoms
    residue.atoms.select_atoms.side_effect = _select_atoms

    monkeypatch.setattr(
        ax,
        "get_custom_residue_moment_of_inertia",
        lambda center_of_mass, positions, masses, custom_rot_axes, dimensions: np.array(
            [1, 1, 1]
        ),
    )
    trans_axes, rot_axes, rot_center, moi = ax.get_residue_axes(
        data_container=u,
        index=0,
        relative_index=0,
    )
    assert np.allclose(trans_axes, np.eye(3))
    assert np.allclose(rot_axes, np.eye(3))
    assert np.allclose(rot_center, [1, 0, 0])
    assert np.allclose(moi, [1, 1, 1])


def test_get_UA_axes_terminal_1_atom(monkeypatch):
    ax = AxesCalculator()
    residue_group = MagicMock()
    residue_group.__len__ = 2
    residue = residue_group.residues[1]
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 1, 1)),
            _atom(index=1, mass=12.0, pos=(3, 3, 3)),
        ],
    )

    def _select_atoms_data_container(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return heavy_atoms
        if q.startswith("resindex "):
            return [heavy_atoms[0]]

    def _select_atoms_residue(q):
        if q == "mass 2 to 999":
            # return heavy atoms group
            return [heavy_atoms[0]]
        if q.startswith("index"):
            return [heavy_atoms[0]]
        if q.startswith("(mass 2 to 999) and bonded "):
            return []

    residue_group.select_atoms.side_effect = _select_atoms_data_container
    residue_group.atoms.select_atoms.side_effect = _select_atoms_data_container
    residue.select_atoms.side_effect = _select_atoms_residue
    residue.atoms.select_atoms.side_effect = _select_atoms_residue
    monkeypatch.setattr(
        ax,
        "get_bonded_axes",
        lambda system, atom, dimensions: (np.eye(3), 3 * np.eye(3)),
    )
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: 2 * np.eye(3)
    )

    trans_axes, rot_axes, rot_center, moi = ax.get_UA_axes(
        data_container=residue_group, index=0, res_position=-1
    )

    assert np.allclose(trans_axes, 2 * np.eye(3))
    assert np.allclose(rot_axes, np.eye(3))
    assert np.allclose(rot_center, [1, 1, 1])
    assert np.allclose(moi, 3 * np.eye(3))


def test_get_terminal_axes_1point(monkeypatch):
    ax = AxesCalculator()
    residue = MagicMock()
    residue.__len__ = 1
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
        ],
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            return heavy_atoms
        if q.startswith("(mass 2 to 999) and bonded"):
            return []

    residue.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax, "get_principal_axes_from_group", lambda *_a, **_k: np.eye(3)
    )
    centre, axes = ax.get_terminal_axes(
        residue, heavy_atoms[0], dimensions=np.array([1, 1, 1])
    )
    assert np.allclose(centre, [1, 0, 0])
    assert np.allclose(axes, np.eye(3))


def test_get_terminal_axes_2points(monkeypatch):
    ax = AxesCalculator()
    residue = MagicMock()
    residue.__len__ = 2
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
            _atom(index=1, mass=12.0, pos=(0, 1, 0)),
        ],
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            return heavy_atoms
        if q.startswith("(mass 2 to 999) and bonded"):
            return [heavy_atoms[1]]

    residue.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax, "get_bonded_vector_axes", lambda a, b_list, c, dimensions: 2 * np.eye(3)
    )
    centre, axes = ax.get_terminal_axes(
        residue, heavy_atoms[0], dimensions=np.array([1, 1, 1])
    )
    assert np.allclose(centre, [1, 0, 0])
    assert np.allclose(axes, 2 * np.eye(3))


def test_get_terminal_axes_3points(monkeypatch):
    ax = AxesCalculator()
    residue = MagicMock()
    residue.__len__ = 4
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(0, 0, 0)),
            _atom(index=1, mass=12.0, pos=(1, 1, 0)),
            _atom(index=2, mass=12.0, pos=(1, 0, 0)),
            _atom(index=3, mass=14.0, pos=(3, 0, 0)),
        ],
    )

    def _select_atoms(q):
        if q == "mass 2 to 999":
            return heavy_atoms
        if q.startswith("(mass 2 to 999) and bonded"):
            return [heavy_atoms[1]]

    residue.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(
        ax, "get_residue_custom_axes", lambda edges, center: ([1, 0, 0], 3 * np.eye(3))
    )

    centre, axes = ax.get_terminal_axes(
        residue, heavy_atoms[0], dimensions=np.array([1, 1, 1])
    )

    assert np.allclose(centre, [1, 0, 0])
    assert np.allclose(axes, 3 * np.eye(3))


def test_get_non_terminal_axes_2points(monkeypatch):
    ax = AxesCalculator()
    residue = MagicMock()
    residue.__len__ = 2
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(2, 0, 0)),
            _atom(index=1, mass=12.0, pos=(0, 2, 0)),
        ],
    )
    monkeypatch.setattr(ax, "get_chain", lambda residue, first, last: [])
    monkeypatch.setattr(
        ax, "get_bonded_vector_axes", lambda a, b_list, c, dimensions: 2 * np.eye(3)
    )
    centre, axes = ax.get_non_terminal_axes(
        residue=residue, edges=heavy_atoms, dimensions=np.array([1, 1, 1])
    )
    assert np.allclose(centre, [1, 1, 0])
    assert np.allclose(axes, 2 * np.eye(3))


def test_get_non_terminal_axes_3points(monkeypatch):
    ax = AxesCalculator()
    residue = MagicMock()
    residue.__len__ = 4
    heavy_atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(0, 0, 0)),
            _atom(index=1, mass=12.0, pos=(1, 0, 0)),
            _atom(index=2, mass=14.0, pos=(3, 0, 0)),
            _atom(index=3, mass=12.0, pos=(1, 1, 0)),
        ],
    )
    edges = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(0, 0, 0)),
            _atom(index=3, mass=12.0, pos=(1, 1, 0)),
        ],
    )
    monkeypatch.setattr(
        ax, "get_chain", lambda residue, first, last: [heavy_atoms[1], heavy_atoms[2]]
    )
    monkeypatch.setattr(
        ax, "get_residue_custom_axes", lambda edges, center: ([1, 0, 0], 3 * np.eye(3))
    )
    centre, axes = ax.get_non_terminal_axes(
        residue=residue, edges=edges, dimensions=np.array([1, 1, 1])
    )

    assert np.allclose(centre, [1, 0, 0])
    assert np.allclose(axes, 3 * np.eye(3))


def test_get_residue_axes_all_atom_MOI(monkeypatch):
    ax = AxesCalculator()
    u = MagicMock()
    u.dimensions = np.array([10.0, 10.0, 10.0, 90, 90, 90])
    residue = u.select_atoms("resindex 1")
    residue.__len__.return_value = 5
    monkeypatch.setattr("CodeEntropy.levels.axes.make_whole", lambda _ag: None)
    atoms = _FakeAtomGroup(
        [
            _atom(index=0, mass=12.0, pos=(1, 0, 0)),
            _atom(index=1, mass=12.0, pos=(0, 1, 0)),
            _atom(index=2, mass=12.0, pos=(0, 0, 1)),
            _atom(index=3, mass=1.0, pos=(0, 0, 0)),
            _atom(index=4, mass=1.0, pos=(1, 1, 1)),
        ],
        positions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1]]),
    )
    residue.atoms.positions = atoms.positions
    residue.atoms.masses = atoms.masses

    def _select_atoms(q):
        if q.startswith("resindex"):
            return atoms[0:2]

    u.atoms.select_atoms.side_effect = _select_atoms
    monkeypatch.setattr(ax, "get_principal_axes_from_group", lambda _ag: 2 * np.eye(3))
    monkeypatch.setattr(
        ax,
        "get_non_terminal_axes",
        lambda residue, edges, dimensions: ([1, 0, 0], np.eye(3)),
    )
    trans_axes, rot_axes, rot_center, moi = ax.get_residue_axes(
        data_container=u, index=1, relative_index=0, residue=None
    )

    assert np.allclose(trans_axes, 2 * np.eye(3))
    assert np.allclose(rot_axes, np.eye(3))
    assert np.allclose(rot_center, [1, 0, 0])
    assert np.allclose(moi, [26, 38, 38])


def _methanol_universe():
    """Methanol C0, O1, H2 (hydroxyl), H3-H5 (methyl), in a large box."""
    universe = mda.Universe.empty(
        n_atoms=6, n_residues=1, atom_resindex=[0] * 6, trajectory=True
    )
    universe.add_TopologyAttr("masses", [12.011, 15.999, 1.008, 1.008, 1.008, 1.008])
    universe.add_TopologyAttr("bonds", [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)])
    universe.dimensions = [50.0, 50.0, 50.0, 90.0, 90.0, 90.0]
    return universe


def _methanol_positions(coh_degrees):
    """Methanol positions with the hydroxyl H in the xy plane at C-O-H angle."""
    angle = np.radians(coh_degrees)
    carbon = np.array([-1.43, 0.0, 0.0])
    hydroxyl_h = 0.96 * np.array([-np.cos(angle), np.sin(angle), 0.0])
    methyl_h = carbon + np.array(
        [[-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89], [-0.36, 1.03, 0.0]]
    )
    return np.vstack([carbon, [0.0, 0.0, 0.0], hydroxyl_h, methyl_h]) + 25.0


def _propanoic_acid_universe():
    """Propanoic acid C0-C1-C2(=O3)-O4-H10, H5-H7 on C0 and H8-H9 on C1."""
    universe = mda.Universe.empty(
        n_atoms=11, n_residues=1, atom_resindex=[0] * 11, trajectory=True
    )
    universe.add_TopologyAttr("masses", [12.011] * 3 + [15.999] * 2 + [1.008] * 6)
    universe.add_TopologyAttr(
        "bonds",
        [(0, 1), (1, 2), (2, 3), (2, 4), (0, 5), (0, 6), (0, 7), (1, 8), (1, 9)]
        + [(4, 10)],
    )
    universe.dimensions = [50.0, 50.0, 50.0, 90.0, 90.0, 90.0]
    universe.atoms.positions = (
        np.array(
            [
                [-1.378, 0.495, -0.065],
                [-0.283, -0.543, 0.103],
                [1.087, 0.063, -0.030],
                [1.367, 1.229, -0.245],
                [2.059, -0.858, 0.119],
                [-1.293, 1.283, 0.690],
                [-2.364, 0.029, 0.035],
                [-1.323, 0.969, -1.051],
                [-0.365, -1.002, 1.093],
                [-0.395, -1.318, -0.663],
                [2.888, -0.346, 0.013],
            ]
        )
        + 25.0
    )
    return universe


def _cached_ua_topology(heavy_atom):
    """Cached UA topology for one heavy atom, as built by the topology node."""
    bonded_heavy, bonded_light = BuildAxesTopologyNode._split_bonded_atoms(heavy_atom)
    return _ua_topology(
        heavy_atom_index=heavy_atom.index,
        ua_atom_indices=[heavy_atom.index, *bonded_light.indices],
        ua_all_atom_indices=[
            heavy_atom.index,
            *bonded_heavy.indices,
            *bonded_light.indices,
        ],
        bonded_heavy_indices=bonded_heavy.indices,
        bonded_light_indices=bonded_light.indices,
    )


def test_get_bonded_axes_single_hydrogen_zeroes_same_axis_every_frame():
    """The O-H united atom of methanol loses the moment about O-H, axis 0."""
    ax = AxesCalculator()
    universe = _methanol_universe()
    dims = universe.dimensions[:3]
    oxygen, hydrogen = universe.atoms[1], universe.atoms[2]

    old_zeroed_axes = set()
    for coh_degrees in (108.0, 150.0):
        universe.atoms.positions = _methanol_positions(coh_degrees)

        axes, moi = ax.get_bonded_axes(universe.atoms, oxygen, dims)

        bond = hydrogen.position - oxygen.position
        assert np.allclose(axes[0], bond / np.linalg.norm(bond), atol=1e-6)
        assert moi[0] == 0.0
        assert np.all(moi[1:] > 0.0)

        # the axes built from the C-O bond have their smallest moment on
        # different axes at these two angles
        old_axes = ax.get_bonded_vector_axes(
            a=oxygen.position,
            b_list=[universe.atoms[0].position],
            c=hydrogen.position,
            dimensions=dims,
        )
        old_moi = np.sum(np.cross(old_axes, bond) ** 2, axis=1)
        old_zeroed_axes.add(int(np.argmin(old_moi)))

    assert len(old_zeroed_axes) == 2


def test_get_bonded_axes_single_hydrogen_second_axis_towards_heavy_neighbour():
    ax = AxesCalculator()
    universe = _methanol_universe()
    universe.atoms.positions = _methanol_positions(108.0)
    oxygen, carbon = universe.atoms[1], universe.atoms[0]

    axes, _ = ax.get_bonded_axes(universe.atoms, oxygen, universe.dimensions[:3])

    oxygen_carbon = carbon.position - oxygen.position
    in_plane = oxygen_carbon - np.dot(oxygen_carbon, axes[0]) * axes[0]
    assert np.allclose(axes[2], in_plane / np.linalg.norm(in_plane), atol=1e-6)
    assert np.allclose(np.abs(axes[1]), [0.0, 0.0, 1.0], atol=1e-6)


def test_get_bonded_axes_collinear_single_hydrogen_raises():
    ax = AxesCalculator()
    universe = _methanol_universe()
    universe.atoms.positions = _methanol_positions(180.0)

    with pytest.raises(ValueError):
        ax.get_bonded_axes(universe.atoms, universe.atoms[1], universe.dimensions[:3])


def test_get_bonded_axes_unchanged_without_single_hydrogen():
    """United atoms with zero or at least two hydrogens keep their axes."""
    ax = AxesCalculator()
    universe = _propanoic_acid_universe()
    dims = universe.dimensions[:3]
    atoms = universe.atoms

    # (heavy atom, b_list, c) of the bonded vector axes before single-hydrogen
    # united atoms had their own axes
    cases = [
        (0, [1], 5),  # CH3: one heavy, three H
        (1, [0, 2], 2),  # CH2: two heavy, two H
        (2, [1, 3, 4], 3),  # C: three heavy, no H
        (3, [2], None),  # =O: one heavy, no H
    ]
    for index, b_indices, c_index in cases:
        heavy_atom = atoms[index]
        ua = heavy_atom + atoms.select_atoms(f"mass 1 to 1.1 and bonded index {index}")
        expected_axes = ax.get_bonded_vector_axes(
            a=heavy_atom.position,
            b_list=atoms[b_indices].positions,
            c=np.zeros(3) if c_index is None else atoms[c_index].position,
            dimensions=dims,
        )
        expected_moi = ax.get_custom_moment_of_inertia(
            UA=ua,
            custom_rotation_axes=expected_axes,
            center_of_mass=heavy_atom.position,
            dimensions=dims,
        )
        expected_axes = ax.get_flipped_axes(
            ua, expected_axes, heavy_atom.position, dims
        )

        axes, moi = ax.get_bonded_axes(atoms, heavy_atom, dims)

        np.testing.assert_array_equal(axes, expected_axes)
        np.testing.assert_array_equal(moi, expected_moi)


def test_get_bonded_axes_single_hydrogen_without_heavy_neighbour_uses_principal():
    """A molecule HX keeps its principal axes and principal moments."""
    ax = AxesCalculator()
    universe = mda.Universe.empty(
        n_atoms=2, n_residues=1, atom_resindex=[0, 0], trajectory=True
    )
    universe.add_TopologyAttr("masses", [35.45, 1.008])
    universe.add_TopologyAttr("bonds", [(0, 1)])
    universe.dimensions = [50.0, 50.0, 50.0, 90.0, 90.0, 90.0]
    universe.atoms.positions = np.array([[25.0, 25.0, 25.0], [26.0, 25.6, 25.4]])
    expected_axes, expected_moi = ax.get_molecule_axes(universe.atoms)

    axes, moi = ax.get_bonded_axes(
        universe.atoms, universe.atoms[0], universe.dimensions[:3]
    )

    np.testing.assert_allclose(axes, expected_axes)
    np.testing.assert_allclose(moi, expected_moi)


@pytest.mark.parametrize("heavy_index", [0, 1, 2, 3, 4])
def test_get_bonded_axes_from_topology_matches_local_axes(heavy_index):
    ax = AxesCalculator()
    universe = _propanoic_acid_universe()
    dims = universe.dimensions[:3]
    heavy_atom = universe.atoms[heavy_index]

    local_axes, local_moi = ax.get_bonded_axes(universe.atoms, heavy_atom, dims)
    cached_axes, cached_moi = ax.get_bonded_axes_from_topology(
        u=universe,
        heavy_atom=heavy_atom,
        topology=_cached_ua_topology(heavy_atom),
        dimensions=dims,
    )

    np.testing.assert_array_equal(cached_axes, local_axes)
    np.testing.assert_array_equal(cached_moi, local_moi)


def test_get_bonded_axes_from_topology_single_hydrogen_zeroes_axis_along_bond():
    ax = AxesCalculator()
    universe = _methanol_universe()
    dims = universe.dimensions[:3]
    oxygen, hydrogen = universe.atoms[1], universe.atoms[2]
    topology = _cached_ua_topology(oxygen)

    for coh_degrees in (108.0, 150.0):
        universe.atoms.positions = _methanol_positions(coh_degrees)

        axes, moi = ax.get_bonded_axes_from_topology(
            u=universe, heavy_atom=oxygen, topology=topology, dimensions=dims
        )

        bond = hydrogen.position - oxygen.position
        assert np.allclose(axes[0], bond / np.linalg.norm(bond), atol=1e-6)
        assert moi[0] == 0.0
        assert np.all(moi[1:] > 0.0)


def test_get_bonded_axes_single_hydrogen_two_heavy_neighbours_uses_lowest_index():
    """The N-H of dimethylamine (methyl H omitted): axis 0 along N-H, axis 2
    towards C0, and the same axes from the cached topology."""
    ax = AxesCalculator()
    universe = mda.Universe.empty(
        n_atoms=4, n_residues=1, atom_resindex=[0] * 4, trajectory=True
    )
    universe.add_TopologyAttr("masses", [12.011, 14.007, 12.011, 1.008])
    universe.add_TopologyAttr("bonds", [(0, 1), (1, 2), (1, 3)])
    universe.dimensions = [50.0, 50.0, 50.0, 90.0, 90.0, 90.0]
    universe.atoms.positions = (
        np.array(
            [
                [1.188, 0.172, 0.129],
                [-0.011, 0.215, -0.698],
                [-1.202, -0.050, 0.097],
                [0.065, -0.487, -1.434],
            ]
        )
        + 25.0
    )
    dims = universe.dimensions[:3]
    nitrogen = universe.atoms[1]

    axes, moi = ax.get_bonded_axes(universe.atoms, nitrogen, dims)
    cached_axes, cached_moi = ax.get_bonded_axes_from_topology(
        u=universe,
        heavy_atom=nitrogen,
        topology=_cached_ua_topology(nitrogen),
        dimensions=dims,
    )

    bond = universe.atoms[3].position - nitrogen.position
    nitrogen_carbon = universe.atoms[0].position - nitrogen.position
    in_plane = nitrogen_carbon - np.dot(nitrogen_carbon, axes[0]) * axes[0]
    assert np.allclose(axes[0], bond / np.linalg.norm(bond), atol=1e-6)
    assert np.allclose(axes[2], in_plane / np.linalg.norm(in_plane), atol=1e-6)
    assert moi[0] == 0.0
    np.testing.assert_array_equal(cached_axes, axes)
    np.testing.assert_array_equal(cached_moi, moi)
