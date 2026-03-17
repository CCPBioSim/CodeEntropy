from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from CodeEntropy.levels.forces import ForceTorqueCalculator, TorqueInputs


def test_get_weighted_forces_applies_partitioning_when_highest_level():
    calc = ForceTorqueCalculator()

    bead = MagicMock()
    bead.atoms = [
        SimpleNamespace(force=np.array([1.0, 0.0, 0.0])),
        SimpleNamespace(force=np.array([0.0, 2.0, 0.0])),
    ]
    bead.total_mass.return_value = 4.0

    trans_axes = np.eye(3)

    out = calc.get_weighted_forces(
        bead=bead,
        trans_axes=trans_axes,
        highest_level=True,
        force_partitioning=2.0,
    )

    np.testing.assert_allclose(out, np.array([1.0, 2.0, 0.0]))


def test_get_weighted_forces_no_partitioning_when_not_highest_level():
    calc = ForceTorqueCalculator()

    bead = MagicMock()
    bead.atoms = [SimpleNamespace(force=np.array([2.0, 0.0, 0.0]))]
    bead.total_mass.return_value = 4.0

    out = calc.get_weighted_forces(
        bead=bead,
        trans_axes=np.eye(3),
        highest_level=False,
        force_partitioning=999.0,
    )

    np.testing.assert_allclose(out, np.array([1.0, 0.0, 0.0]))


def test_get_weighted_forces_raises_on_non_positive_mass():
    calc = ForceTorqueCalculator()

    bead = MagicMock()
    bead.atoms = [SimpleNamespace(force=np.array([1.0, 0.0, 0.0]))]
    bead.total_mass.return_value = 0.0

    with pytest.raises(ValueError):
        calc.get_weighted_forces(
            bead=bead,
            trans_axes=np.eye(3),
            highest_level=False,
            force_partitioning=1.0,
        )


def test_get_weighted_torques_uses_axes_manager_displacements(axes_manager_identity):
    calc = ForceTorqueCalculator()

    bead = MagicMock()
    bead.positions = np.array([[1.0, 0.0, 0.0]])
    bead.forces = np.array([[0.0, 1.0, 0.0]])

    out = calc.get_weighted_torques(
        bead=bead,
        rot_axes=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        force_partitioning=1.0,
        moment_of_inertia=np.array([4.0, 9.0, 16.0]),
        axes_manager=axes_manager_identity,
        box=None,
    )

    np.testing.assert_allclose(out, np.array([0.0, 0.0, 0.25]))


def test_get_weighted_torques_skips_zero_or_invalid_moi_components(
    axes_manager_identity,
):
    calc = ForceTorqueCalculator()

    bead = MagicMock()
    bead.positions = np.array([[1.0, 0.0, 0.0]])
    bead.forces = np.array([[0.0, 1.0, 0.0]])

    out = calc.get_weighted_torques(
        bead=bead,
        rot_axes=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        force_partitioning=1.0,
        moment_of_inertia=np.array([0.0, -1.0, 16.0]),
        axes_manager=axes_manager_identity,
        box=None,
    )

    np.testing.assert_allclose(out, np.array([0.0, 0.0, 0.25]))


def test_compute_frame_covariance_outer_products():
    calc = ForceTorqueCalculator()

    f_vecs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])]
    t_vecs = [np.array([0.0, 0.0, 3.0])]

    F, T = calc.compute_frame_covariance(f_vecs, t_vecs)

    assert F.shape == (6, 6)
    assert T.shape == (3, 3)

    flat_f = np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(F, np.outer(flat_f, flat_f))

    flat_t = np.array([0.0, 0.0, 3.0])
    np.testing.assert_allclose(T, np.outer(flat_t, flat_t))


def test_outer_second_moment_empty_returns_0x0():
    calc = ForceTorqueCalculator()
    F, T = calc.compute_frame_covariance([], [])
    assert F.shape == (0, 0)
    assert T.shape == (0, 0)


def test_outer_second_moment_raises_if_vector_not_length_3():
    calc = ForceTorqueCalculator()
    with pytest.raises(ValueError):
        calc.compute_frame_covariance([np.array([1.0, 2.0])], [])


def test_compute_weighted_force_rejects_wrong_axes_shape():
    ft = ForceTorqueCalculator()
    bead = MagicMock()
    bead.atoms = []
    bead.total_mass.return_value = 10.0

    with pytest.raises(ValueError):
        ft._compute_weighted_force(
            bead,
            trans_axes=np.zeros((2, 2)),
            apply_partitioning=False,
            force_partitioning=1.0,
        )


def test_compute_weighted_torque_rejects_wrong_rot_axes_shape():
    ft = ForceTorqueCalculator()
    bead = MagicMock()
    bead.positions = np.zeros((1, 3))
    bead.forces = np.zeros((1, 3))

    inputs = TorqueInputs(
        rot_axes=np.zeros((2, 2)),
        center=np.zeros(3),
        moment_of_inertia=np.ones(3),
        axes_manager=MagicMock(),
        box=None,
        force_partitioning=1.0,
    )

    with pytest.raises(ValueError):
        ft._compute_weighted_torque(bead, inputs)


def test_compute_weighted_torque_rejects_wrong_moi_shape():
    ft = ForceTorqueCalculator()
    bead = MagicMock()
    bead.positions = np.zeros((1, 3))
    bead.forces = np.zeros((1, 3))

    inputs = TorqueInputs(
        rot_axes=np.eye(3),
        center=np.zeros(3),
        moment_of_inertia=np.ones(2),
        axes_manager=MagicMock(),
        box=None,
        force_partitioning=1.0,
    )

    with pytest.raises(ValueError):
        ft._compute_weighted_torque(bead, inputs)


def test_compute_weighted_torque_skips_zero_torque_and_nonpositive_moi(monkeypatch):
    ft = ForceTorqueCalculator()

    bead = MagicMock()
    bead.positions = np.array([[1.0, 0.0, 0.0]])
    bead.forces = np.array([[0.0, 0.0, 0.0]])

    monkeypatch.setattr(
        ft,
        "_displacements_relative_to_center",
        lambda **kwargs: np.array([[1.0, 0.0, 0.0]]),
    )

    inputs = TorqueInputs(
        rot_axes=np.eye(3),
        center=np.zeros(3),
        moment_of_inertia=np.array([0.0, -1.0, 2.0]),
        axes_manager=MagicMock(),
        box=None,
        force_partitioning=1.0,
    )

    out = ft._compute_weighted_torque(bead, inputs)
    assert np.allclose(out, np.zeros(3))


def test_compute_weighted_torque_skips_nonpositive_moi_components():
    calc = ForceTorqueCalculator()

    bead = SimpleNamespace(
        positions=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
        forces=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
    )

    inputs = SimpleNamespace(
        center=np.array([0.0, 0.0, 0.0]),
        rot_axes=np.eye(3),
        moment_of_inertia=np.array([1.0, 0.0, -1.0], dtype=float),  # triggers skips
        force_partitioning=1.0,
        axes_manager=None,
        box=np.array([10.0, 10.0, 10.0], dtype=float),
    )

    calc._displacements_relative_to_center = lambda **kwargs: bead.positions

    weighted = calc._compute_weighted_torque(bead=bead, inputs=inputs)

    assert np.allclose(weighted, np.array([1.0, 0.0, 0.0]))


def test_displacements_requires_axes_manager():
    with pytest.raises(ValueError, match="axes_manager must be provided"):
        ForceTorqueCalculator._displacements_relative_to_center(
            center=np.zeros(3),
            positions=np.zeros((1, 3)),
            axes_manager=None,
            box=None,
        )


def test_displacements_calls_axes_manager_get_vector():
    axes_manager = MagicMock()
    expected = np.array([[1.0, 2.0, 3.0]])
    axes_manager.get_vector.return_value = expected

    center = np.zeros(3)
    positions = np.array([[1.0, 2.0, 3.0]])

    result = ForceTorqueCalculator._displacements_relative_to_center(
        center=center,
        positions=positions,
        axes_manager=axes_manager,
        box=None,
    )

    axes_manager.get_vector.assert_called_once_with(center, positions, None)
    assert np.array_equal(result, expected)
