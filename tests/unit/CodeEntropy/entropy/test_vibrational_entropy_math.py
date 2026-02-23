import numpy as np
import pytest

from CodeEntropy.entropy.vibrational import VibrationalEntropy


@pytest.fixture()
def run_manager():
    class RM:
        def change_lambda_units(self, x):
            return np.asarray(x)

        def get_KT2J(self, temp):
            return 1e-34

    return RM()


def test_matrix_eigenvalues_returns_complex_dtype_possible(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    m = np.array([[0.0, -1.0], [1.0, 0.0]])
    eigs = ve._matrix_eigenvalues(m)
    assert eigs.shape == (2,)


def test_frequencies_from_lambdas_filters_nonpositive_and_near_zero(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)

    lambdas = np.array([-1.0, 0.0, 1e-12, 1.0, 4.0])
    freqs = ve._frequencies_from_lambdas(lambdas, temp=298.0)

    assert freqs.size == 2
    assert np.all(freqs > 0)


def test_frequencies_from_lambdas_filters_complex(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    lambdas = np.array([1.0 + 2.0j, 9.0 + 0.0j, 16.0])

    freqs = ve._frequencies_from_lambdas(lambdas, temp=298.0)

    assert freqs.size == 2
    assert np.all(freqs > 0)


def test_entropy_components_returns_empty_when_all_invalid(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)

    ve._matrix_eigenvalues = lambda m: np.array([-1.0, 0.0, 0.0])
    comps = ve._entropy_components(np.eye(3), temp=298.0)

    assert comps.size == 0


def test_entropy_components_from_frequencies_returns_correct_shape(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)

    freqs = np.array([1.0, 2.0, 3.0], dtype=float)
    comps = ve._entropy_components_from_frequencies(freqs, temp=298.0)

    assert comps.shape == (3,)
    assert isinstance(comps, np.ndarray)

    assert np.all(comps >= 0) or np.isinf(comps).any()


def test_split_halves_even_length(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)

    arr = np.arange(10, dtype=float)
    a, b = ve._split_halves(arr)

    assert a.shape == (5,)
    assert b.shape == (5,)
    assert np.all(a == np.array([0, 1, 2, 3, 4], dtype=float))
    assert np.all(b == np.array([5, 6, 7, 8, 9], dtype=float))


def test_split_halves_odd_length_returns_empty_second(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)

    arr = np.arange(5, dtype=float)
    a, b = ve._split_halves(arr)

    assert a.shape == (5,)
    assert b.size == 0


def test_sum_components_empty_returns_zero(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    assert (
        ve._sum_components(np.array([], dtype=float), "force", highest_level=True)
        == 0.0
    )


def test_sum_components_force_highest_sums_all(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    comps = np.arange(12, dtype=float)
    assert ve._sum_components(comps, "force", highest_level=True) == pytest.approx(
        float(np.sum(comps))
    )


def test_sum_components_force_not_highest_drops_first_six(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    comps = np.arange(12, dtype=float)
    assert ve._sum_components(comps, "force", highest_level=False) == pytest.approx(
        float(np.sum(comps[6:]))
    )


def test_sum_components_torque_sums_all(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    comps = np.arange(12, dtype=float)
    assert ve._sum_components(comps, "torque", highest_level=False) == pytest.approx(
        float(np.sum(comps))
    )


def test_sum_components_forcetorque_trans_uses_first_three(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    comps = np.arange(10, dtype=float)
    assert ve._sum_components(
        comps, "forcetorqueTRANS", highest_level=False
    ) == pytest.approx(float(np.sum(comps[:3])))


def test_sum_components_forcetorque_rot_uses_after_three(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    comps = np.arange(10, dtype=float)
    assert ve._sum_components(
        comps, "forcetorqueROT", highest_level=False
    ) == pytest.approx(float(np.sum(comps[3:])))


def test_sum_components_unknown_matrix_type_raises(run_manager):
    ve = VibrationalEntropy(run_manager=run_manager)
    comps = np.arange(6, dtype=float)

    with pytest.raises(ValueError):
        ve._sum_components(comps, "nope", highest_level=True)


def test_vibrational_entropy_calculation_end_to_end_returns_float(
    run_manager, monkeypatch
):
    ve = VibrationalEntropy(run_manager=run_manager)

    monkeypatch.setattr(ve, "_matrix_eigenvalues", lambda m: np.array([1.0, 2.0, 3.0]))
    monkeypatch.setattr(ve, "_convert_lambda_units", lambda x: np.asarray(x))
    monkeypatch.setattr(
        ve, "_frequencies_from_lambdas", lambda lambdas, temp: np.array([1.0, 2.0, 3.0])
    )

    out = ve.vibrational_entropy_calculation(
        np.eye(3), matrix_type="torque", temp=298.0, highest_level=False
    )

    assert isinstance(out, float)
    assert out >= 0.0
