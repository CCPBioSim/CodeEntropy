import logging

import pytest


def test_validate_inputs_valid_does_not_raise(resolver, dummy_universe, make_args):
    args = make_args()
    resolver.validate_inputs(dummy_universe, args)


def test_check_input_start_raises_when_start_exceeds_trajectory(resolver, make_args):
    u = type("U", (), {"trajectory": [None] * 10})()
    args = make_args(start=11)

    with pytest.raises(ValueError):
        resolver._check_input_start(u, args)


def test_check_input_end_raises_when_end_exceeds_trajectory(resolver, make_args):
    u = type("U", (), {"trajectory": [None] * 10})()
    args = make_args(end=11)

    with pytest.raises(ValueError):
        resolver._check_input_end(u, args)


def test_check_input_step_negative_logs_warning(resolver, make_args, caplog):
    args = make_args(step=-1)

    with caplog.at_level(logging.WARNING):
        resolver._check_input_step(args)

    assert "Negative 'step' value" in caplog.text


@pytest.mark.parametrize("bin_width", [-1, 361])
def test_check_input_bin_width_out_of_range_raises(resolver, make_args, bin_width):
    args = make_args(bin_width=bin_width)

    with pytest.raises(ValueError):
        resolver._check_input_bin_width(args)


def test_check_input_temperature_negative_raises(resolver, make_args):
    args = make_args(temperature=-0.1)

    with pytest.raises(ValueError):
        resolver._check_input_temperature(args)


def test_check_input_force_partitioning_non_default_logs_warning(
    resolver, make_args, caplog
):
    args = make_args(force_partitioning=0.7)

    with caplog.at_level(logging.WARNING):
        resolver._check_input_force_partitioning(args)

    assert "differs from the default" in caplog.text
