import argparse as _argparse

import pytest

from CodeEntropy.config.argparse import ConfigResolver


@pytest.mark.parametrize("value", ["true", "True", "t", "yes", "1"])
def test_str2bool_true_variants(value):
    assert ConfigResolver.str2bool(value) is True


@pytest.mark.parametrize("value", ["false", "False", "f", "no", "0"])
def test_str2bool_false_variants(value):
    assert ConfigResolver.str2bool(value) is False


def test_str2bool_bool_passthrough():
    assert ConfigResolver.str2bool(True) is True
    assert ConfigResolver.str2bool(False) is False


def test_str2bool_non_string_non_bool_raises():
    with pytest.raises(_argparse.ArgumentTypeError):
        ConfigResolver.str2bool(123)


def test_str2bool_invalid_string_raises():
    with pytest.raises(_argparse.ArgumentTypeError):
        ConfigResolver.str2bool("maybe")
