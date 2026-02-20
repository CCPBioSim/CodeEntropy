import pytest

from CodeEntropy.config.runtime import CodeEntropyRunner


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_validate_required_args_missing_traj_raises():
    args = Args(top_traj_file=None, selection_string="all")

    with pytest.raises(ValueError):
        CodeEntropyRunner._validate_required_args(args)


def test_validate_required_args_missing_selection_raises():
    args = Args(top_traj_file=["a"], selection_string=None)

    with pytest.raises(ValueError):
        CodeEntropyRunner._validate_required_args(args)


def test_validate_required_args_ok():
    args = Args(top_traj_file=["a"], selection_string="all")
    CodeEntropyRunner._validate_required_args(args)
