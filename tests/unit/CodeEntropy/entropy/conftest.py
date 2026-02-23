from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class FakeResidue:
    def __init__(self, resname="RES"):
        self.resname = resname


class FakeMol:
    def __init__(self, residues=None):
        self.residues = residues or [FakeResidue("ALA"), FakeResidue("GLY")]


class FakeAtoms:
    def __init__(self, fragments):
        self.fragments = fragments


class FakeUniverse:
    def __init__(self, n_frames=10, fragments=None):
        self.trajectory = list(range(n_frames))
        self.atoms = FakeAtoms(fragments or [FakeMol()])


@pytest.fixture()
def args():
    # Only fields used by entropy modules.
    return SimpleNamespace(
        temperature=298.0,
        bin_width=30,
        grouping="molecules",
        water_entropy=False,
        combined_forcetorque=True,
        selection_string="all",
        start=0,
        end=-1,
        step=1,
    )


@pytest.fixture()
def reporter():
    return MagicMock()


@pytest.fixture()
def run_manager():
    rm = MagicMock()
    rm.change_lambda_units.side_effect = lambda x: x
    rm.get_KT2J.return_value = 2.479e-21
    return rm


@pytest.fixture()
def reduced_universe():
    return FakeUniverse(n_frames=12, fragments=[FakeMol()])


@pytest.fixture()
def shared_data(args, reporter, run_manager, reduced_universe):
    return {
        "args": args,
        "reporter": reporter,
        "run_manager": run_manager,
        "reduced_universe": reduced_universe,
        "universe": reduced_universe,
        "groups": {0: [0]},
        "levels": {0: ["united_atom", "residue"]},
        "start": 0,
        "end": 12,
        "step": 1,
        "n_frames": 12,
    }
