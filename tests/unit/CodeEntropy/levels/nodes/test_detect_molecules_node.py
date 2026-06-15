"""Atomic unit tests for molecule detection and grouping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode


class FakeUniverse:
    """Minimal universe exposing molecule fragments."""

    def __init__(self, n_fragments: int):
        self.atoms = SimpleNamespace(fragments=[object() for _ in range(n_fragments)])


def test_run_uses_existing_reduced_universe_and_configured_grouping():
    universe = FakeUniverse(3)
    node = DetectMoleculesNode()
    node._grouping = MagicMock()
    node._grouping.grouping_molecules.return_value = {0: [0, 1], 1: [2]}

    shared_data = {
        "reduced_universe": universe,
        "universe": FakeUniverse(99),
        "args": SimpleNamespace(grouping="molecules"),
    }

    result = node.run(shared_data)

    node._grouping.grouping_molecules.assert_called_once_with(universe, "molecules")
    assert shared_data["groups"] == {0: [0, 1], 1: [2]}
    assert shared_data["number_molecules"] == 3
    assert result == {
        "groups": {0: [0, 1], 1: [2]},
        "number_molecules": 3,
    }


def test_run_falls_back_to_universe_when_reduced_universe_missing():
    universe = FakeUniverse(2)
    node = DetectMoleculesNode()
    node._grouping = MagicMock()
    node._grouping.grouping_molecules.return_value = {0: [0], 1: [1]}

    shared_data = {
        "universe": universe,
        "args": SimpleNamespace(grouping="each"),
    }

    node.run(shared_data)

    assert shared_data["reduced_universe"] is universe
    node._grouping.grouping_molecules.assert_called_once_with(universe, "each")


def test_run_uses_default_grouping_when_args_has_no_grouping_attribute():
    universe = FakeUniverse(1)
    node = DetectMoleculesNode()
    node._grouping = MagicMock()
    node._grouping.grouping_molecules.return_value = {0: [0]}

    shared_data = {
        "reduced_universe": universe,
        "args": SimpleNamespace(),
    }

    node.run(shared_data)

    node._grouping.grouping_molecules.assert_called_once_with(universe, "each")


def test_run_requires_args():
    with pytest.raises(KeyError):
        DetectMoleculesNode().run({"reduced_universe": FakeUniverse(1)})


def test_ensure_reduced_universe_raises_when_no_universe_available():
    with pytest.raises(KeyError, match="shared_data must contain 'universe'"):
        DetectMoleculesNode()._ensure_reduced_universe({})


def test_get_grouping_strategy_reads_args_with_default():
    node = DetectMoleculesNode()

    assert (
        node._get_grouping_strategy({"args": SimpleNamespace(grouping="molecules")})
        == "molecules"
    )
    assert node._get_grouping_strategy({"args": SimpleNamespace()}) == "each"


def test_count_molecules_counts_fragments():
    assert DetectMoleculesNode._count_molecules(FakeUniverse(4)) == 4
