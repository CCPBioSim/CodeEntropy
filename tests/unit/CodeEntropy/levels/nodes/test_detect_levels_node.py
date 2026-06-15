"""Atomic unit tests for hierarchy-level detection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from CodeEntropy.levels.nodes.detect_levels import DetectLevelsNode


def test_run_detects_levels_and_writes_shared_data():
    universe = object()
    node = DetectLevelsNode()
    node._hierarchy = MagicMock()
    node._hierarchy.select_levels.return_value = (
        2,
        [["united_atom"], ["united_atom", "residue"]],
    )

    shared_data = {"reduced_universe": universe}

    result = node.run(shared_data)

    node._hierarchy.select_levels.assert_called_once_with(universe)
    assert shared_data["number_molecules"] == 2
    assert shared_data["levels"] == [["united_atom"], ["united_atom", "residue"]]
    assert result == {
        "number_molecules": 2,
        "levels": [["united_atom"], ["united_atom", "residue"]],
    }


def test_run_requires_reduced_universe():
    with pytest.raises(KeyError):
        DetectLevelsNode().run({})


def test_detect_levels_delegates_to_hierarchy_builder():
    universe = object()
    node = DetectLevelsNode()
    node._hierarchy = MagicMock()
    node._hierarchy.select_levels.return_value = (1, [["polymer"]])

    assert node._detect_levels(universe) == (1, [["polymer"]])
    node._hierarchy.select_levels.assert_called_once_with(universe)
