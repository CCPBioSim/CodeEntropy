from types import SimpleNamespace
from unittest.mock import patch

import pytest

from CodeEntropy.levels.nodes.detect_molecules import DetectMoleculesNode


def test_run_sets_reduced_universe_when_missing(args, universe_with_fragments):
    node = DetectMoleculesNode()

    shared = {
        "universe": universe_with_fragments,
        "args": args,
    }

    with patch.object(node._grouping, "grouping_molecules", return_value={0: [1]}):
        out = node.run(shared)

    assert shared["reduced_universe"] is universe_with_fragments
    assert shared["groups"] == {0: [1]}
    assert shared["number_molecules"] == len(universe_with_fragments.atoms.fragments)
    assert out["number_molecules"] == shared["number_molecules"]


def test_run_uses_args_grouping_strategy(universe_with_fragments):
    node = DetectMoleculesNode()
    shared = {
        "universe": universe_with_fragments,
        "args": SimpleNamespace(grouping="molecules"),
    }

    with patch.object(
        node._grouping, "grouping_molecules", return_value={"g": [1]}
    ) as gm:
        node.run(shared)

    gm.assert_called_once()
    assert gm.call_args[0][1] == "molecules"


def test_ensure_reduced_universe_raises_if_missing_universe():
    node = DetectMoleculesNode()
    with pytest.raises(KeyError):
        node._ensure_reduced_universe({})
