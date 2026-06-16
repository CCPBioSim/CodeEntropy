"""Unit tests for the conformational-state static node."""

from __future__ import annotations

from types import SimpleNamespace

from CodeEntropy.levels.nodes import conformations
from CodeEntropy.levels.nodes.conformations import ComputeConformationalStatesNode


class FakeConformationStateBuilder:
    """Test double for ConformationStateBuilder."""

    def __init__(self, universe_operations):
        self.universe_operations = universe_operations
        self.calls = []

    def build_conformational_states(
        self,
        *,
        data_container,
        levels,
        groups,
        bin_width,
        frame_selection,
        progress=None,
    ):
        self.calls.append(
            {
                "data_container": data_container,
                "levels": levels,
                "groups": groups,
                "bin_width": bin_width,
                "frame_selection": frame_selection,
                "progress": progress,
            }
        )
        return (
            {"ua_key": ["state_a"]},
            [["res_state"]],
            {"ua_key": 1},
            [1],
        )


def test_compute_conformational_states_node_runs_and_writes_shared_data(monkeypatch):
    builder_holder = {}

    def builder_factory(universe_operations):
        builder = FakeConformationStateBuilder(universe_operations)
        builder_holder["builder"] = builder
        return builder

    monkeypatch.setattr(
        conformations,
        "ConformationStateBuilder",
        builder_factory,
    )

    universe_operations = object()
    node = ComputeConformationalStatesNode(universe_operations)

    universe = object()
    frame_selection = object()
    progress = object()

    shared_data = {
        "reduced_universe": universe,
        "levels": [["united_atom", "residue"]],
        "groups": {0: [0]},
        "frame_selection": frame_selection,
        "args": SimpleNamespace(bin_width=30),
    }

    result = node.run(shared_data, progress=progress)

    assert shared_data["conformational_states"] == {
        "ua": {"ua_key": ["state_a"]},
        "res": [["res_state"]],
    }
    assert shared_data["flexible_dihedrals"] == {
        "ua": {"ua_key": 1},
        "res": [1],
    }
    assert result == {
        "conformational_states": shared_data["conformational_states"],
    }

    builder = builder_holder["builder"]
    assert builder.universe_operations is universe_operations
    assert builder.calls == [
        {
            "data_container": universe,
            "levels": [["united_atom", "residue"]],
            "groups": {0: [0]},
            "bin_width": 30,
            "frame_selection": frame_selection,
            "progress": progress,
        }
    ]


def test_compute_conformational_states_node_converts_bin_width_to_int(monkeypatch):
    captured = {}

    class Builder:
        def __init__(self, universe_operations):
            pass

        def build_conformational_states(self, **kwargs):
            captured.update(kwargs)
            return {}, [], {}, []

    monkeypatch.setattr(conformations, "ConformationStateBuilder", Builder)

    node = ComputeConformationalStatesNode()
    shared_data = {
        "reduced_universe": object(),
        "levels": [],
        "groups": {},
        "frame_selection": object(),
        "args": SimpleNamespace(bin_width="45"),
    }

    node.run(shared_data)

    assert captured["bin_width"] == 45
