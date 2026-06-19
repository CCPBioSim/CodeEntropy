from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from CodeEntropy.levels.conformation_dag import ConformationDAG
from CodeEntropy.trajectory.frames import FrameSelection


def _make_frame_selection(
    start: int = 0,
    stop: int = 3,
    step: int = 1,
) -> FrameSelection:
    """Build a FrameSelection for ConformationDAG tests.

    Args:
        start: Inclusive source-frame start.
        stop: Exclusive source-frame stop.
        step: Source-frame stride.

    Returns:
        FrameSelection covering the requested bounds.
    """
    return FrameSelection.from_bounds(start=start, stop=stop, step=step)


def test_build_returns_self():
    """Test that build returns the DAG instance."""
    dag = ConformationDAG(universe_operations=MagicMock())

    assert dag.build() is dag


def test_execute_uses_execution_policy_chunk_size_and_stores_outputs():
    """Test that execute stores conformational and flexible-dihedral outputs."""
    dag = ConformationDAG(universe_operations=MagicMock())
    dag._policy = MagicMock()
    dag._policy.frame_chunk_size.return_value = 2
    dag._builder = MagicMock()
    dag._builder.build_conformational_states.return_value = (
        {(0, 0): ["0"]},
        [["1"]],
        {(0, 0): 1},
        [0],
    )

    frame_selection = _make_frame_selection(start=0, stop=3, step=1)
    progress = MagicMock()
    shared_data = {
        "reduced_universe": "universe",
        "levels": {7: ["united_atom"]},
        "groups": {0: [7]},
        "frame_selection": frame_selection,
        "args": SimpleNamespace(bin_width=30),
    }

    out = dag.execute(shared_data, progress=progress)

    dag._policy.frame_chunk_size.assert_called_once_with(
        shared_data,
        n_frames=frame_selection.n_frames,
    )
    dag._builder.build_conformational_states.assert_called_once_with(
        data_container="universe",
        levels={7: ["united_atom"]},
        groups={0: [7]},
        bin_width=30,
        frame_selection=frame_selection,
        progress=progress,
        chunk_size=2,
    )

    assert shared_data["conformational_states"] == {
        "ua": {(0, 0): ["0"]},
        "res": [["1"]],
    }
    assert shared_data["flexible_dihedrals"] == {
        "ua": {(0, 0): 1},
        "res": [0],
    }
    assert out == {"conformational_states": shared_data["conformational_states"]}


def test_execute_converts_bin_width_to_int():
    """Test that execute converts args.bin_width before calling the builder."""
    dag = ConformationDAG()
    dag._policy = MagicMock()
    dag._policy.frame_chunk_size.return_value = 3
    dag._builder = MagicMock()
    dag._builder.build_conformational_states.return_value = ({}, [], {}, [])

    frame_selection = _make_frame_selection(start=0, stop=3, step=1)
    shared_data = {
        "reduced_universe": object(),
        "levels": {},
        "groups": {},
        "frame_selection": frame_selection,
        "args": SimpleNamespace(bin_width="45"),
    }

    dag.execute(shared_data)

    assert dag._builder.build_conformational_states.call_args.kwargs["bin_width"] == 45
    assert dag._builder.build_conformational_states.call_args.kwargs["chunk_size"] == 3


def test_execute_passes_real_frame_selection_to_builder():
    """Test that execute forwards the existing FrameSelection object unchanged."""
    dag = ConformationDAG()
    dag._policy = MagicMock()
    dag._policy.frame_chunk_size.return_value = 1
    dag._builder = MagicMock()
    dag._builder.build_conformational_states.return_value = ({}, [], {}, [])

    frame_selection = _make_frame_selection(start=10, stop=31, step=10)
    shared_data = {
        "reduced_universe": "universe",
        "levels": {},
        "groups": {},
        "frame_selection": frame_selection,
        "args": SimpleNamespace(bin_width=30),
    }

    dag.execute(shared_data)

    assert (
        dag._builder.build_conformational_states.call_args.kwargs["frame_selection"]
        is frame_selection
    )
    dag._policy.frame_chunk_size.assert_called_once_with(
        shared_data,
        n_frames=frame_selection.n_frames,
    )
