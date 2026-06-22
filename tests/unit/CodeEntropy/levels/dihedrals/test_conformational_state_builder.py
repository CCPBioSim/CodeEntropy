from __future__ import annotations

from unittest.mock import MagicMock

from CodeEntropy.levels.dihedrals.angle_observations import (
    ConformationChunkTask,
    DihedralAngleObservable,
)
from CodeEntropy.levels.dihedrals.conformational_state_builder import (
    ConformationStateBuilder,
)
from CodeEntropy.levels.dihedrals.peak_detection import DihedralPeakData
from CodeEntropy.levels.dihedrals.state_assignment import (
    ConformationStateData,
    ConformationStatePartial,
)
from CodeEntropy.levels.dihedrals.topology import MoleculeDihedralTopology
from CodeEntropy.trajectory.frames import FrameSelection


def _make_frame_selection(
    start: int = 0,
    stop: int = 2,
    step: int = 1,
) -> FrameSelection:
    """Build a FrameSelection for builder tests.

    Args:
        start: Inclusive source-frame start.
        stop: Exclusive source-frame stop.
        step: Source-frame stride.

    Returns:
        FrameSelection covering the requested bounds.
    """
    return FrameSelection.from_bounds(start=start, stop=stop, step=step)


def test_build_conformational_states_defaults_chunk_size_to_selected_frame_count():
    builder = ConformationStateBuilder(universe_operations=MagicMock())
    builder._build_conformational_states_serial_chunked = MagicMock(
        return_value=("states_ua", "states_res", "flex_ua", "flex_res")
    )
    frame_selection = _make_frame_selection(start=0, stop=3, step=1)

    out = builder.build_conformational_states(
        data_container="universe",
        levels={7: ["residue"]},
        groups={0: [7]},
        bin_width=30.0,
        frame_selection=frame_selection,
    )

    assert out == ("states_ua", "states_res", "flex_ua", "flex_res")
    builder._build_conformational_states_serial_chunked.assert_called_once_with(
        data_container="universe",
        levels={7: ["residue"]},
        groups={0: [7]},
        bin_width=30.0,
        frame_selection=frame_selection,
        chunk_size=3,
        progress=None,
    )


def test_build_conformational_states_passes_explicit_chunk_size():
    builder = ConformationStateBuilder(universe_operations=MagicMock())
    builder._build_conformational_states_serial_chunked = MagicMock(
        return_value=({}, [], {}, [])
    )
    frame_selection = _make_frame_selection(start=0, stop=3, step=1)

    builder.build_conformational_states(
        data_container="universe",
        levels={7: ["residue"]},
        groups={0: [7]},
        bin_width=30.0,
        frame_selection=frame_selection,
        chunk_size=2,
    )

    assert (
        builder._build_conformational_states_serial_chunked.call_args.kwargs[
            "chunk_size"
        ]
        == 2
    )


def test_chunked_serial_rejects_invalid_chunk_size():
    builder = ConformationStateBuilder(universe_operations=MagicMock())

    try:
        builder._build_conformational_states_serial_chunked(
            data_container="universe",
            levels={},
            groups={},
            bin_width=30.0,
            frame_selection=_make_frame_selection(start=0, stop=1, step=1),
            chunk_size=0,
        )
    except ValueError as exc:
        assert "chunk_size must be >= 1" in str(exc)
    else:
        raise AssertionError("Expected invalid chunk size to raise ValueError")


def test_build_conformational_states_with_progress_handles_no_groups():
    builder = ConformationStateBuilder(universe_operations=MagicMock())
    progress = MagicMock()
    progress.add_task.return_value = 123

    states_ua, states_res, flex_ua, flex_res = builder.build_conformational_states(
        data_container=MagicMock(),
        levels={},
        groups={},
        bin_width=30.0,
        frame_selection=_make_frame_selection(start=0, stop=1, step=1),
        progress=progress,
    )

    assert states_ua == {}
    assert states_res == []
    assert flex_ua == {}
    assert flex_res == []
    progress.add_task.assert_called_once()
    progress.update.assert_called_once_with(123, title="No groups")
    progress.advance.assert_called_once_with(123)


def test_build_conformational_states_with_progress_skips_empty_molecule_group():
    builder = ConformationStateBuilder(universe_operations=MagicMock())
    progress = MagicMock()
    progress.add_task.return_value = 5

    states_ua, states_res, flex_ua, flex_res = builder.build_conformational_states(
        data_container=MagicMock(),
        levels={},
        groups={0: []},
        bin_width=30.0,
        frame_selection=_make_frame_selection(start=0, stop=1, step=1),
        progress=progress,
    )

    assert states_ua == {}
    assert states_res == [[]]
    assert flex_ua == {}
    assert flex_res == []
    progress.update.assert_called_with(5, title="Group 0 (empty)")
    progress.advance.assert_called_with(5)


def test_chunked_serial_group_flow_calls_domain_phases_in_order():
    builder = ConformationStateBuilder(universe_operations=MagicMock())
    frame_selection = _make_frame_selection(start=0, stop=2, step=1)
    topology = MoleculeDihedralTopology(0, 7, 0, 1, {0: ["ua"]}, ["res"])
    task = ConformationChunkTask(0, 7, 0, 0, (0, 1), frame_selection)
    observable = DihedralAngleObservable(task, 1, {}, None)
    peak_data = DihedralPeakData(peaks_ua=[[[10.0]]], peaks_res=[])
    partial = ConformationStatePartial(task, [], 0, {}, {})
    state_data = ConformationStateData([], 0, {(0, 0): ["0"]}, {(0, 0): 0})

    builder._discover_group_dihedral_topology = MagicMock(return_value=[topology])
    builder._build_conformation_chunk_tasks = MagicMock(return_value=[task])
    builder._collect_angle_observable = MagicMock(return_value=observable)
    builder._reduce_angle_observables_to_peak_data = MagicMock(return_value=peak_data)
    builder._assign_state_partial_from_observable = MagicMock(return_value=partial)
    builder._reduce_state_partials = MagicMock(return_value=state_data)

    states_ua, states_res, flexible_ua, flexible_res = (
        builder._build_conformational_states_serial_chunked(
            data_container="universe",
            levels={7: ["united_atom"]},
            groups={0: [7]},
            bin_width=30.0,
            frame_selection=frame_selection,
            chunk_size=2,
        )
    )

    assert states_ua == {(0, 0): ["0"]}
    assert states_res == [[], []]
    assert flexible_ua == {(0, 0): 0}
    assert flexible_res == [0]
    builder._discover_group_dihedral_topology.assert_called_once_with(
        data_container="universe",
        group_id=0,
        molecules=[7],
        level_list=["united_atom"],
    )
    builder._build_conformation_chunk_tasks.assert_called_once_with(
        topologies=[topology],
        frame_selection=frame_selection,
        chunk_size=2,
    )
    builder._collect_angle_observable.assert_called_once_with(
        topology=topology,
        task=task,
        level_list=["united_atom"],
    )
    builder._reduce_angle_observables_to_peak_data.assert_called_once_with(
        observables=[observable],
        level_list=["united_atom"],
        bin_width=30.0,
    )
    builder._assign_state_partial_from_observable.assert_called_once_with(
        observable=observable,
        topology=topology,
        level_list=["united_atom"],
        peaks_ua=peak_data.peaks_ua,
        peaks_res=peak_data.peaks_res,
    )
    builder._reduce_state_partials.assert_called_once_with([partial])


def test_chunked_serial_with_progress_updates_and_advances_non_empty_group():
    builder = ConformationStateBuilder(universe_operations=MagicMock())
    progress = MagicMock()
    progress.add_task.return_value = 44

    frame_selection = _make_frame_selection(start=0, stop=2, step=1)
    topology = MoleculeDihedralTopology(0, 7, 0, 1, {0: ["ua"]}, ["res"])
    task = ConformationChunkTask(0, 7, 0, 0, (0, 1), frame_selection)
    observable = DihedralAngleObservable(task, 1, {}, None)
    peak_data = DihedralPeakData(peaks_ua=[[[10.0]]], peaks_res=[])
    partial = ConformationStatePartial(task, [], 0, {}, {})
    state_data = ConformationStateData([], 0, {(0, 0): ["0"]}, {(0, 0): 0})

    builder._discover_group_dihedral_topology = MagicMock(return_value=[topology])
    builder._build_conformation_chunk_tasks = MagicMock(return_value=[task])
    builder._collect_angle_observable = MagicMock(return_value=observable)
    builder._reduce_angle_observables_to_peak_data = MagicMock(return_value=peak_data)
    builder._assign_state_partial_from_observable = MagicMock(return_value=partial)
    builder._reduce_state_partials = MagicMock(return_value=state_data)

    builder._build_conformational_states_serial_chunked(
        data_container="universe",
        levels={7: ["united_atom"]},
        groups={0: [7]},
        bin_width=30.0,
        frame_selection=frame_selection,
        chunk_size=2,
        progress=progress,
    )

    progress.update.assert_any_call(44, title="Group 0")
    progress.advance.assert_called_with(44)
