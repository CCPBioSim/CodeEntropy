from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from CodeEntropy.levels.dihedrals.angle_observations import (
    ConformationChunkTask,
    DihedralAngleObservable,
)
from CodeEntropy.levels.dihedrals.state_assignment import (
    ConformationStateAssigner,
    ConformationStateData,
    ConformationStatePartial,
)
from CodeEntropy.levels.dihedrals.topology import MoleculeDihedralTopology
from CodeEntropy.trajectory.frames import FrameSelection


class _StateAssigner(ConformationStateAssigner):
    """Concrete state assigner for unit tests."""

    def __init__(self) -> None:
        """Initialize the test assigner."""
        self._universe_operations = MagicMock()


def _make_task(molecule_order: int, chunk_id: int) -> ConformationChunkTask:
    """Build a minimal conformation task for reducer tests.

    Args:
        molecule_order: Molecule ordering value.
        chunk_id: Chunk ordering value.

    Returns:
        ConformationChunkTask for a single frame.
    """
    return ConformationChunkTask(
        group_id=0,
        molecule_id=molecule_order,
        molecule_order=molecule_order,
        chunk_id=chunk_id,
        frame_indices=(chunk_id,),
        frame_selection=FrameSelection.from_bounds(chunk_id, chunk_id + 1, 1),
    )


def test_pad_peak_values_converts_ragged_peaks_to_arrays():
    padded, counts = _StateAssigner._pad_peak_values([[10.0, 20.0], [30.0]])

    np.testing.assert_allclose(padded, np.array([[10.0, 20.0], [30.0, 0.0]]))
    np.testing.assert_array_equal(counts, np.array([2, 1], dtype=np.int64))


def test_pad_peak_values_handles_empty_peak_list():
    padded, counts = _StateAssigner._pad_peak_values([])

    assert padded.shape == (0, 1)
    assert counts.shape == (0,)


def test_state_strings_from_labels_builds_legacy_state_strings():
    labels = np.array([[0, 1], [1, 0]], dtype=np.int64)

    states = _StateAssigner._state_strings_from_labels(labels)

    assert states == ["01", "10"]


def test_state_strings_from_labels_filters_empty_states():
    labels = np.empty((2, 0), dtype=np.int64)

    states = _StateAssigner._state_strings_from_labels(labels)

    assert states == []


def test_process_conformations_from_angles_assigns_states_and_flexible_count():
    assigner = _StateAssigner()
    angles = np.array([[5.0], [15.0]], dtype=np.float64)

    states, flexible = assigner._process_conformations_from_angles(
        peaks=[[5.0, 15.0]],
        angles=angles,
    )

    assert states == ["0", "1"]
    assert flexible == 1


def test_process_conformations_from_angles_handles_no_dihedrals():
    assigner = _StateAssigner()
    angles = np.empty((2, 0), dtype=np.float64)

    states, flexible = assigner._process_conformations_from_angles(
        peaks=[],
        angles=angles,
    )

    assert states == []
    assert flexible == 0


def test_assign_state_partial_from_observable_handles_ua_and_residue_levels():
    assigner = _StateAssigner()
    task = _make_task(molecule_order=0, chunk_id=0)
    topology = MoleculeDihedralTopology(
        group_id=2,
        molecule_id=7,
        molecule_order=0,
        num_residues=2,
        ua_dihedrals_by_residue={0: ["ua0"], 1: []},
        residue_dihedrals=["res0"],
    )
    observable = DihedralAngleObservable(
        task=task,
        num_residues=2,
        ua_angles_by_residue={
            0: np.array([[5.0], [15.0]], dtype=np.float64),
            1: np.empty((2, 0), dtype=np.float64),
        },
        residue_angles=np.array([[30.0], [40.0]], dtype=np.float64),
    )

    partial = assigner._assign_state_partial_from_observable(
        observable=observable,
        topology=topology,
        level_list=["united_atom", "residue"],
        peaks_ua=[[[5.0, 15.0]], []],
        peaks_res=[[30.0, 40.0]],
    )

    assert partial.states_ua_updates[(2, 0)] == ["0", "1"]
    assert partial.flexible_ua_updates[(2, 0)] == 1
    assert partial.states_ua_updates[(2, 1)] == []
    assert partial.flexible_ua_updates[(2, 1)] == 0
    assert partial.state_res == ["0", "1"]
    assert partial.flex_res == 1


def test_reduce_state_partials_preserves_molecule_then_chunk_order_and_max_flex():
    assigner = _StateAssigner()
    partials = [
        ConformationStatePartial(
            task=_make_task(molecule_order=1, chunk_id=0),
            state_res=["m1c0"],
            flex_res=0,
            states_ua_updates={(0, 0): ["m1c0"]},
            flexible_ua_updates={(0, 0): 0},
        ),
        ConformationStatePartial(
            task=_make_task(molecule_order=0, chunk_id=1),
            state_res=["m0c1"],
            flex_res=2,
            states_ua_updates={(0, 0): ["m0c1"]},
            flexible_ua_updates={(0, 0): 2},
        ),
        ConformationStatePartial(
            task=_make_task(molecule_order=0, chunk_id=0),
            state_res=["m0c0"],
            flex_res=1,
            states_ua_updates={(0, 0): ["m0c0"]},
            flexible_ua_updates={(0, 0): 1},
        ),
    ]

    state_data = assigner._reduce_state_partials(partials)

    assert state_data.state_res == ["m0c0", "m0c1", "m1c0"]
    assert state_data.flex_res == 2
    assert state_data.states_ua_updates[(0, 0)] == ["m0c0", "m0c1", "m1c0"]
    assert state_data.flexible_ua_updates[(0, 0)] == 2


def test_merge_group_state_data_extends_existing_ua_states():
    states_ua = {(0, 0): ["0"]}
    states_res = []
    flexible_ua = {(0, 0): 1}
    flexible_res = []
    state_data = ConformationStateData(
        state_res=["1"],
        flex_res=0,
        states_ua_updates={(0, 0): ["1"], (0, 1): ["2"]},
        flexible_ua_updates={(0, 0): 2, (0, 1): 0},
    )

    _StateAssigner._merge_group_state_data(
        state_data=state_data,
        states_ua=states_ua,
        states_res=states_res,
        flexible_ua=flexible_ua,
        flexible_res=flexible_res,
    )

    assert states_ua[(0, 0)] == ["0", "1"]
    assert states_ua[(0, 1)] == ["2"]
    assert flexible_ua[(0, 0)] == 2
    assert flexible_ua[(0, 1)] == 0
    assert states_res == [["1"]]
    assert flexible_res == [0]
