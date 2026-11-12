from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from CodeEntropy.levels.dihedrals.angle_observations import (
    ConformationChunkTask,
    DihedralAngleCollector,
)
from CodeEntropy.levels.dihedrals.topology import MoleculeDihedralTopology
from CodeEntropy.trajectory.frames import FrameSelection


class _AngleCollector(DihedralAngleCollector):
    """Concrete angle collector for unit tests."""

    def __init__(self) -> None:
        """Initialize the test collector."""
        self._universe_operations = MagicMock()


def _make_frame_selection(*indices: int) -> FrameSelection:
    """Build a FrameSelection from explicit indices.

    Args:
        *indices: Absolute frame indices.

    Returns:
        FrameSelection containing the requested indices.
    """
    return FrameSelection(indices=tuple(indices))


def _make_topology() -> MoleculeDihedralTopology:
    """Build a small molecule topology used by angle-observation tests.

    Returns:
        MoleculeDihedralTopology with one UA dihedral and one residue dihedral.
    """
    return MoleculeDihedralTopology(
        group_id=0,
        molecule_id=7,
        molecule_order=0,
        num_residues=2,
        ua_dihedrals_by_residue={0: ["ua0"], 1: []},
        residue_dihedrals=["res0"],
    )


def test_frame_selection_from_chunk_preserves_absolute_indices():
    frame_selection = _AngleCollector._frame_selection_from_chunk((10, 20, 30))

    assert frame_selection.indices == (10, 20, 30)
    assert frame_selection.analysis_indices == (10, 20, 30)


def test_frame_selection_from_single_frame_chunk_preserves_absolute_index():
    frame_selection = _AngleCollector._frame_selection_from_chunk((42,))

    assert frame_selection.indices == (42,)


def test_frame_selection_from_chunk_rejects_invalid_chunks():
    with pytest.raises(ValueError, match="empty chunk"):
        _AngleCollector._frame_selection_from_chunk(())

    with pytest.raises(ValueError, match="strictly increasing"):
        _AngleCollector._frame_selection_from_chunk((2, 1))

    with pytest.raises(ValueError, match="regularly strided"):
        _AngleCollector._frame_selection_from_chunk((0, 2, 5))


def test_build_conformation_chunk_tasks_orders_by_molecule_then_chunk():
    collector = _AngleCollector()
    frame_selection = _make_frame_selection(10, 20, 30)
    topologies = [
        MoleculeDihedralTopology(0, "mol-a", 0, 1, {}, []),
        MoleculeDihedralTopology(0, "mol-b", 1, 1, {}, []),
    ]

    tasks = collector._build_conformation_chunk_tasks(
        topologies=topologies,
        frame_selection=frame_selection,
        chunk_size=2,
    )

    assert [
        (task.molecule_id, task.chunk_id, task.frame_indices) for task in tasks
    ] == [
        ("mol-a", 0, (10, 20)),
        ("mol-a", 1, (30,)),
        ("mol-b", 0, (10, 20)),
        ("mol-b", 1, (30,)),
    ]


def test_extract_positive_angle_array_wraps_negative_values():
    collector = _AngleCollector()
    dihedral_results = SimpleNamespace(
        results=SimpleNamespace(
            angles=np.array([[-10.0, 20.0], [30.0, -40.0]], dtype=float)
        )
    )

    angles = collector._extract_positive_angle_array(
        dihedral_results=dihedral_results,
        num_dihedrals=2,
        number_frames=2,
    )

    np.testing.assert_allclose(angles, np.array([[350.0, 20.0], [30.0, 320.0]]))


def test_collect_angle_observable_collects_ua_and_residue_arrays():
    collector = _AngleCollector()
    task = ConformationChunkTask(
        group_id=0,
        molecule_id=7,
        molecule_order=0,
        chunk_id=0,
        frame_indices=(0, 1),
        frame_selection=FrameSelection.from_bounds(0, 2, 1),
    )
    topology = _make_topology()

    ua_results = SimpleNamespace(
        results=SimpleNamespace(angles=np.array([[-10.0], [10.0]], dtype=float))
    )
    residue_results = SimpleNamespace(
        results=SimpleNamespace(angles=np.array([[30.0], [-40.0]], dtype=float))
    )
    collector._run_dihedrals = MagicMock(side_effect=[ua_results, residue_results])

    observable = collector._collect_angle_observable(
        topology=topology,
        task=task,
        level_list=["united_atom", "residue"],
    )

    np.testing.assert_allclose(observable.ua_angles_by_residue[0], [[350.0], [10.0]])
    assert observable.ua_angles_by_residue[1].shape == (2, 0)
    np.testing.assert_allclose(observable.residue_angles, [[30.0], [320.0]])
    assert collector._run_dihedrals.call_count == 2


def test_run_dihedrals_uses_frame_selection_bounds():
    collector = _AngleCollector()
    frame_selection = FrameSelection.from_bounds(10, 40, 10)
    fake_runner = MagicMock()
    fake_runner.run.return_value = "result"

    with patch("CodeEntropy.levels.dihedrals.angle_observations.Dihedral") as fake_cls:
        fake_cls.return_value = fake_runner
        out = collector._run_dihedrals(
            dihedrals=["D0"],
            frame_selection=frame_selection,
        )

    assert out == "result"
    fake_cls.assert_called_once_with(["D0"])
    fake_runner.run.assert_called_once_with(start=10, stop=31, step=10)


def test_run_dihedrals_raises_when_no_dihedrals():
    collector = _AngleCollector()

    with pytest.raises(ValueError, match="no dihedrals"):
        collector._run_dihedrals(
            dihedrals=[],
            frame_selection=FrameSelection.from_bounds(0, 1, 1),
        )


def test_analysis_run_bounds_raises_when_frame_selection_empty():
    with pytest.raises(ValueError, match="Frame selection is empty"):
        _AngleCollector._analysis_run_bounds(FrameSelection(indices=()))
