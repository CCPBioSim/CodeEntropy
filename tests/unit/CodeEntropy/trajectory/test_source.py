from unittest.mock import MagicMock

from CodeEntropy.trajectory.frames import FrameSelection
from CodeEntropy.trajectory.source import FrameSource


def test_len_returns_selection_length():
    selection = FrameSelection(indices=(0, 2, 4))
    source = FrameSource(universe=MagicMock(), selection=selection)

    assert len(source) == 3


def test_iter_indices_delegates_to_selection_indices():
    selection = FrameSelection(indices=(1, 3, 5))
    source = FrameSource(universe=MagicMock(), selection=selection)

    assert list(source.iter_indices()) == [1, 3, 5]


def test_iter_source_indices_delegates_to_selection_source_indices():
    selection = FrameSelection(indices=(1, 3, 5))
    source = FrameSource(universe=MagicMock(), selection=selection)

    assert list(source.iter_source_indices()) == [1, 3, 5]


def test_iter_pairs_delegates_to_selection_pairs():
    selection = FrameSelection(indices=(10, 20))
    source = FrameSource(universe=MagicMock(), selection=selection)

    assert list(source.iter_pairs()) == [(0, 10), (1, 20)]


def test_seek_uses_absolute_trajectory_index():
    timestep = object()

    universe = MagicMock()
    universe.trajectory = MagicMock()
    universe.trajectory.__getitem__.return_value = timestep

    selection = FrameSelection(indices=(2, 4, 6))
    source = FrameSource(universe=universe, selection=selection)

    out = source.seek(4)

    assert out is timestep
    universe.trajectory.__getitem__.assert_called_once_with(4)


def test_seek_casts_frame_index_to_int():
    timestep = object()

    universe = MagicMock()
    universe.trajectory = MagicMock()
    universe.trajectory.__getitem__.return_value = timestep

    selection = FrameSelection(indices=(0, 1, 2))
    source = FrameSource(universe=universe, selection=selection)

    out = source.seek("2")

    assert out is timestep
    universe.trajectory.__getitem__.assert_called_once_with(2)


def test_seek_allows_underlying_trajectory_to_raise_index_error():
    universe = MagicMock()
    universe.trajectory = MagicMock()
    universe.trajectory.__getitem__.side_effect = IndexError("bad frame")

    selection = FrameSelection(indices=(0, 1))
    source = FrameSource(universe=universe, selection=selection)

    try:
        source.seek(99)
    except IndexError as exc:
        assert str(exc) == "bad frame"
    else:
        raise AssertionError("Expected IndexError")
