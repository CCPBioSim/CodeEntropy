import pytest

from CodeEntropy.trajectory.frames import FrameSelection


def test_from_bounds_uses_python_range_semantics():
    selection = FrameSelection.from_bounds(start=2, stop=10, step=3)

    assert selection.indices == (2, 5, 8)


def test_from_bounds_casts_bounds_to_ints():
    selection = FrameSelection.from_bounds(start=0.0, stop=5.0, step=2.0)

    assert selection.indices == (0, 2, 4)


def test_from_bounds_rejects_zero_step():
    with pytest.raises(ValueError, match="Frame step must be positive"):
        FrameSelection.from_bounds(start=0, stop=10, step=0)


def test_from_bounds_rejects_negative_step():
    with pytest.raises(ValueError, match="Frame step must be positive"):
        FrameSelection.from_bounds(start=10, stop=0, step=-1)


def test_len_returns_number_of_selected_frames():
    selection = FrameSelection(indices=(0, 2, 4))

    assert len(selection) == 3


def test_iter_yields_absolute_frame_indices():
    selection = FrameSelection(indices=(3, 6, 9))

    assert list(selection) == [3, 6, 9]


def test_n_frames_matches_len():
    selection = FrameSelection(indices=(1, 4, 7))

    assert selection.n_frames == 3


def test_source_indices_aliases_indices():
    selection = FrameSelection(indices=(1, 2, 3))

    assert selection.source_indices is selection.indices


def test_analysis_indices_aliases_indices():
    selection = FrameSelection(indices=(1, 2, 3))

    assert selection.analysis_indices is selection.indices


def test_source_start_returns_first_frame():
    selection = FrameSelection(indices=(5, 10, 15))

    assert selection.source_start == 5


def test_source_start_returns_none_for_empty_selection():
    selection = FrameSelection(indices=())

    assert selection.source_start is None


def test_source_stop_exclusive_returns_one_past_last_frame():
    selection = FrameSelection(indices=(5, 10, 15))

    assert selection.source_stop_exclusive == 16


def test_source_stop_exclusive_returns_none_for_empty_selection():
    selection = FrameSelection(indices=())

    assert selection.source_stop_exclusive is None


def test_iter_indices_yields_indices():
    selection = FrameSelection(indices=(2, 4, 6))

    assert list(selection.iter_indices()) == [2, 4, 6]


def test_iter_source_indices_yields_indices():
    selection = FrameSelection(indices=(2, 4, 6))

    assert list(selection.iter_source_indices()) == [2, 4, 6]


def test_iter_analysis_indices_yields_indices():
    selection = FrameSelection(indices=(2, 4, 6))

    assert list(selection.iter_analysis_indices()) == [2, 4, 6]


def test_iter_pairs_yields_local_and_absolute_indices():
    selection = FrameSelection(indices=(10, 20, 30))

    assert list(selection.iter_pairs()) == [(0, 10), (1, 20), (2, 30)]


def test_infer_step_returns_one_for_empty_selection():
    selection = FrameSelection(indices=())

    assert selection.infer_step() == 1


def test_infer_step_returns_one_for_single_frame_selection():
    selection = FrameSelection(indices=(7,))

    assert selection.infer_step() == 1


def test_infer_step_returns_regular_stride():
    selection = FrameSelection(indices=(2, 5, 8, 11))

    assert selection.infer_step() == 3


def test_infer_step_rejects_non_increasing_indices():
    selection = FrameSelection(indices=(4, 4, 5))

    with pytest.raises(ValueError, match="strictly increasing"):
        selection.infer_step()


def test_infer_step_rejects_irregular_indices():
    selection = FrameSelection(indices=(0, 2, 5))

    with pytest.raises(ValueError, match="not regularly spaced"):
        selection.infer_step()


def test_infer_source_step_delegates_to_infer_step():
    selection = FrameSelection(indices=(1, 4, 7))

    assert selection.infer_source_step() == 3


def test_infer_analysis_step_delegates_to_infer_step():
    selection = FrameSelection(indices=(1, 4, 7))

    assert selection.infer_analysis_step() == 3
