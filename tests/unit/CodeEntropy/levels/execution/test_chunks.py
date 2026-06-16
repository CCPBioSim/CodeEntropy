"""Unit tests for frame chunking helpers."""

from __future__ import annotations

import pytest

from CodeEntropy.levels.execution.chunks import chunk_frame_indices


def test_chunk_frame_indices_splits_into_fixed_size_chunks():
    assert chunk_frame_indices([0, 1, 2, 3, 4], chunk_size=2) == [
        (0, 1),
        (2, 3),
        (4,),
    ]


def test_chunk_frame_indices_returns_empty_list_for_no_frames():
    assert chunk_frame_indices([], chunk_size=3) == []


def test_chunk_frame_indices_returns_single_chunk_when_chunk_size_exceeds_frames():
    assert chunk_frame_indices([1, 2], chunk_size=10) == [(1, 2)]


def test_chunk_frame_indices_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        chunk_frame_indices([0], chunk_size=0)
