"""Frame chunking helpers for map-reduce execution."""

from __future__ import annotations


def chunk_frame_indices(
    frame_indices: list[int],
    chunk_size: int,
) -> list[tuple[int, ...]]:
    """Split frame indices into deterministic fixed-size chunks.

    Args:
        frame_indices: Ordered selected frame indices to split.
        chunk_size: Maximum number of frame indices per chunk.

    Returns:
        A list of frame-index tuples preserving input order.

    Raises:
        ValueError: If ``chunk_size`` is less than one.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    return [
        tuple(frame_indices[i : i + chunk_size])
        for i in range(0, len(frame_indices), chunk_size)
    ]
