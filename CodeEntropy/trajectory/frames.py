"""Frame-selection primitives for trajectory-indexed execution.

Frame-index contract:
    - FrameSelection.indices are absolute MDAnalysis trajectory indices.
    - MDAnalysis trajectory access must use these absolute frame indices.
    - Arrays produced by analyses over FrameSelection are indexed locally with
      enumerate(FrameSelection.indices).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSelection:
    """Absolute trajectory frame selection.

    Attributes:
        indices: Absolute source-trajectory frame indices selected for analysis.
    """

    indices: tuple[int, ...]

    @classmethod
    def from_bounds(cls, start: int, stop: int, step: int) -> FrameSelection:
        """Build a frame selection from Python range semantics.

        Args:
            start: Inclusive source-trajectory start frame.
            stop: Exclusive source-trajectory stop frame.
            step: Frame stride.

        Returns:
            FrameSelection containing absolute source-trajectory frame indices.

        Raises:
            ValueError: If ``step`` is not positive.
        """
        if step <= 0:
            raise ValueError(f"Frame step must be positive, got {step}")

        return cls(indices=tuple(range(int(start), int(stop), int(step))))

    def __len__(self) -> int:
        """Return the number of selected frames."""
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        """Iterate over absolute source-trajectory frame indices."""
        return iter(self.indices)

    @property
    def n_frames(self) -> int:
        """Return the number of selected frames."""
        return len(self)

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Return absolute source-trajectory frame indices.

        This compatibility property is intentionally identical to ``indices``.
        """
        return self.indices

    @property
    def analysis_indices(self) -> tuple[int, ...]:
        """Return active analysis frame indices.

        Physical frame slicing has been removed, so analysis indices are absolute
        source-trajectory indices.
        """
        return self.indices

    @property
    def source_start(self) -> int | None:
        """Return the first selected source frame, or None if empty."""
        return self.indices[0] if self.indices else None

    @property
    def source_stop_exclusive(self) -> int | None:
        """Return one past the final selected source frame, or None if empty."""
        return self.indices[-1] + 1 if self.indices else None

    def iter_indices(self) -> Iterator[int]:
        """Yield absolute source-trajectory frame indices."""
        yield from self.indices

    def iter_source_indices(self) -> Iterator[int]:
        """Yield absolute source-trajectory frame indices."""
        yield from self.indices

    def iter_analysis_indices(self) -> Iterator[int]:
        """Yield active analysis frame indices.

        Since physical frame slicing has been removed, these are absolute source
        trajectory frame indices.
        """
        yield from self.indices

    def iter_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield ``(local_i, absolute_frame_index)`` pairs."""
        yield from enumerate(self.indices)

    def infer_step(self) -> int:
        """Infer the regular stride in selected frame indices.

        Returns:
            Integer step between selected frames. Returns 1 for zero or one frame.

        Raises:
            ValueError: If the frame selection is not regularly spaced.
        """
        if len(self.indices) <= 1:
            return 1

        step = self.indices[1] - self.indices[0]
        if step <= 0:
            raise ValueError("Frame indices must be strictly increasing.")

        for left, right in zip(self.indices, self.indices[1:], strict=False):
            if right - left != step:
                raise ValueError("Frame selection is not regularly spaced.")

        return step

    def infer_source_step(self) -> int:
        """Return the regular source-frame stride."""
        return self.infer_step()

    def infer_analysis_step(self) -> int:
        """Return the regular analysis-frame stride."""
        return self.infer_step()
