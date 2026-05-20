"""Frame-selection primitives for trajectory-indexed execution.

This module defines the frame-index contract used by the workflow.

During the migration, a workflow may still use a physically frame-sliced
``reduced_universe``. In that case:

    source_indices:
        Absolute indices into the original/source trajectory.

    analysis_indices:
        Local indices into the physically frame-sliced analysis universe.

Once physical frame slicing is removed, ``analysis_indices`` and
``source_indices`` should become identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

IndexSpace = Literal["source", "local"]


@dataclass(frozen=True)
class FrameSelection:
    """Selected trajectory frames and their active analysis index space.

    Attributes:
        source_indices: Absolute source-trajectory frame indices selected for
            analysis.
        analysis_indices: Frame indices used to seek the active analysis universe.
            These are local indices while the workflow still physically
            frame-slices the universe. They become absolute source indices after
            physical frame slicing is removed.
        analysis_index_space: Indicates whether ``analysis_indices`` are
            ``"local"`` to a frame-sliced universe or ``"source"`` absolute
            trajectory indices.
    """

    source_indices: tuple[int, ...]
    analysis_indices: tuple[int, ...]
    analysis_index_space: IndexSpace

    @classmethod
    def from_bounds(
        cls,
        start: int,
        stop: int,
        step: int,
        *,
        physical_frame_slicing: bool,
    ) -> FrameSelection:
        """Build a frame selection from Python range semantics.

        Args:
            start: Inclusive source-trajectory start frame.
            stop: Exclusive source-trajectory stop frame.
            step: Frame stride.
            physical_frame_slicing: If True, the active analysis universe is
                assumed to be physically sliced, so analysis indices are local
                ``0..n_frames-1``. If False, analysis indices are the same as
                source indices.

        Returns:
            FrameSelection describing both source and active-analysis index spaces.

        Raises:
            ValueError: If ``step`` is not positive.
        """
        if step <= 0:
            raise ValueError(f"Frame step must be positive, got {step}")

        source_indices = tuple(range(int(start), int(stop), int(step)))

        if physical_frame_slicing:
            analysis_indices = tuple(range(len(source_indices)))
            analysis_index_space: IndexSpace = "local"
        else:
            analysis_indices = source_indices
            analysis_index_space = "source"

        return cls(
            source_indices=source_indices,
            analysis_indices=analysis_indices,
            analysis_index_space=analysis_index_space,
        )

    def __len__(self) -> int:
        """Return the number of selected frames."""
        return len(self.source_indices)

    def __iter__(self) -> Iterator[int]:
        """Iterate over active analysis-universe frame indices."""
        return iter(self.analysis_indices)

    @property
    def n_frames(self) -> int:
        """Return the number of selected frames."""
        return len(self)

    @property
    def source_start(self) -> int | None:
        """Return the first selected source frame, or None if empty."""
        return self.source_indices[0] if self.source_indices else None

    @property
    def source_stop_exclusive(self) -> int | None:
        """Return one past the final selected source frame, or None if empty."""
        return self.source_indices[-1] + 1 if self.source_indices else None

    def iter_analysis_indices(self) -> Iterator[int]:
        """Yield frame indices valid for the active analysis universe."""
        yield from self.analysis_indices

    def iter_source_indices(self) -> Iterator[int]:
        """Yield absolute source-trajectory frame indices."""
        yield from self.source_indices

    def iter_pairs(self) -> Iterator[tuple[int, int, int]]:
        """Yield ``(local_i, source_index, analysis_index)`` triples."""
        for local_i, (source_index, analysis_index) in enumerate(
            zip(self.source_indices, self.analysis_indices, strict=True)
        ):
            yield local_i, source_index, analysis_index

    def infer_source_step(self) -> int:
        """Infer the regular stride in source-frame index space.

        Returns:
            Integer step between selected source frames. Returns 1 if zero or one
            frames are selected.

        Raises:
            ValueError: If the source frame selection is not regularly spaced.
        """
        return self._infer_regular_step(self.source_indices, "source")

    def infer_analysis_step(self) -> int:
        """Infer the regular stride in active analysis-frame index space.

        Returns:
            Integer step between active analysis frames. Returns 1 if zero or one
            frames are selected.

        Raises:
            ValueError: If the analysis frame selection is not regularly spaced.
        """
        return self._infer_regular_step(self.analysis_indices, "analysis")

    @staticmethod
    def _infer_regular_step(indices: tuple[int, ...], label: str) -> int:
        """Infer a regular positive stride from indices."""
        if len(indices) <= 1:
            return 1

        step = indices[1] - indices[0]
        if step <= 0:
            raise ValueError(f"{label} frame indices must be strictly increasing.")

        for left, right in zip(indices, indices[1:], strict=False):
            if right - left != step:
                raise ValueError(f"{label} frame selection is not regularly spaced.")

        return step
