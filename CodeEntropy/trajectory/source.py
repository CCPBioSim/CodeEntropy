"""MDAnalysis frame access boundary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from CodeEntropy.trajectory.frames import FrameSelection


@dataclass
class FrameSource:
    """Single owner of selected MDAnalysis trajectory frame access.

    Attributes:
        universe: Active MDAnalysis Universe used for analysis.
        selection: Absolute trajectory frame selection.
    """

    universe: Any
    selection: FrameSelection

    def __len__(self) -> int:
        """Return the number of selected frames."""
        return len(self.selection)

    def iter_indices(self) -> Iterator[int]:
        """Yield absolute selected trajectory frame indices."""
        yield from self.selection.iter_indices()

    def iter_source_indices(self) -> Iterator[int]:
        """Yield absolute selected trajectory frame indices."""
        yield from self.selection.iter_source_indices()

    def iter_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield ``(local_i, absolute_frame_index)`` pairs."""
        yield from self.selection.iter_pairs()

    def seek(self, frame_index: int) -> Any:
        """Move the universe to an absolute trajectory frame.

        Args:
            frame_index: Absolute source-trajectory frame index.

        Returns:
            The MDAnalysis timestep for the selected frame.
        """
        return self.universe.trajectory[int(frame_index)]
