"""MDAnalysis frame access boundary.

FrameSource is the central place where workflow-owned frame seeking happens.
"""

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
        selection: FrameSelection describing source and analysis frame indices.

    Notes:
        During the migration, ``selection.analysis_indices`` may be local indices
        into a physically frame-sliced universe. After physical frame slicing is
        removed, they should be absolute source trajectory indices.
    """

    universe: Any
    selection: FrameSelection

    def __len__(self) -> int:
        """Return the number of selected frames."""
        return len(self.selection)

    def iter_indices(self) -> Iterator[int]:
        """Yield frame indices valid for the active analysis universe."""
        yield from self.selection.iter_analysis_indices()

    def iter_source_indices(self) -> Iterator[int]:
        """Yield absolute source-trajectory frame indices."""
        yield from self.selection.iter_source_indices()

    def iter_pairs(self) -> Iterator[tuple[int, int, int]]:
        """Yield ``(local_i, source_index, analysis_index)`` triples."""
        yield from self.selection.iter_pairs()

    def seek(self, frame_index: int) -> Any:
        """Move the active analysis universe to a frame.

        Args:
            frame_index: Frame index in the active analysis-universe index space.
                This is local while physical frame slicing remains enabled, and
                absolute once physical frame slicing is removed.

        Returns:
            The MDAnalysis timestep for the selected frame.
        """
        return self.universe.trajectory[int(frame_index)]

    def seek_pair(self, source_index: int, analysis_index: int) -> Any:
        """Seek using an explicit source/analysis pair.

        Args:
            source_index: Absolute source-trajectory frame index. Currently used
                for logging/debugging and future Dask mapping.
            analysis_index: Frame index valid for the active analysis universe.

        Returns:
            The MDAnalysis timestep for ``analysis_index``.
        """
        _ = int(source_index)
        return self.seek(int(analysis_index))
