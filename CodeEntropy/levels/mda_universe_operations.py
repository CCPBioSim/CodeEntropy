"""MDAnalysis universe utilities.

This module provides helper functions for creating and manipulating MDAnalysis
`Universe` objects used throughout the CodeEntropy workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectorySlice:
    """Frame slicing configuration for trajectory selection.

    Attributes:
        start: Starting frame index (inclusive). If None, defaults to 0.
        end: Ending frame index (exclusive). If None, defaults to len(trajectory).
        step: Step between frames. Must be >= 1.
    """

    start: Optional[int] = None
    end: Optional[int] = None
    step: int = 1


class UniverseOperations:
    """Utility methods for creating and manipulating MDAnalysis universes.

    This class focuses on a small set of responsibilities:
      - Build reduced universes by selecting frames or atoms.
      - Extract a single molecule (fragment) as its own universe.
      - Merge coordinates and forces from separate trajectories into one universe.

    Notes:
        These methods return new `MDAnalysis.Universe` objects backed by in-memory
        trajectories. This makes downstream operations deterministic and avoids
        side effects on the original universe.
    """

    def new_U_select_frame(
        self,
        u: mda.Universe,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ) -> mda.Universe:
        """Create a reduced universe by slicing frames.

        Args:
            u: Source universe containing topology, coordinates, forces, and dimensions.
            start: Starting frame index (inclusive). If None, defaults to 0.
            end: Ending frame index (exclusive). If None, defaults to len(u.trajectory).
            step: Step between frames. Must be >= 1.

        Returns:
            A new universe containing the same atoms as `u` but only the selected frames

        Raises:
            ValueError: If `step` is less than 1.
        """
        if step < 1:
            raise ValueError("step must be >= 1")

        if start is None:
            start = 0
        if end is None:
            end = len(u.trajectory)

        select_atom = u.select_atoms("all", updating=True)

        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )
        dimensions = (
            AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom)
            .run()
            .results["timeseries"][start:end:step]
        )

        u2 = mda.Merge(select_atom)
        u2.load_new(
            coordinates,
            format=MemoryReader,
            forces=forces,
            dimensions=dimensions,
        )

        logger.debug("Created reduced universe by frames: %s", u2)
        return u2

    def new_U_select_atom(
        self, u: mda.Universe, select_string: str = "all"
    ) -> mda.Universe:
        """Create a reduced universe by selecting a subset of atoms.

        Args:
            u: Source universe containing topology, coordinates, forces, and dimensions.
            select_string: MDAnalysis selection string.

        Returns:
            A new universe containing only the selected atoms across all frames.
        """
        select_atom = u.select_atoms(select_string, updating=True)

        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        dimensions = (
            AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )

        u2 = mda.Merge(select_atom)
        u2.load_new(
            coordinates,
            format=MemoryReader,
            forces=forces,
            dimensions=dimensions,
        )

        logger.debug("Created reduced universe by atoms: %s", u2)
        return u2

    def get_molecule_container(
        self, universe: mda.Universe, molecule_id: int
    ) -> mda.Universe:
        """Extract a single molecule (fragment) from a universe.

        Args:
            universe: Universe containing the system.
            molecule_id: Index of the fragment (molecule) to extract.

        Returns:
            A new universe containing only the atoms from the specified fragment.

        Raises:
            IndexError: If `molecule_id` is out of range.
            ValueError: If the fragment has no atoms.
        """
        fragments = universe.atoms.fragments
        frag = fragments[molecule_id]
        if len(frag) == 0:
            raise ValueError(f"Fragment {molecule_id} is empty.")

        selection_string = f"index {frag.indices[0]}:{frag.indices[-1]}"
        return self.new_U_select_atom(universe, selection_string)

    def merge_forces(
        self,
        tprfile: str,
        trrfile: str,
        forcefile: str,
        fileformat: Optional[str] = None,
        kcal: bool = False,
    ) -> mda.Universe:
        """Merge coordinates and forces trajectories into a single universe.

        Args:
            tprfile: Topology input file.
            trrfile: Coordinate trajectory file.
            forcefile: Force trajectory file.
            fileformat: Optional MDAnalysis format string (e.g., "TRR").
            kcal: If True, convert forces from kcal to kJ by multiplying by 4.184.

        Returns:
            A universe where coordinates come from `trrfile` and forces come from
            `forcefile`.

        Raises:
            ValueError: If the coordinate and force trajectories are incompatible.
        """
        logger.debug("Loading coordinates universe: %s", trrfile)
        u = mda.Universe(tprfile, trrfile, format=fileformat)

        logger.debug("Loading forces universe: %s", forcefile)
        u_force = mda.Universe(tprfile, forcefile, format=fileformat)

        select_atom = u.select_atoms("all")
        select_atom_force = u_force.select_atoms("all")

        coordinates = (
            AnalysisFromFunction(lambda ag: ag.positions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )
        forces = (
            AnalysisFromFunction(lambda ag: ag.forces.copy(), select_atom_force)
            .run()
            .results["timeseries"]
        )
        dimensions = (
            AnalysisFromFunction(lambda ag: ag.dimensions.copy(), select_atom)
            .run()
            .results["timeseries"]
        )

        if kcal:
            forces = forces * 4.184

        logger.debug("Merging forces with coordinates universe.")
        new_universe = mda.Merge(select_atom)
        new_universe.load_new(coordinates, forces=forces, dimensions=dimensions)

        return new_universe
