"""
MDAnalysis universe utilities.

This module contains helpers for creating reduced MDAnalysis `Universe` objects by
sub-selecting frames and/or atoms, and for building a `Universe` that combines
coordinates from one trajectory with forces sourced from a second trajectory.
"""

from __future__ import annotations

import logging
from typing import Optional

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.exceptions import NoDataError

logger = logging.getLogger(__name__)


class UniverseOperations:
    """Functions to create and manipulate MDAnalysis Universe objects."""

    def __init__(self) -> None:
        """Initialise the operations helper."""
        self._universe = None

    def select_frames(
        self,
        u: mda.Universe,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ) -> mda.Universe:
        """Create a reduced universe by dropping frames according to user selection.

        Parameters
        ----------
        u:
            A Universe object with topology, coordinates and (optionally) forces.
        start:
            Frame index to start analysis. If None, defaults to 0.
        end:
            Frame index to stop analysis (Python slicing semantics). If None, defaults
            to the full trajectory length.
        step:
            Step size between frames.

        Returns
        -------
        mda.Universe:
            A reduced universe containing the selected frames, with coordinates,
            forces (if present) and unit cell dimensions loaded into memory.
        """
        if start is None:
            start = 0
        if end is None:
            end = len(u.trajectory)

        select_atom = u.select_atoms("all", updating=True)

        coordinates = self._extract_timeseries(select_atom, kind="positions")[
            start:end:step
        ]
        forces = self._extract_timeseries(select_atom, kind="forces")[start:end:step]
        dimensions = self._extract_timeseries(select_atom, kind="dimensions")[
            start:end:step
        ]

        u2 = mda.Merge(select_atom)
        u2.load_new(
            coordinates,
            format=MemoryReader,
            forces=forces,
            dimensions=dimensions,
        )

        logger.debug("MDAnalysis.Universe - reduced universe (frame-selected): %s", u2)
        return u2

    def select_atoms(self, u: mda.Universe, select_string: str = "all") -> mda.Universe:
        """Create a reduced universe by dropping atoms according to user selection.

        Parameters
        ----------
        u:
            A Universe object with topology, coordinates and (optionally) forces.
        select_string:
            MDAnalysis `select_atoms` selection string.

        Returns
        -------
        mda.Universe:
            A reduced universe containing only the selected atoms. Coordinates,
            forces (if present) and dimensions are loaded into memory.
        """
        select_atom = u.select_atoms(select_string, updating=True)

        coordinates = self._extract_timeseries(select_atom, kind="positions")
        forces = self._extract_timeseries(select_atom, kind="forces")
        dimensions = self._extract_timeseries(select_atom, kind="dimensions")

        u2 = mda.Merge(select_atom)
        u2.load_new(
            coordinates,
            format=MemoryReader,
            forces=forces,
            dimensions=dimensions,
        )

        logger.debug("MDAnalysis.Universe - reduced universe (atom-selected): %s", u2)
        return u2

    def extract_fragment(
        self, universe: mda.Universe, molecule_id: int
    ) -> mda.Universe:
        """Extract a single molecule (fragment) as a standalone reduced universe.

        Args:
            universe:
                The source universe.
            molecule_id:
                Fragment index in `universe.atoms.fragments`.

        Returns:
            mda.Universe:
                A reduced universe containing only the atoms of the selected fragment.
        """
        frag = universe.atoms.fragments[molecule_id]
        selection_string = f"index {frag.indices[0]}:{frag.indices[-1]}"
        return self.select_atoms(universe, selection_string)

    def merge_forces(
        self,
        tprfile: str,
        trrfile,
        forcefile: str,
        fileformat: Optional[str] = None,
        kcal: bool = False,
        *,
        force_format: Optional[str] = None,
        fallback_to_positions_if_no_forces: bool = True,
    ) -> mda.Universe:
        """Create a universe by merging coordinates and forces from different files.

        This method loads:
        - coordinates + dimensions from the coordinate trajectory (tprfile + trrfile)
        - forces from the force trajectory (tprfile + forcefile)

        If the force trajectory does not expose forces in MDAnalysis (e.g. the file
        does not contain forces, or the reader does not provide them), then:
        - if `fallback_to_positions_if_no_forces` is True, positions from the force
          trajectory are used as the "forces" array (backwards-compatible behaviour
          with earlier implementations).
        - otherwise, the underlying `NoDataError` is raised.

        Args:
            tprfile:
                Topology input file.
            trrfile:
                Coordinate trajectory file(s). This can be a single path or a list,
                as accepted by MDAnalysis.
            forcefile:
                Trajectory containing forces.
            fileformat:
                Optional file format for the coordinate trajectory, as recognised by
                MDAnalysis.
            kcal:
                If True, scale the force array by 4.184 to convert from kcal to kJ.
            force_format:
                Optional file format for the force trajectory. If not provided, uses
                `fileformat`.
            fallback_to_positions_if_no_forces:
                If True, and the force trajectory has no accessible forces, use
                positions from the force trajectory as a fallback (legacy behaviour).

        Returns:
            mda.Universe:
                A new Universe containing coordinates, forces and dimensions loaded
                into memory.
        """
        logger.debug("Loading coordinate Universe with %s", trrfile)
        u = mda.Universe(tprfile, trrfile, format=fileformat)

        ff = force_format if force_format is not None else fileformat
        logger.debug("Loading force Universe with %s", forcefile)
        u_force = mda.Universe(tprfile, forcefile, format=ff)

        select_atom = u.select_atoms("all")
        select_atom_force = u_force.select_atoms("all")

        coordinates = self._extract_timeseries(select_atom, kind="positions")
        dimensions = self._extract_timeseries(select_atom, kind="dimensions")

        forces = self._extract_force_timeseries_with_fallback(
            select_atom_force,
            fallback_to_positions_if_no_forces=fallback_to_positions_if_no_forces,
        )

        if kcal:
            forces *= 4.184

        logger.debug("Merging forces with coordinates universe.")
        new_universe = mda.Merge(select_atom)
        new_universe.load_new(
            coordinates,
            forces=forces,
            dimensions=dimensions,
        )

        return new_universe

    def _extract_timeseries(self, atomgroup, *, kind: str):
        """Extract a time series array for the requested kind from an AtomGroup.

        Args:
            atomgroup:
                MDAnalysis AtomGroup (may be updating).
            kind:
                One of {"positions", "forces", "dimensions"}.

        Returns:
            np.ndarray:
                Time series with shape:
                - positions: (n_frames, n_atoms, 3)
                - forces:    (n_frames, n_atoms, 3) if available, else raises
                NoDataError
                - dimensions:(n_frames, 6) or (n_frames, 3) depending on reader
        """
        if kind == "positions":
            func = self._positions_copy
        elif kind == "forces":
            func = self._forces_copy
        elif kind == "dimensions":
            func = self._dimensions_copy
        else:
            raise ValueError(f"Unknown timeseries kind: {kind}")

        return AnalysisFromFunction(func, atomgroup).run().results["timeseries"]

    def _positions_copy(self, ag):
        """Return a copy of positions for AnalysisFromFunction."""
        return ag.positions.copy()

    def _forces_copy(self, ag):
        """Return a copy of forces for AnalysisFromFunction."""
        return ag.forces.copy()

    def _dimensions_copy(self, ag):
        """Return a copy of box dimensions for AnalysisFromFunction."""
        return ag.dimensions.copy()

    def _extract_force_timeseries_with_fallback(
        self,
        atomgroup_force,
        *,
        fallback_to_positions_if_no_forces: bool,
    ):
        """Extract force timeseries, optionally falling back to positions.

        This isolates the behaviour that changed your runtime outcome: older code
        used positions from the force trajectory, which never triggered `NoDataError`.
        This method keeps that behaviour available for backwards compatibility.
        """
        try:
            return self._extract_timeseries(atomgroup_force, kind="forces")
        except NoDataError:
            if not fallback_to_positions_if_no_forces:
                raise
            return self._extract_timeseries(atomgroup_force, kind="positions")
