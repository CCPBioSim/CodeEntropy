"""
MDAnalysis universe utilities.

This module contains helpers for creating reduced MDAnalysis `Universe` objects by
sub-selecting frames and/or atoms, and for building a `Universe` that combines
coordinates from one trajectory with forces sourced from a second trajectory.
"""

from __future__ import annotations

import logging
from typing import Any

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.exceptions import NoDataError

logger = logging.getLogger(__name__)


class UniverseOperations:
    """Functions to create and manipulate MDAnalysis Universe objects.

    This helper provides methods to:
      - Build reduced universes by selecting subsets of frames or atoms.
      - Extract a single fragment (molecule) into a standalone universe.
      - Merge coordinates from one trajectory with forces sourced from another
            trajectory.
    """

    def __init__(self) -> None:
        """Initialise the operations helper."""
        self._universe = None

    def select_frames(
        self,
        u: mda.Universe,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
    ) -> mda.Universe:
        """Create a reduced universe from explicit frame bounds.

        Args:
            u: Universe with topology, coordinates and optionally forces.
            start: Inclusive start frame. If None, defaults to 0.
            end: Exclusive stop frame. If None, defaults to full trajectory length.
            step: Frame stride.

        Returns:
            A reduced in-memory universe containing the selected frames.

        Raises:
            ValueError: If ``step`` is not positive or no frames are selected.
        """
        if start is None:
            start = 0
        if end is None:
            end = len(u.trajectory)

        if step <= 0:
            raise ValueError(f"Frame step must be positive, got {step}")

        frame_indices = tuple(range(int(start), int(end), int(step)))
        return self.select_frame_indices(u, frame_indices)

    def select_frame_indices(
        self,
        u: mda.Universe,
        frame_indices: tuple[int, ...] | list[int],
    ) -> mda.Universe:
        """Create a reduced universe from explicit trajectory frame indices.

        Args:
            u: Universe with topology, coordinates and optionally forces.
            frame_indices: Explicit trajectory frame indices to extract.

        Returns:
            A reduced in-memory universe containing the selected frames.

        Raises:
            ValueError: If ``frame_indices`` is empty.
        """
        if not frame_indices:
            raise ValueError(
                "Cannot build a reduced universe from an empty frame list."
            )

        select_atom = u.select_atoms("all", updating=True)
        reduced = self._build_memory_universe_from_atomgroup(select_atom, frame_indices)

        logger.debug(
            "MDAnalysis.Universe - reduced universe (frame-selected): %s", reduced
        )
        return reduced

    def select_atoms(self, u: mda.Universe, select_string: str = "all") -> mda.Universe:
        """Create a reduced universe by selecting atoms.

        Args:
            u: Universe with topology, coordinates and optionally forces.
            select_string: MDAnalysis selection string.

        Returns:
            A reduced universe containing only the selected atoms. Coordinates, forces
            if present, and dimensions are loaded into memory.
        """
        select_atom = u.select_atoms(select_string, updating=True)
        frame_indices = tuple(range(len(u.trajectory)))

        reduced = self._build_memory_universe_from_atomgroup(select_atom, frame_indices)

        logger.debug(
            "MDAnalysis.Universe - reduced universe (atom-selected): %s", reduced
        )
        return reduced

    def _build_memory_universe_from_atomgroup(
        self,
        atomgroup,
        frame_indices: tuple[int, ...] | list[int],
    ) -> mda.Universe:
        """Build an in-memory Universe for an AtomGroup over explicit frames.

        Args:
            atomgroup: MDAnalysis AtomGroup to copy into the new universe.
            frame_indices: Explicit trajectory frame indices to extract.

        Returns:
            In-memory MDAnalysis Universe.

        Raises:
            ValueError: If no frames are provided.
        """
        if not frame_indices:
            raise ValueError("Cannot build a memory universe from an empty frame list.")

        universe = atomgroup.universe

        coordinates: list[np.ndarray] = []
        forces: list[np.ndarray] | None = []
        dimensions: list[np.ndarray] = []

        for frame_index in frame_indices:
            universe.trajectory[int(frame_index)]

            coordinates.append(atomgroup.positions.copy())
            dimensions.append(universe.dimensions.copy())

            if forces is not None:
                try:
                    forces.append(atomgroup.forces.copy())
                except NoDataError:
                    forces = None

        merged = mda.Merge(atomgroup)

        load_kwargs: dict[str, Any] = {
            "format": MemoryReader,
            "dimensions": np.asarray(dimensions),
        }

        if forces is not None:
            load_kwargs["forces"] = np.asarray(forces)

        merged.load_new(
            np.asarray(coordinates),
            **load_kwargs,
        )

        return merged

    def extract_fragment(
        self, universe: mda.Universe, molecule_id: int
    ) -> mda.Universe:
        """Extract a single molecule (fragment) as a standalone reduced universe.

        Args:
            universe: The source universe.
            molecule_id: Fragment index in `universe.atoms.fragments`.

        Returns:
            A reduced universe containing only the atoms of the selected fragment.
        """
        frag = universe.atoms.fragments[molecule_id]
        selection_string = f"index {frag.indices[0]}:{frag.indices[-1]}"
        return self.select_atoms(universe, selection_string)

    def extract_fragment_atomgroup(self, universe: mda.Universe, molecule_id: int):
        """Return a molecule fragment as an AtomGroup.

        This helper mirrors the atom-index range used by ``extract_fragment`` but
        avoids building a standalone in-memory universe. It is intended for
        topology discovery paths where only the static atom selection is needed
        and trajectory coordinates do not need to be copied.

        Args:
            universe: Source MDAnalysis universe.
            molecule_id: Fragment index in ``universe.atoms.fragments``.

        Returns:
            MDAnalysis AtomGroup containing the atoms in the selected fragment.
        """
        frag = universe.atoms.fragments[int(molecule_id)]
        selection_string = f"index {frag.indices[0]}:{frag.indices[-1]}"
        return universe.select_atoms(selection_string, updating=False)

    def convert_lammps(
        self,
        tprfile: str,
        trrfile,
        fileformat: str | None = None,
    ) -> mda.Universe:
        """Update the units produced from the universe produced from LAMMPS
        format topology and trajectory files. MDA currently has a bug that
        results in forces not being converted to the correct units
        (see issue for more details:
        https://github.com/MDAnalysis/mdanalysis/issues/5115
        )
        The method currently expects the following additional columns in the
        lammps dump file: fx fy fz c_5 c_7
        where c_5 and c_7 are the atom potential and kinetic energies respectively.

        This method loads:

        - Coordinates and dimensions from the coordinate trajectory
          (``tprfile`` + ``trrfile``).

        Args:
            tprfile: Topology input file.
            trrfile: Coordinate trajectory file(s). This can be a single path or a
                list, as accepted by MDAnalysis.
            fileformat: Optional file format for the coordinate trajectory, as
                recognised by MDAnalysis.

        Returns:
            MDAnalysis.Universe: A new Universe containing coordinates, forces and
            dimensions loaded into memory.

        Raises:
            ValueError: If fileformat is not one of the supported values.
        """

        def _convert_lammps_forces_energies(ts):
            """
            Convert lammps forces from kcal/mol/Ang to kJ/mol/Ang.
            Assumes columns for per-atom potential (c_5) and kinetic energies (c_7)
            are provided and converts these too.

            Args:
                ts: MDAnalysis timeseries from the trajectory.

            Returns:
                A converted time series.
            """
            ts.forces *= 4.184
            ts.data["c_5"] *= 4.184
            ts.data["c_7"] *= 4.184
            return ts

        def _convert_lammps_forces(ts):
            """
            Convert lammps forces from kcal/mol/Ang to kJ/mol/Ang.

            Args:
                ts: MDAnalysis timeseries from the trajectory.

            Returns:
                A converted time series.
            """
            ts.forces *= 4.184
            return ts

        if fileformat == "LAMMPSDUMP":
            try:
                return mda.Universe(
                    tprfile,
                    trrfile,
                    format=fileformat,
                    additional_columns=["fx", "fy", "fz", "c_5", "c_7"],
                    transformations=[_convert_lammps_forces_energies],
                )
            except KeyError:
                logger.debug(
                    f"Warning: Energy columns not found in LAMMPSDUMP: {trrfile}"
                )
                return mda.Universe(
                    tprfile,
                    trrfile,
                    format=fileformat,
                    additional_columns=["fx", "fy", "fz"],
                    transformations=[_convert_lammps_forces],
                )
        else:
            raise ValueError(
                f"Incorrect file format: {fileformat}, LAMMPSDUMP expected"
            )

    def merge_forces(
        self,
        tprfile: str,
        trrfile,
        forcefile: str,
        fileformat: str | None = None,
        kcal: bool = False,
        *,
        force_format: str | None = None,
        fallback_to_positions_if_no_forces: bool = True,
    ) -> mda.Universe:
        """Create a universe by merging coordinates and forces from different files.

        This method loads:

        - Coordinates and dimensions from the coordinate trajectory
          (``tprfile`` + ``trrfile``).
        - Forces from the force trajectory (``tprfile`` + ``forcefile``).

        If the force trajectory does not expose forces in MDAnalysis (e.g., the file
        does not contain forces, or the reader does not provide them), then:

        - If ``fallback_to_positions_if_no_forces`` is True, positions from the
          force trajectory are used as the "forces" array (backwards-compatible
          behaviour with earlier implementations).
        - Otherwise, the underlying ``NoDataError`` is raised.

        Args:
            tprfile: Topology input file.
            trrfile: Coordinate trajectory file(s). This can be a single path or a
                list, as accepted by MDAnalysis.
            forcefile: Trajectory containing forces.
            fileformat: Optional file format for the coordinate trajectory, as
                recognised by MDAnalysis.
            kcal: If True, scale the force array by 4.184 to convert from kcal to kJ.
            force_format: Optional file format for the force trajectory. If not
                provided, uses ``fileformat``.
            fallback_to_positions_if_no_forces: If True, and the force trajectory has
                no accessible forces, use positions from the force trajectory as a
                fallback (legacy behaviour).

        Returns:
            MDAnalysis.Universe: A new Universe containing coordinates, forces and
            dimensions loaded into memory.

        """
        logger.debug(f"Loading coordinate Universe with {trrfile}")
        u = mda.Universe(tprfile, trrfile, format=fileformat)

        ff = force_format if force_format is not None else fileformat
        logger.debug(f"Loading force Universe with {forcefile}")
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

    def _extract_timeseries(self, atomgroup, *, kind: str) -> np.ndarray:
        """Extract a time series array using explicit frame indexing.

        Args:
            atomgroup: MDAnalysis AtomGroup.
            kind: One of ``"positions"``, ``"forces"``, or ``"dimensions"``.

        Returns:
            NumPy array containing the requested data for all trajectory frames.

        Raises:
            ValueError: If ``kind`` is unknown.
            NoDataError: If ``kind`` is ``"forces"`` and forces are unavailable.
        """
        valid_kinds = {"positions", "forces", "dimensions"}
        if kind not in valid_kinds:
            raise ValueError(f"Unknown timeseries kind: {kind}")

        universe = atomgroup.universe
        values: list[np.ndarray] = []

        for frame_index in range(len(universe.trajectory)):
            universe.trajectory[int(frame_index)]

            if kind == "positions":
                values.append(atomgroup.positions.copy())
            elif kind == "forces":
                values.append(atomgroup.forces.copy())
            else:
                values.append(universe.dimensions.copy())

        return np.asarray(values)

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

        Args:
            atomgroup_force: MDAnalysis AtomGroup sourced from the force trajectory.
            fallback_to_positions_if_no_forces: If True, fall back to extracting
                positions when forces are unavailable; otherwise re-raise NoDataError.

        Returns:
            A time series array of shape (n_frames, n_atoms, 3). The returned array
            contains forces when available, otherwise positions if fallback is enabled.

        Raises:
            NoDataError: If forces are unavailable and
            fallback_to_positions_if_no_forces is False.
        """
        try:
            return self._extract_timeseries(atomgroup_force, kind="forces")
        except NoDataError:
            if not fallback_to_positions_if_no_forces:
                raise
            return self._extract_timeseries(atomgroup_force, kind="positions")
