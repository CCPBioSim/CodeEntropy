"""Entropy manager orchestration.

This module defines `EntropyWorkflow`, which coordinates the end-to-end entropy
workflow:
- Determine trajectory bounds and frame count.
- Build a reduced universe based on atom selection.
- Identify molecule groups and hierarchy levels.
- Optionally compute water entropy and adjust selection.
- Execute the level DAG (matrix/state preparation).
- Execute the entropy graph (entropy calculations and aggregation).
- Finalize and persist results.

The manager intentionally delegates calculations to dedicated components.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from CodeEntropy.core.logging import LoggingConfig
from CodeEntropy.entropy.graph import EntropyGraph
from CodeEntropy.entropy.water import WaterEntropy
from CodeEntropy.levels.hierarchy import HierarchyBuilder
from CodeEntropy.levels.level_dag import LevelDAG

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()

SharedData = Dict[str, Any]


@dataclass(frozen=True)
class TrajectorySlice:
    """Trajectory slicing parameters.

    Attributes:
        start: Inclusive start frame index.
        end: Exclusive end frame index (or a concrete index derived from args).
        step: Step size between frames.
        n_frames: Number of frames in the slice.
    """

    start: int
    end: int
    step: int
    n_frames: int


class EntropyWorkflow:
    """Coordinate entropy calculations across structural levels.

    This class is responsible for orchestration and IO-level concerns (selection,
    grouping, running graphs, and finalizing results). Domain calculations live in
    dedicated components (LevelDAG, EntropyGraph, WaterEntropy, etc.).
    """

    def __init__(
        self,
        run_manager: Any,
        args: Any,
        universe: Any,
        reporter: Any,
        group_molecules: Any,
        dihedral_analysis: Any,
        universe_operations: Any,
    ) -> None:
        """Initialize the entropy workflow manager.

        Args:
            run_manager: Manager for universe IO and unit conversions.
            args: Parsed CLI/user arguments.
            universe: MDAnalysis Universe representing the simulation system.
            reporter: Collector for per-molecule and per-residue outputs.
            group_molecules: Component that groups molecules for averaging.
            dihedral_analysis: Component used to compute conformational states.
                (Stored for completeness; computation is typically triggered by nodes.)
            universe_operations: Adapter providing common universe operations.
        """
        self._run_manager = run_manager
        self._args = args
        self._universe = universe
        self._reporter = reporter
        self._group_molecules = group_molecules
        self._dihedral_analysis = dihedral_analysis
        self._universe_operations = universe_operations

    def execute(self) -> None:
        """Run the full entropy workflow and emit results.

        This orchestrates the complete entropy pipeline:
            1. Build trajectory slice.
            2. Apply atom selection to create a reduced universe.
            3. Detect hierarchy levels.
            4. Group molecules.
            5. Split groups into water and non-water.
            6. Optionally compute water entropy (only if solute exists).
            7. Run level DAG and entropy graph.
            8. Finalize and persist results.
        """
        traj = self._build_trajectory_slice()
        console.print(
            f"Analyzing a total of {traj.n_frames} frames in this calculation."
        )

        reduced_universe = self._build_reduced_universe()

        levels = self._detect_levels(reduced_universe)
        groups = self._group_molecules.grouping_molecules(
            reduced_universe, self._args.grouping
        )

        nonwater_groups, water_groups = self._split_water_groups(
            reduced_universe, groups
        )

        if self._args.water_entropy and water_groups and nonwater_groups:
            self._compute_water_entropy(traj, water_groups)
        else:
            nonwater_groups.update(water_groups)

        shared_data = self._build_shared_data(
            reduced_universe=reduced_universe,
            levels=levels,
            groups=nonwater_groups,
            traj=traj,
        )

        with self._reporter.progress(transient=False) as p:
            self._run_level_dag(shared_data, progress=p)
            self._run_entropy_graph(shared_data, progress=p)

        self._finalize_molecule_results()
        self._reporter.log_tables()

    def _build_shared_data(
        self,
        reduced_universe: Any,
        levels: Any,
        groups: Mapping[int, Any],
        traj: TrajectorySlice,
    ) -> SharedData:
        """Build the shared_data dict used by nodes and graphs.

        Args:
            reduced_universe: Universe after applying selection.
            levels: Level definition per molecule id.
            groups: Mapping of group id -> list of molecule ids.
            traj: Trajectory slice parameters.

        Returns:
            Shared data dictionary for DAG/graph execution.
        """
        shared_data: SharedData = {
            "entropy_manager": self,
            "run_manager": self._run_manager,
            "reporter": self._reporter,
            "args": self._args,
            "universe": self._universe,
            "reduced_universe": reduced_universe,
            "levels": levels,
            "groups": dict(groups),
            "start": traj.start,
            "end": traj.end,
            "step": traj.step,
            "n_frames": traj.n_frames,
        }
        return shared_data

    def _run_level_dag(
        self, shared_data: SharedData, *, progress: object | None = None
    ) -> None:
        """Execute the structural/level DAG.

        Args:
            shared_data: Shared data dict that will be mutated by the DAG.
            progress: Optional progress sink provided by ResultsReporter.progress().
        """
        LevelDAG(self._universe_operations).build().execute(
            shared_data, progress=progress
        )

    def _run_entropy_graph(
        self, shared_data: SharedData, *, progress: object | None = None
    ) -> None:
        """Execute the entropy calculation graph and merge results into shared_data.

        Args:
            shared_data: Shared data dict that will be mutated by the graph.
            progress: Optional progress sink provided by ResultsReporter.progress().
        """
        entropy_results = EntropyGraph().build().execute(shared_data, progress=progress)
        shared_data.update(entropy_results)

    def _build_trajectory_slice(self) -> TrajectorySlice:
        """Compute trajectory slicing parameters from args.

        Returns:
            A TrajectorySlice describing the frames to analyze.
        """
        start, end, step = self._get_trajectory_bounds()
        n_frames = self._get_number_frames(start, end, step)
        return TrajectorySlice(start=start, end=end, step=step, n_frames=n_frames)

    def _get_trajectory_bounds(self) -> Tuple[int, int, int]:
        """Return start, end, and step frame indices from args.

        Returns:
            Tuple of (start, end, step).
        """
        start = self._args.start or 0
        end = len(self._universe.trajectory) if self._args.end == -1 else self._args.end
        step = self._args.step or 1
        return start, end, step

    def _get_number_frames(self, start: int, end: int, step: int) -> int:
        """Compute the number of frames in a trajectory slice.

        Args:
            start: Inclusive start frame index.
            end: Exclusive end frame index.
            step: Step between frames.

        Returns:
            Number of frames processed.
        """
        return math.floor((end - start) / step)

    def _build_reduced_universe(self) -> Any:
        """Apply atom selection and return the reduced universe.

        If `selection_string` is "all", the original universe is returned.

        Returns:
            MDAnalysis Universe (original or reduced).
        """
        selection = self._args.selection_string
        if selection == "all":
            return self._universe

        reduced = self._universe_operations.select_atoms(self._universe, selection)
        name = f"{len(reduced.trajectory)}_frame_dump_atom_selection"
        self._run_manager.write_universe(reduced, name)
        return reduced

    def _detect_levels(self, reduced_universe: Any) -> Any:
        """Detect hierarchy levels for each molecule in the reduced universe.

        Args:
            reduced_universe: Reduced MDAnalysis Universe.

        Returns:
            Levels structure as returned by `HierarchyBuilder.select_levels`.
        """
        level_hierarchy = HierarchyBuilder()
        _number_molecules, levels = level_hierarchy.select_levels(reduced_universe)
        return levels

    def _split_water_groups(
        self,
        universe: Any,
        groups: Mapping[int, Any],
    ) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        """Partition molecule groups into water and non-water groups.

        This method identifies which molecule groups correspond to water
        molecules based on residue membership.

        Args:
            universe (Any):
                The MDAnalysis Universe used to build the molecule groups
                (typically the reduced_universe).
            groups (Mapping[int, Any]):
                Mapping of group_id -> list of molecule fragment indices.

        Returns:
            Tuple[Dict[int, Any], Dict[int, Any]]:
                A tuple containing:

                - nonwater_groups:
                    Mapping of group_id -> molecule ids that are NOT water.
                - water_groups:
                    Mapping of group_id -> molecule ids that contain water.
        """
        water_atoms = universe.select_atoms("water")
        water_resids = {res.resid for res in water_atoms.residues}

        water_groups = {
            gid: mol_ids
            for gid, mol_ids in groups.items()
            if any(
                res.resid in water_resids
                for mol in [universe.atoms.fragments[i] for i in mol_ids]
                for res in mol.residues
            )
        }
        nonwater_groups = {
            gid: g for gid, g in groups.items() if gid not in water_groups
        }
        return nonwater_groups, water_groups

    def _compute_water_entropy(
        self, traj: TrajectorySlice, water_groups: Mapping[int, Any]
    ) -> None:
        """Compute water entropy for each water group and adjust selection string.

        Args:
            traj: Trajectory slice parameters.
            water_groups: Mapping of group id -> molecule ids for waters.
        """
        if not water_groups or not self._args.water_entropy:
            return

        water_entropy = WaterEntropy(self._args, self._reporter)

        for group_id in water_groups.keys():
            water_entropy.calculate_and_log(
                universe=self._universe,
                start=traj.start,
                end=traj.end,
                step=traj.step,
                group_id=group_id,
            )

        self._args.selection_string = (
            f"{self._args.selection_string} and not water"
            if self._args.selection_string != "all"
            else "not water"
        )

        logger.debug("WaterEntropy: molecule_data=%s", self._reporter.molecule_data)
        logger.debug("WaterEntropy: residue_data=%s", self._reporter.residue_data)

    def _finalize_molecule_results(self) -> None:
        """Aggregate group totals and persist results to JSON.

        Computes total entropy per group and appends "Group Total" rows to the
        molecule results table, then writes molecule and residue tables to the
        configured output file via the data logger.
        """
        entropy_by_group = defaultdict(float)

        for group_id, level, _etype, result in self._reporter.molecule_data:
            if level == "Group Total":
                continue
            try:
                entropy_by_group[group_id] += float(result)
            except (TypeError, ValueError):
                logger.warning("Skipping invalid entry: %s, %s", group_id, result)

        for group_id, total in entropy_by_group.items():
            self._reporter.molecule_data.append(
                (group_id, "Group Total", "Group Total Entropy", total)
            )

        molecule_df = pd.DataFrame(
            self._reporter.molecule_data,
            columns=["Group ID", "Level", "Type", "Result (J/mol/K)"],
        )
        residue_df = pd.DataFrame(
            self._reporter.residue_data,
            columns=[
                "Group ID",
                "Residue Name",
                "Level",
                "Type",
                "Frame Count",
                "Result (J/mol/K)",
            ],
        )
        self._reporter.save_dataframes_as_json(
            molecule_df,
            residue_df,
            self._args.output_file,
            args=self._args,
            include_raw_tables=False,
        )
