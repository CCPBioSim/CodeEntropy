"""Water entropy aggregation.

This module wraps the waterEntropy routines and maps their
outputs into the project `ResultsReporter` format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import waterEntropy.recipes.interfacial_solvent as GetSolvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WaterEntropyInput:
    """Inputs for water entropy computation.

    Attributes:
        universe: MDAnalysis Universe containing the system.
        start: Start frame index (inclusive).
        end: End frame index (exclusive, or -1 depending on caller convention).
        step: Frame stride.
        temperature: Temperature in Kelvin.
        group_id: Group ID used for logging.
    """

    universe: Any
    start: int
    end: int
    step: int
    temperature: float
    group_id: Optional[int] = None


class WaterEntropy:
    """Compute and log water entropy contributions.

    This class calls the external `waterEntropy` routine to compute:
      - orientational entropy per residue
      - translational vibrational entropy
      - rotational vibrational entropy

    Then it logs residue-level entries and adds a group label.
    """

    def __init__(
        self,
        args: Any,
        reporter: Any,
        solver: Callable[..., Tuple[dict, Any, Any, Any, Any]] = (
            GetSolvent.get_interfacial_water_orient_entropy
        ),
    ) -> None:
        """Initialize the water entropy calculator.

        Args:
            args: Argument namespace; must include `temperature`.
            reporter: Logger used to record residue and group results.
            solver: Callable compatible with
                `get_interfacial_water_orient_entropy
                (universe, start, end, step, temperature, parallel=True)`.
                Dependency injection allows unit testing without the external package.
        """
        self._args = args
        self._reporter = reporter
        self._solver = solver

    def calculate_and_log(
        self,
        universe: Any,
        start: int,
        end: int,
        step: int,
        group_id: Optional[int] = None,
    ) -> None:
        """Compute water entropy and write results to the data logger.

        Args:
            universe: MDAnalysis Universe containing water.
            start: Start frame index.
            end: End frame index.
            step: Frame stride.
            group_id: Group ID to assign all water contributions to.
        """
        inputs = WaterEntropyInput(
            universe=universe,
            start=start,
            end=end,
            step=step,
            temperature=float(self._args.temperature),
            group_id=group_id,
        )
        self._calculate_and_log_from_inputs(inputs)

    def _calculate_and_log_from_inputs(self, inputs: WaterEntropyInput) -> None:
        """Run the solver and log all returned entropy components."""
        Sorient_dict, covariances, vibrations, _unused, _water_count = self._run_solver(
            inputs
        )

        self._log_orientational_entropy(Sorient_dict, inputs.group_id)
        self._log_translational_entropy(vibrations, covariances, inputs.group_id)
        self._log_rotational_entropy(vibrations, covariances, inputs.group_id)
        self._log_group_label(inputs.universe, Sorient_dict, inputs.group_id)

    def _run_solver(self, inputs: WaterEntropyInput):
        """Call the external solver.

        Args:
            inputs: WaterEntropyInput.

        Returns:
            Tuple of solver outputs.
        """
        logger.info(
            "[WaterEntropy] Computing water entropy (start=%s, end=%s, step=%s)",
            inputs.start,
            inputs.end,
            inputs.step,
        )
        return self._solver(
            inputs.universe,
            inputs.start,
            inputs.end,
            inputs.step,
            inputs.temperature,
            parallel=True,
        )

    def _log_orientational_entropy(
        self, Sorient_dict: Mapping[Any, Mapping[str, Any]], group_id: Optional[int]
    ) -> None:
        """Log orientational entropy entries.

        Args:
            Sorient_dict: Mapping of residue ids to {resname: [entropy, count]}.
            group_id: Group ID to assign logs to.
        """
        for _resid, resname_dict in Sorient_dict.items():
            for resname, values in resname_dict.items():
                if isinstance(values, list) and len(values) == 2:
                    entropy, count = values
                    self._reporter.add_residue_data(
                        group_id, resname, "Water", "Orientational", count, entropy
                    )

    def _log_translational_entropy(
        self, vibrations: Any, covariances: Any, group_id: Optional[int]
    ) -> None:
        """Log translational vibrational entropy entries.

        Args:
            vibrations: Solver vibrations object with `translational_S`.
            covariances: Solver covariances object with `counts`.
            group_id: Group ID to assign logs to.
        """
        translational = getattr(vibrations, "translational_S", {}) or {}
        counts = getattr(covariances, "counts", {}) or {}

        for (solute_id, _), entropy in translational.items():
            value = (
                float(np.sum(entropy))
                if isinstance(entropy, (list, np.ndarray))
                else float(entropy)
            )
            count = counts.get((solute_id, "WAT"), 1)
            resname = self._solute_id_to_resname(solute_id)
            self._reporter.add_residue_data(
                group_id, resname, "Water", "Transvibrational", count, value
            )

    def _log_rotational_entropy(
        self, vibrations: Any, covariances: Any, group_id: Optional[int]
    ) -> None:
        """Log rotational vibrational entropy entries.

        Args:
            vibrations: Solver vibrations object with `rotational_S`.
            covariances: Solver covariances object with `counts`.
            group_id: Group ID to assign logs to.
        """
        rotational = getattr(vibrations, "rotational_S", {}) or {}
        counts = getattr(covariances, "counts", {}) or {}

        for (solute_id, _), entropy in rotational.items():
            value = (
                float(np.sum(entropy))
                if isinstance(entropy, (list, np.ndarray))
                else float(entropy)
            )
            count = counts.get((solute_id, "WAT"), 1)
            resname = self._solute_id_to_resname(solute_id)
            self._reporter.add_residue_data(
                group_id, resname, "Water", "Rovibrational", count, value
            )

    def _log_group_label(
        self,
        universe: Any,
        Sorient_dict: Mapping[Any, Mapping[str, Any]],
        group_id: Optional[int],
    ) -> None:
        """Log a group label summarizing the water entries.

        Args:
            universe: MDAnalysis Universe.
            Sorient_dict: Orientational entropy dict used to infer residue names.
            group_id: Group ID.
        """
        water_selection = universe.select_atoms("resname WAT")
        actual_water_residues = len(water_selection.residues)

        water_resnames = set(water_selection.residues.resnames)
        residue_names = {
            resname
            for res_dict in Sorient_dict.values()
            for resname in res_dict.keys()
            if str(resname).upper() in {str(r).upper() for r in water_resnames}
        }

        residue_group = "_".join(sorted(residue_names)) if residue_names else "WAT"
        self._reporter.add_group_label(
            group_id, residue_group, actual_water_residues, len(water_selection.atoms)
        )

    @staticmethod
    def _solute_id_to_resname(solute_id: str) -> str:
        """Convert a solver solute_id to a residue-like name.

        Args:
            solute_id: Identifier returned by the solver.

        Returns:
            Residue name string.
        """
        if "_" in str(solute_id):
            return str(solute_id).rsplit("_", 1)[0]
        return str(solute_id)
