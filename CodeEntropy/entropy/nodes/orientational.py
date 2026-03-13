"""Node for computing orientational entropy from neighbors."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping, Sequence
from typing import (
    Any,
)

import numpy as np

from CodeEntropy.entropy.orientational import OrientationalEntropy

logger = logging.getLogger(__name__)

GroupId = int
ResidueId = int
StateKey = tuple[GroupId, ResidueId]
StateSequence = Sequence[Any] | np.ndarray


class OrientationalEntropyNode:
    """Compute orientational entropy using precomputed neighbors and symmetry.

    This node reads number of neighbors and symmetry from ``shared_data`` and
    computes entropy contributions at the molecular (highest) level.

    Results are written back into ``shared_data["orientational_entropy"]``.
    """

    def run(self, shared_data: MutableMapping[str, Any], **_: Any) -> dict[str, Any]:
        """Execute orientational entropy calculation.

        Args:
            shared_data: Shared workflow state dictionary.

        Returns:
            Dictionary containing orientational entropy results.

        Raises:
            KeyError: If required keys are missing.
        """
        groups = shared_data["groups"]
        levels = shared_data["levels"]
        neighbors = shared_data["neighbors"]
        symmetry_number = shared_data["symmetry_number"]
        linear = shared_data["linear"]
        reporter = shared_data.get("reporter")

        oe = self._build_entropy_engine()

        results: dict[int, float] = {}

        for group_id, mol_ids in groups.items():
            results[group_id] = 0
            if not mol_ids:
                continue
            rep_mol_id = mol_ids[0]
            highest_level = levels[rep_mol_id][-1]

            neighbor = neighbors[group_id]
            symmetry = symmetry_number[group_id]
            line = linear[group_id]

            result_value = oe.calculate(
                neighbor,
                symmetry,
                line,
            )
            results[group_id] = result_value

            if reporter is not None:
                reporter.add_results_data(
                    group_id, highest_level, "Orientational", result_value
                )

        shared_data["orientational_entropy"] = results

        return {"orientational_entropy": results}

    def _build_entropy_engine(self) -> OrientationalEntropy:
        """Create the entropy calculation engine."""
        return OrientationalEntropy()
