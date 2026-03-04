"""Aggregates entropy outputs produced by upstream DAG nodes."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

EntropyResults = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AggregateEntropyNode:
    """Aggregate entropy results into a single shared output object.

    This node is intentionally small and single-purpose:
    it gathers previously-computed entropy components from `shared_data`
    and writes a canonical `shared_data["entropy_results"]` mapping.

    Attributes:
        vibrational_key: Key in `shared_data` where vibrational entropy is stored.
        configurational_key: Key in `shared_data` where configurational entropy is
        stored.
        output_key: Key in `shared_data` where the aggregated mapping is written.
    """

    vibrational_key: str = "vibrational_entropy"
    configurational_key: str = "configurational_entropy"
    orientational_key: str = "orientational_entropy"
    output_key: str = "entropy_results"

    def run(
        self, shared_data: MutableMapping[str, Any], **_: Any
    ) -> dict[str, EntropyResults]:
        """Run the aggregation step.

        Args:
            shared_data: Shared workflow state. Must contain (or may contain) keys
                for vibrational and configurational entropy results.

        Returns:
            A dict containing a single key, `"entropy_results"`, which maps to the
            aggregated results dict.

        Side Effects:
            Writes the aggregated results to `shared_data[self.output_key]`.

        Notes:
            This node does not validate the shapes/types of upstream results.
            Validation should live with the producer nodes (single responsibility).
        """
        results = self._collect_entropy_results(shared_data)
        shared_data[self.output_key] = results
        return {self.output_key: results}

    def _collect_entropy_results(
        self, shared_data: Mapping[str, Any]
    ) -> EntropyResults:
        """Collect entropy results from shared data.

        Args:
            shared_data: Shared workflow state.

        Returns:
            A mapping with keys `"vibrational_entropy"` and `"configurational_entropy"`.
        """
        return {
            "vibrational_entropy": self._get_optional(
                shared_data, self.vibrational_key
            ),
            "configurational_entropy": self._get_optional(
                shared_data, self.configurational_key
            ),
            "orientational_entropy": self._get_optional(
                shared_data, self.orientational_key
            ),
        }

    @staticmethod
    def _get_optional(shared_data: Mapping[str, Any], key: str) -> Any | None:
        """Fetch an optional value from shared data.

        Args:
            shared_data: Shared workflow state.
            key: Key to fetch.

        Returns:
            The value if present, otherwise None.
        """
        return shared_data.get(key)
