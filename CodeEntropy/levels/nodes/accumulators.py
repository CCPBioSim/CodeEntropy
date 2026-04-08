"""Initialize covariance accumulators.

This module defines a LevelDAG static node that allocates all per-frame reduction
accumulators (means) and counters used by downstream frame processing.

The node owns only initialization concerns (single responsibility):
- create group-id <-> index mappings
- allocate force/torque covariance mean containers
- allocate optional combined force-torque (FT) mean containers
- allocate per-level frame counters

The structure created here is treated as the canonical storage layout for the
rest of the pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SharedData = MutableMapping[str, Any]


@dataclass(frozen=True)
class GroupIndex:
    """Bidirectional mapping between group ids and contiguous indices."""

    group_id_to_index: dict[int, int]
    index_to_group_id: list[int]


@dataclass(frozen=True)
class CovarianceAccumulators:
    """Container for covariance mean accumulators and frame counters."""

    force_covariances: dict[str, Any]
    torque_covariances: dict[str, Any]
    frame_counts: dict[str, Any]
    forcetorque_covariances: dict[str, Any]
    forcetorque_counts: dict[str, Any]


class InitCovarianceAccumulatorsNode:
    """Allocate accumulators and counters for per-frame reductions.

    Produces the following keys in `shared_data`:

    Canonical mean accumulators:
        - force_covariances: {"ua": dict, "res": list, "poly": list}
        - torque_covariances: {"ua": dict, "res": list, "poly": list}
        - forcetorque_covariances: {"res": list, "poly": list}  (6N x 6N means)

    Counters:
        - frame_counts: {"ua": dict, "res": np.ndarray[int], "poly": np.ndarray[int]}
        - forcetorque_counts: {"res": np.ndarray[int], "poly": np.ndarray[int]}

    Group index mapping:
        - group_id_to_index: {group_id: index}
        - index_to_group_id: [group_id_by_index]

    Backwards-compatible aliases (kept for older consumers):
        - force_torque_stats -> forcetorque_covariances
        - force_torque_counts -> forcetorque_counts
    """

    def run(self, shared_data: dict[str, Any]) -> dict[str, Any]:
        """Initialize and attach all accumulator structures into shared_data.

        Args:
            shared_data: Shared pipeline dictionary. Must contain "groups".

        Returns:
            A dict of keys written into shared_data.

        Raises:
            KeyError: If "groups" is missing from shared_data.
        """
        groups = shared_data["groups"]
        group_index = self._build_group_index(groups)

        accumulators = self._build_accumulators(
            n_groups=len(group_index.index_to_group_id)
        )

        self._attach_to_shared_data(shared_data, group_index, accumulators)
        self._attach_backwards_compatible_aliases(shared_data)

        return self._build_return_payload(shared_data)

    @staticmethod
    def _build_group_index(groups: dict[int, Any]) -> GroupIndex:
        """Build group id <-> index mappings.

        Args:
            groups: Mapping of group id to group members.

        Returns:
            GroupIndex mapping object.
        """
        group_ids = sorted(groups.keys())
        gid2i = {gid: i for i, gid in enumerate(group_ids)}
        return GroupIndex(group_id_to_index=gid2i, index_to_group_id=list(group_ids))

    @staticmethod
    def _build_accumulators(n_groups: int) -> CovarianceAccumulators:
        """Allocate empty covariance means and counters.

        Args:
            n_groups: Number of molecule groups.

        Returns:
            CovarianceAccumulators containing allocated containers.
        """
        force_cov = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}
        torque_cov = {"ua": {}, "res": [None] * n_groups, "poly": [None] * n_groups}

        frame_counts = {
            "ua": {},
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        forcetorque_cov = {"res": [None] * n_groups, "poly": [None] * n_groups}
        forcetorque_counts = {
            "res": np.zeros(n_groups, dtype=int),
            "poly": np.zeros(n_groups, dtype=int),
        }

        return CovarianceAccumulators(
            force_covariances=force_cov,
            torque_covariances=torque_cov,
            frame_counts=frame_counts,
            forcetorque_covariances=forcetorque_cov,
            forcetorque_counts=forcetorque_counts,
        )

    @staticmethod
    def _attach_to_shared_data(
        shared_data: SharedData, group_index: GroupIndex, acc: CovarianceAccumulators
    ) -> None:
        """Attach canonical structures to shared_data.

        Args:
            shared_data: Shared pipeline dictionary.
            group_index: GroupIndex object.
            acc: CovarianceAccumulators object.
        """
        shared_data["group_id_to_index"] = group_index.group_id_to_index
        shared_data["index_to_group_id"] = group_index.index_to_group_id

        shared_data["force_covariances"] = acc.force_covariances
        shared_data["torque_covariances"] = acc.torque_covariances
        shared_data["frame_counts"] = acc.frame_counts

        shared_data["forcetorque_covariances"] = acc.forcetorque_covariances
        shared_data["forcetorque_counts"] = acc.forcetorque_counts

    @staticmethod
    def _attach_backwards_compatible_aliases(shared_data: SharedData) -> None:
        """Attach backwards-compatible aliases.

        Args:
            shared_data: Shared pipeline dictionary.
        """
        shared_data["force_torque_stats"] = {
            "res": list(shared_data["forcetorque_covariances"]["res"]),
            "poly": list(shared_data["forcetorque_covariances"]["poly"]),
        }
        shared_data["force_torque_counts"] = {
            "res": shared_data["forcetorque_counts"]["res"].copy(),
            "poly": shared_data["forcetorque_counts"]["poly"].copy(),
        }

    @staticmethod
    def _build_return_payload(shared_data: SharedData) -> dict[str, Any]:
        """Build the return payload containing initialized keys.

        Args:
            shared_data: Shared pipeline dictionary.

        Returns:
            Dict of keys to values that were set in shared_data.
        """
        return {
            "group_id_to_index": shared_data["group_id_to_index"],
            "index_to_group_id": shared_data["index_to_group_id"],
            "force_covariances": shared_data["force_covariances"],
            "torque_covariances": shared_data["torque_covariances"],
            "frame_counts": shared_data["frame_counts"],
            "forcetorque_covariances": shared_data["forcetorque_covariances"],
            "forcetorque_counts": shared_data["forcetorque_counts"],
            "force_torque_stats": shared_data["force_torque_stats"],
            "force_torque_counts": shared_data["force_torque_counts"],
        }
