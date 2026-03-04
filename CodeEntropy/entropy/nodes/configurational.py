"""Node for computing configurational entropy from conformational states."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import (
    Any,
)

import numpy as np

from CodeEntropy.entropy.configurational import ConformationalEntropy

logger = logging.getLogger(__name__)

GroupId = int
ResidueId = int
StateKey = tuple[GroupId, ResidueId]
StateSequence = Sequence[Any] | np.ndarray


class ConfigurationalEntropyNode:
    """Compute configurational entropy using precomputed conformational states.

    This node reads conformational state assignments from ``shared_data`` and
    computes entropy contributions at different structural levels.

    Results are written back into ``shared_data["configurational_entropy"]``.
    """

    def run(self, shared_data: MutableMapping[str, Any], **_: Any) -> dict[str, Any]:
        """Execute configurational entropy calculation.

        Args:
            shared_data: Shared workflow state dictionary.

        Returns:
            Dictionary containing configurational entropy results.

        Raises:
            KeyError: If required keys are missing.
        """
        n_frames = self._get_n_frames(shared_data)
        groups = shared_data["groups"]
        levels = shared_data["levels"]
        universe = shared_data["reduced_universe"]
        reporter = shared_data.get("reporter")

        states_ua, states_res = self._get_state_containers(shared_data)
        ce = self._build_entropy_engine()

        fragments = universe.atoms.fragments
        results: dict[int, dict[str, float]] = {}

        for group_id, mol_ids in groups.items():
            results[group_id] = {"ua": 0.0, "res": 0.0, "poly": 0.0}
            if not mol_ids:
                continue

            rep_mol_id = mol_ids[0]
            rep_mol = fragments[rep_mol_id]
            level_list = levels[rep_mol_id]

            if "united_atom" in level_list:
                ua_total = self._compute_ua_entropy_for_group(
                    ce=ce,
                    group_id=group_id,
                    residues=rep_mol.residues,
                    states_ua=states_ua,
                    n_frames=n_frames,
                    reporter=reporter,
                )
                results[group_id]["ua"] = ua_total

            if "residue" in level_list:
                res_val = self._compute_residue_entropy_for_group(
                    ce=ce,
                    group_id=group_id,
                    states_res=states_res,
                    n_frames=n_frames,
                )
                results[group_id]["res"] = res_val

                if reporter is not None:
                    reporter.add_results_data(
                        group_id, "residue", "Conformational", res_val
                    )

        shared_data["configurational_entropy"] = results

        return {"configurational_entropy": results}

    def _build_entropy_engine(self) -> ConformationalEntropy:
        """Create the entropy calculation engine."""
        return ConformationalEntropy()

    def _get_state_containers(
        self, shared_data: Mapping[str, Any]
    ) -> tuple[
        dict[StateKey, StateSequence],
        dict[GroupId, StateSequence] | Sequence[StateSequence | None],
    ]:
        """Retrieve conformational state containers.

        Args:
            shared_data: Shared workflow state.

        Returns:
            Tuple of united atom and residue state containers.
        """
        conf_states = shared_data.get("conformational_states", {}) or {}
        return conf_states.get("ua", {}) or {}, conf_states.get("res", {})

    def _get_n_frames(self, shared_data: Mapping[str, Any]) -> int:
        """Return the number of frames analysed.

        Args:
            shared_data: Shared workflow state.

        Returns:
            Number of frames.

        Raises:
            KeyError: If frame count is missing.
        """
        n_frames = shared_data.get("n_frames", shared_data.get("number_frames"))
        if n_frames is None:
            raise KeyError("shared_data must contain n_frames or number_frames")
        return int(n_frames)

    def _compute_ua_entropy_for_group(
        self,
        *,
        ce: ConformationalEntropy,
        group_id: int,
        residues: Iterable[Any],
        states_ua: Mapping[StateKey, StateSequence],
        n_frames: int,
        reporter: Any | None,
    ) -> float:
        """Compute united atom entropy for a group.

        Args:
            ce: Entropy calculator.
            group_id: Group identifier.
            residues: Residue iterable.
            states_ua: Mapping of states.
            n_frames: Frame count.
            reporter: Optional logger.

        Returns:
            Total entropy for united atom level.
        """
        total = 0.0

        for res_id, res in enumerate(residues):
            states = states_ua.get((group_id, res_id))
            val = self._entropy_or_zero(ce, states)
            total += val

            if reporter is not None:
                reporter.add_residue_data(
                    group_id=group_id,
                    resname=getattr(res, "resname", "UNK"),
                    level="united_atom",
                    entropy_type="Conformational",
                    frame_count=n_frames,
                    value=val,
                )

        if reporter is not None:
            reporter.add_results_data(group_id, "united_atom", "Conformational", total)

        return total

    def _compute_residue_entropy_for_group(
        self,
        *,
        ce: ConformationalEntropy,
        group_id: int,
        states_res: dict[int, StateSequence] | Sequence[StateSequence | None],
        n_frames: int,
    ) -> float:
        """Compute residue-level entropy for a group."""
        group_states = self._get_group_states(states_res, group_id)
        return self._entropy_or_zero(ce, group_states)

    def _entropy_or_zero(
        self,
        ce: ConformationalEntropy,
        states: StateSequence | None,
    ) -> float:
        """Return entropy value or zero if no state data exists."""
        if not self._has_state_data(states):
            return 0.0
        return float(ce.conformational_entropy_calculation(states))

    @staticmethod
    def _get_group_states(
        states_res: dict[int, StateSequence] | Sequence[StateSequence | None],
        group_id: int,
    ) -> StateSequence | None:
        """Fetch group states from container."""
        if isinstance(states_res, dict):
            return states_res.get(group_id)
        if group_id < len(states_res):
            return states_res[group_id]
        return None

    @staticmethod
    def _has_state_data(states: StateSequence | None) -> bool:
        """Check if state container has usable data."""
        if states is None:
            return False
        if isinstance(states, np.ndarray):
            return bool(np.any(states))
        try:
            return any(states)
        except TypeError:
            return bool(states)
