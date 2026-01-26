import logging
from typing import Any, Dict

from CodeEntropy.levels.dihedral_tools import DihedralAnalysis

logger = logging.getLogger(__name__)


class ComputeConformationalStatesNode:
    """
    Builds conformational state descriptors (UA + residue) from dihedral angles,
    and stores them into shared_data using a stable contract.
    """

    def __init__(self, universe_operations):
        self._dih = DihedralAnalysis(universe_operations)

    def run(self, shared_data: Dict[str, Any]):
        u = shared_data["reduced_universe"]
        levels = shared_data["levels"]
        groups = shared_data["groups"]

        start = shared_data["start"]
        end = shared_data["end"]
        step = shared_data["step"]
        bin_width = shared_data["args"].bin_width

        logger.info("[ComputeConformationalStatesNode] Building conformational states")

        states_ua, states_res = self._dih.build_conformational_states(
            data_container=u,
            levels=levels,
            groups=groups,
            start=start,
            end=end,
            step=step,
            bin_width=bin_width,
        )

        shared_data["conformational_states"] = {
            "ua": states_ua,
            "res": states_res,
        }

        shared_data["states_united_atom"] = states_ua
        shared_data["states_residue"] = states_res
