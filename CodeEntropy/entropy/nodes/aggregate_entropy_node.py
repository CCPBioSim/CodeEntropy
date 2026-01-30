# CodeEntropy/entropy/nodes/aggregate_entropy_node.py

from typing import Any, Dict


class AggregateEntropyNode:
    """
    Aggregates entropy outputs into shared_data for downstream use.
    """

    def run(
        self,
        shared_data: Dict[str, Any],
        vibrational_entropy=None,
        configurational_entropy=None,
        **_,
    ):
        shared_data["entropy_results"] = {
            "vibrational": vibrational_entropy,
            "configurational": configurational_entropy,
        }
        return {"entropy_results": shared_data["entropy_results"]}
