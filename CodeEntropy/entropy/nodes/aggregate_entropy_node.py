import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AggregateEntropyNode:
    """
    Aggregates entropy results for convenience.
    """

    def run(
        self,
        shared_data: Dict[str, Any],
        vibrational_entropy: Dict[str, Any],
        configurational_entropy: Dict[str, Any],
        **_kwargs,
    ) -> Dict[str, Any]:
        out = {
            "vibrational_entropy": vibrational_entropy.get("vibrational_entropy", {}),
            "configurational_entropy": configurational_entropy.get(
                "configurational_entropy", {}
            ),
        }
        logger.info("[AggregateEntropyNode] Done")
        return {"entropy": out}
