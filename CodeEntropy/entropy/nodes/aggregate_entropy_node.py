from typing import Any, Dict


class AggregateEntropyNode:
    def run(self, shared_data: Dict[str, Any], **_) -> Dict[str, Any]:
        out = {
            "vibrational_entropy": shared_data.get("vibrational_entropy"),
            "configurational_entropy": shared_data.get("configurational_entropy"),
        }
        shared_data["entropy_results"] = out
        return {"entropy_results": out}
