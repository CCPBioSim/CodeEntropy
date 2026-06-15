"""Internal policy for hierarchy-level frame map-reduce execution.

Users provide compute resources. CodeEntropy keeps scheduling choices such as
chunk size and in-flight task limits internal so the public configuration remains
stable and simple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionPolicy:
    """Internal policy for scalable, deterministic frame execution."""

    target_frame_chunks_per_worker: int = 2
    min_frame_chunk_size: int = 1
    max_frame_chunk_size: int = 32
    max_frame_in_flight_multiplier: int = 1

    def requested_worker_count(self, shared_data: dict[str, Any]) -> int:
        """Return the worker-process count requested by the current run."""
        args = shared_data.get("args")

        dask_workers = getattr(args, "dask_workers", None)
        if dask_workers is not None:
            return max(1, int(dask_workers))

        if bool(getattr(args, "hpc", False)):
            hpc_nodes = max(1, int(getattr(args, "hpc_nodes", 1) or 1))
            hpc_processes = max(1, int(getattr(args, "hpc_processes", 1) or 1))
            return hpc_nodes * hpc_processes

        return 1

    def frame_chunk_size(self, shared_data: dict[str, Any], *, n_frames: int) -> int:
        """Choose a deterministic frame chunk size for the current run."""
        n_frames = max(1, int(n_frames))
        workers = self.requested_worker_count(shared_data)
        target_chunks = max(1, workers * int(self.target_frame_chunks_per_worker))
        chunk_size = math.ceil(n_frames / target_chunks)

        return max(
            int(self.min_frame_chunk_size),
            min(int(self.max_frame_chunk_size), int(chunk_size)),
        )

    def max_frame_in_flight_tasks(
        self,
        shared_data: dict[str, Any],
        *,
        n_chunks: int,
    ) -> int:
        """Choose how many frame chunks may be active at once."""
        workers = self.requested_worker_count(shared_data)
        max_in_flight = workers * int(self.max_frame_in_flight_multiplier)
        return max(1, min(int(n_chunks), int(max_in_flight)))
