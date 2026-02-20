"""Command-line entry point for CodeEntropy.

This module provides the CLI entry point used to run the multiscale cell
correlation entropy workflow.

The entry point is intentionally small and only responsible for:
  1) Creating a job folder.
  2) Constructing a RunManager.
  3) Executing the entropy workflow.
  4) Handling fatal errors with a non-zero exit code.
"""

from __future__ import annotations

import logging

from CodeEntropy.config.runtime import RunManager

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the entropy workflow.

    Raises:
        SystemExit: Exits with status code 1 on any unhandled exception.
    """
    folder = RunManager.create_job_folder()

    try:
        run_manager = RunManager(folder=folder)
        run_manager.run_entropy_workflow()
    except Exception as exc:
        logger.critical(
            "Fatal error during entropy calculation: %s", exc, exc_info=True
        )
        raise SystemExit(1) from exc
