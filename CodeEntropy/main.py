"""Command-line entry point for CodeEntropy.

This module provides the program entry point used to run the multiscale cell
correlation entropy workflow.

The entry point is intentionally small and only responsible for:
  1) Creating a job folder.
  2) Constructing a :class:`~CodeEntropy.config.run.RunManager`.
  3) Executing the entropy workflow.
  4) Handling fatal errors with a non-zero exit code.

All scientific computation and I/O orchestration lives in RunManager and the
workflow components it coordinates.
"""

from __future__ import annotations

import logging

from CodeEntropy.config.run import RunManager

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the entropy workflow.

    Main function for calculating the entropy of a system using the multiscale cell
    correlation method.

    This function is the CLI entry point. It creates the output/job folder, then
    delegates to :class:`~CodeEntropy.config.run.RunManager` to execute the full
    workflow.

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


if __name__ == "__main__":
    main()  # pragma: no cover
