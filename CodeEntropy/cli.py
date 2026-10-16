"""Command-line entry point for CodeEntropy.

This module provides the CLI entry point used to run the multiscale cell
correlation entropy workflow.

The entry point is intentionally small and only responsible for:
  1) Handling informational flags such as ``--version``.
  2) Creating a job folder.
  3) Constructing a CodeEntropyRunner.
  4) Executing the entropy workflow.
  5) Handling fatal errors with a non-zero exit code.
"""

from __future__ import annotations

import logging

from CodeEntropy.config.argparse import ConfigResolver
from CodeEntropy.config.runtime import CodeEntropyRunner

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the entropy workflow.

    Arguments are parsed before any side effects so that informational flags such
    as ``--version`` and ``--help`` exit without creating a job folder or
    rendering the splash screen.

    Raises:
        SystemExit: Exits with status code 1 on any unhandled exception.
    """
    ConfigResolver().build_parser().parse_known_args()

    folder = CodeEntropyRunner.create_job_folder()

    try:
        run_manager = CodeEntropyRunner(folder=folder)
        run_manager.run_entropy_workflow()
    except Exception:
        logger.exception("Fatal error during entropy calculation")
        raise SystemExit(1) from None
