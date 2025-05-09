import logging
import sys

from CodeEntropy.run import RunManager

logger = logging.getLogger(__name__)


def main():
    """
    Main function for calculating the entropy of a system using the multiscale cell
    correlation method.
    """

    # Setup initial services
    folder = RunManager.create_job_folder()

    try:
        run_manager = RunManager(folder=folder)
        run_manager.run_entropy_workflow()
    except Exception as e:
        logger.critical(f"Fatal error during entropy calculation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":

    main()
