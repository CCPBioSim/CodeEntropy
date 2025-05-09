import unittest
from unittest.mock import MagicMock, patch

from CodeEntropy.main import main


class TestMain(unittest.TestCase):
    """
    Unit tests for the main functionality of CodeEntropy.
    """

    @patch("CodeEntropy.main.sys.exit")
    @patch("CodeEntropy.main.RunManager")
    def test_main_successful_run(self, mock_RunManager, mock_exit):
        """
        Test that main runs successfully and does not call sys.exit.
        """
        # Mock RunManager's methods to simulate successful execution
        mock_run_manager_instance = MagicMock()
        mock_RunManager.return_value = mock_run_manager_instance

        # Simulate that RunManager.create_job_folder returns a folder
        mock_RunManager.create_job_folder.return_value = "dummy_folder"

        # Simulate the successful completion of the run_entropy_workflow method
        mock_run_manager_instance.run_entropy_workflow.return_value = None

        # Run the main function
        main()

        # Verify that sys.exit was not called
        mock_exit.assert_not_called()

        # Verify that RunManager's methods were called correctly
        mock_RunManager.create_job_folder.assert_called_once()
        mock_run_manager_instance.run_entropy_workflow.assert_called_once()

    @patch("CodeEntropy.main.sys.exit")
    @patch("CodeEntropy.main.RunManager")
    @patch("CodeEntropy.main.logger")
    def test_main_exception_triggers_exit(
        self, mock_logger, mock_RunManager, mock_exit
    ):
        """
        Test that main logs a critical error and exits if RunManager
        raises an exception.
        """
        # Simulate an exception being raised in run_entropy_workflow
        mock_run_manager_instance = MagicMock()
        mock_RunManager.return_value = mock_run_manager_instance

        # Simulate that RunManager.create_job_folder returns a folder
        mock_RunManager.create_job_folder.return_value = "dummy_folder"

        # Simulate an exception in the run_entropy_workflow method
        mock_run_manager_instance.run_entropy_workflow.side_effect = Exception(
            "Test exception"
        )

        # Run the main function and mock sys.exit to ensure it gets called
        main()

        # Ensure sys.exit(1) was called due to the exception
        mock_exit.assert_called_once_with(1)

        # Ensure that the logger logged the critical error with exception details
        mock_logger.critical.assert_called_once_with(
            "Fatal error during entropy calculation: Test exception", exc_info=True
        )


if __name__ == "__main__":
    unittest.main()
