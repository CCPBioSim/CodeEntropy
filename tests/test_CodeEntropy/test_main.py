import os
import shutil
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from CodeEntropy.main import main
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestMain(BaseTestCase):
    """
    Unit tests for the main functionality of CodeEntropy.
    """

    def setUp(self):
        super().setUp()
        self.code_entropy = main

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
        mock_RunManager.create_job_folder.return_value = "mock_folder/job001"

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
        mock_RunManager.create_job_folder.return_value = "mock_folder/job001"

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

    def test_main_entry_point_runs(self):
        """
        Test that the CLI entry point (main.py) runs successfully with minimal required
        arguments.
        """
        # Prepare input files
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data")
        )
        tpr_path = shutil.copy(os.path.join(data_dir, "md_A4_dna.tpr"), self.test_dir)
        trr_path = shutil.copy(
            os.path.join(data_dir, "md_A4_dna_xf.trr"), self.test_dir
        )

        config_path = os.path.join(self.test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("run1:\n" "  end: 60\n" "  selection_string: resid 1\n")

        citation_path = os.path.join(self.test_dir, "CITATION.cff")
        with open(citation_path, "w") as f:
            f.write("run1:\n" "  end: 60\n" "  selection_string: resid 1\n")

        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "CodeEntropy.main",
                "--top_traj_file",
                tpr_path,
                trr_path,
            ],
            cwd=self.test_dir,
            capture_output=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 0)

        # Check for job folder and output file
        job_dir = os.path.join(self.test_dir, "job001")
        output_file = os.path.join(job_dir, "output_file.json")

        self.assertTrue(os.path.exists(job_dir))
        self.assertTrue(os.path.exists(output_file))

        with open(output_file) as f:
            content = f.read()
            self.assertIn("DA", content)


if __name__ == "__main__":
    unittest.main()
