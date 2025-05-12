import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from CodeEntropy.run import RunManager


class TestRunManager(unittest.TestCase):
    """
    Unit tests for the RunManager class. These tests verify the
    correct behavior of run manager.
    """

    def setUp(self):
        """
        Set up a temporary directory as the working directory before each test.
        """
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """
        Clean up by removing the temporary directory and restoring the original working
        directory.
        """
        os.chdir(self._orig_dir)
        shutil.rmtree(self.test_dir)

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_empty_directory(self, mock_listdir, mock_makedirs):
        """
        Test that 'job001' is created when the directory is initially empty.
        """
        mock_listdir.return_value = []
        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job001")
        self.assertEqual(new_folder_path, expected_path)
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_with_existing_folders(self, mock_listdir, mock_makedirs):
        """
        Test that the next sequential job folder (e.g., 'job004') is created when
        existing folders 'job001', 'job002', and 'job003' are present.
        """
        mock_listdir.return_value = ["job001", "job002", "job003"]
        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job004")
        self.assertEqual(new_folder_path, expected_path)
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_with_non_matching_folders(
        self, mock_listdir, mock_makedirs
    ):
        """
        Test that 'job001' is created when the directory contains only non-job-related
        folders.
        """
        mock_listdir.return_value = ["folderA", "another_one"]
        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job001")
        self.assertEqual(new_folder_path, expected_path)
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_mixed_folder_names(self, mock_listdir, mock_makedirs):
        """
        Test that the correct next job folder (e.g., 'job003') is created when both
        job and non-job folders exist in the directory.
        """
        mock_listdir.return_value = ["job001", "abc", "job002", "random"]
        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job003")
        self.assertEqual(new_folder_path, expected_path)
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    @patch("os.makedirs")
    @patch("os.listdir")
    def test_create_job_folder_with_invalid_job_suffix(
        self, mock_listdir, mock_makedirs
    ):
        """
        Test that invalid job folder names like 'jobABC' are ignored when determining
        the next job number.
        """
        # Simulate existing folders, one of which is invalid
        mock_listdir.return_value = ["job001", "jobABC", "job002"]

        new_folder_path = RunManager.create_job_folder()
        expected_path = os.path.join(self.test_dir, "job003")

        self.assertEqual(new_folder_path, expected_path)
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)


if __name__ == "__main__":
    unittest.main()
