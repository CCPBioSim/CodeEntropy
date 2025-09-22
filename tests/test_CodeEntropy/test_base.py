import os
import shutil
import tempfile
import unittest


class BaseTestCase(unittest.TestCase):
    """
    Base test case class for cross-platform unit tests.

    Provides:
    1. A unique temporary directory for each test to avoid filesystem conflicts.
    2. Automatic restoration of the working directory after each test.
    3. Prepares a logs folder path for tests that need logging configuration.
    """

    def setUp(self):
        """
        Prepare the test environment before each test method runs.

        Actions performed:
        1. Creates a unique temporary directory for the test.
        2. Creates a 'logs' subdirectory within the temp directory.
        3. Changes the current working directory to the temporary directory.
        """
        # Create a unique temporary test directory
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self.logs_path = os.path.join(self.test_dir, "logs")
        os.makedirs(self.logs_path, exist_ok=True)

        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """
        Clean up the test environment after each test method runs.

        Actions performed:
        1. Restores the original working directory.
        2. Deletes the temporary test directory along with all its contents.
        """
        os.chdir(self._orig_dir)

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
