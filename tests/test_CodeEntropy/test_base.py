import logging
import os
import shutil
import tempfile
import unittest
import uuid


class BaseTestCase(unittest.TestCase):
    """
    Base class for tests with cross-platform setup and teardown.
    Creates unique temporary directories and pre-creates expected log files.
    """

    def setUp(self):
        # Unique temporary test directory
        self.test_dir = tempfile.mkdtemp(prefix="CodeEntropy_")
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Unique job folder + logs
        self.job_id = f"job_{uuid.uuid4().hex[:6]}"
        self.job_path = os.path.join(self.test_dir, self.job_id)
        self.logs_path = os.path.join(self.job_path, "logs")
        os.makedirs(self.logs_path, exist_ok=True)

        # Pre-create log files
        for fname in ["mdanalysis.log", "program.log", "program.com"]:
            with open(os.path.join(self.logs_path, fname), "w") as f:
                f.write("")

    def tearDown(self):
        # Shutdown logging and remove handlers (important for Windows)
        logging.shutdown()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Restore working directory
        os.chdir(self._orig_dir)

        # Remove temp directory (fail loudly if locked)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=False)
