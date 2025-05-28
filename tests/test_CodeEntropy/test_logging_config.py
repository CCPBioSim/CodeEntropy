import logging
import os
import tempfile
import unittest

from CodeEntropy.config.logging_config import LoggingConfig


class TestLoggingConfig(unittest.TestCase):

    def setUp(self):
        # Use a temporary directory for logs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = os.path.join(self.temp_dir.name, "logs")
        self.logging_config = LoggingConfig(folder=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_directory_created(self):
        """Check if the log directory is created upon init"""
        self.assertTrue(os.path.exists(self.log_dir))
        self.assertTrue(os.path.isdir(self.log_dir))

    def test_setup_logging_returns_logger(self):
        """Ensure setup_logging returns a logger instance"""
        logger = self.logging_config.setup_logging()
        self.assertIsInstance(logger, logging.Logger)

    def test_expected_log_files_created(self):
        """Ensure log file paths are configured correctly in the logging config"""
        self.logging_config.setup_logging()

        # Map actual output files to their corresponding handler keys
        expected_handlers = {
            "program.out": "stdout",
            "program.log": "logfile",
            "program.err": "errorfile",
            "program.com": "commandfile",
            "mdanalysis.log": "mdanalysis_log",
        }

        for filename, handler_key in expected_handlers.items():
            expected_path = os.path.join(self.log_dir, filename)
            actual_path = self.logging_config.LOGGING["handlers"][handler_key][
                "filename"
            ]
            self.assertEqual(actual_path, expected_path)

    def test_update_logging_level(self):
        """Ensure logging levels are updated correctly"""
        self.logging_config.setup_logging()

        # Update to DEBUG
        self.logging_config.update_logging_level(logging.DEBUG)
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)

        # Check that at least one handler is DEBUG
        handler_levels = [h.level for h in root_logger.handlers]
        self.assertIn(logging.DEBUG, handler_levels)

        # Update to INFO
        self.logging_config.update_logging_level(logging.INFO)
        self.assertEqual(root_logger.level, logging.INFO)

    def test_mdanalysis_and_command_loggers_exist(self):
        """Ensure specialized loggers are set up with correct configuration"""
        log_level = logging.DEBUG
        self.logging_config = LoggingConfig(
            folder=self.temp_dir.name, log_level=log_level
        )
        self.logging_config.setup_logging()

        mda_logger = logging.getLogger("MDAnalysis")
        cmd_logger = logging.getLogger("commands")

        self.assertEqual(mda_logger.level, log_level)
        self.assertEqual(cmd_logger.level, logging.INFO)
        self.assertFalse(mda_logger.propagate)
        self.assertFalse(cmd_logger.propagate)


if __name__ == "__main__":
    unittest.main()
