import logging
import os
import unittest
from unittest.mock import MagicMock

from CodeEntropy.config.logging_config import LoggingConfig
from tests.test_CodeEntropy.test_base import BaseTestCase


class TestLoggingConfig(BaseTestCase):
    """
    Unit tests for LoggingConfig.
    """

    def setUp(self):
        super().setUp()
        self.log_dir = self.logs_path
        self.logging_config = LoggingConfig(folder=self.test_dir)

        self.mock_text = "Test console output"
        self.logging_config.console.export_text = MagicMock(return_value=self.mock_text)

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

        # Map expected filenames to the corresponding handler keys in LoggingConfig
        expected_handlers = {
            "program.log": "main",
            "program.err": "error",
            "program.com": "command",
            "mdanalysis.log": "mdanalysis",
        }

        for filename, handler_key in expected_handlers.items():
            expected_path = os.path.join(self.logging_config.log_dir, filename)
            actual_path = self.logging_config.handlers[handler_key].baseFilename
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
        self.logging_config = LoggingConfig(folder=self.test_dir, level=log_level)
        self.logging_config.setup_logging()

        mda_logger = logging.getLogger("MDAnalysis")
        cmd_logger = logging.getLogger("commands")

        self.assertEqual(mda_logger.level, log_level)
        self.assertEqual(cmd_logger.level, logging.INFO)
        self.assertFalse(mda_logger.propagate)
        self.assertFalse(cmd_logger.propagate)

    def test_save_console_log_writes_file(self):
        """
        Test that save_console_log creates a log file in the expected location
        and writes the console's recorded output correctly.
        """
        filename = "test_log.txt"
        self.logging_config.save_console_log(filename)

        output_path = os.path.join(self.test_dir, "logs", filename)
        # Check file exists
        self.assertTrue(os.path.exists(output_path))

        # Read content and check it matches mocked export_text output
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, self.mock_text)

        # Ensure export_text was called once
        self.logging_config.console.export_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
