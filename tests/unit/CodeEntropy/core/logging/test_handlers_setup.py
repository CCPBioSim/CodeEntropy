import logging
import os

from rich.logging import RichHandler

from CodeEntropy.core.logging import LoggingConfig


def test_init_creates_log_dir(tmp_path):
    cfg = LoggingConfig(folder=str(tmp_path))
    assert os.path.isdir(cfg.log_dir)


def test_setup_handlers_creates_expected_handlers(config):
    assert set(config.handlers.keys()) == {
        "rich",
        "main",
        "error",
        "command",
        "mdanalysis",
    }

    assert isinstance(config.handlers["rich"], RichHandler)
    assert isinstance(config.handlers["main"], logging.FileHandler)
    assert isinstance(config.handlers["error"], logging.FileHandler)
    assert isinstance(config.handlers["command"], logging.FileHandler)
    assert isinstance(config.handlers["mdanalysis"], logging.FileHandler)


def test_handler_paths_match_expected_filenames(config):
    expected = {
        "main": "program.log",
        "error": "program.err",
        "command": "program.com",
        "mdanalysis": "mdanalysis.log",
    }

    for handler_key, filename in expected.items():
        handler = config.handlers[handler_key]
        assert os.path.basename(handler.baseFilename) == filename
        assert os.path.dirname(handler.baseFilename) == config.log_dir
