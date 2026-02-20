import logging

import pytest

from CodeEntropy.core.logging import LoggingConfig


@pytest.fixture(autouse=True)
def _isolate_global_logging():
    """
    LoggingConfig modifies global loggers. Keep tests atomic by clearing handlers
    after each test so tests don't leak state into each other.
    """
    yield
    for name in ("", "commands", "MDAnalysis"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


@pytest.fixture()
def config(tmp_path):
    return LoggingConfig(folder=str(tmp_path), level=logging.INFO)
