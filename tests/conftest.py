from __future__ import annotations

import sys

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """
    Force UTF-8 output for the terminal reporter.

    On Windows CI stdout is a pipe using cp1252, and pytest-modern crashes
    when it writes characters that cp1252 can't encode. This has to run
    before pytest-modern sets up its console.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
