import logging

from CodeEntropy.core.logging import ErrorFilter


def test_error_filter_allows_error_and_critical():
    f = ErrorFilter()

    record_error = logging.LogRecord("x", logging.ERROR, "f.py", 1, "msg", (), None)
    record_crit = logging.LogRecord("x", logging.CRITICAL, "f.py", 1, "msg", (), None)

    assert f.filter(record_error) is True
    assert f.filter(record_crit) is True


def test_error_filter_blocks_below_error():
    f = ErrorFilter()

    record_warn = logging.LogRecord("x", logging.WARNING, "f.py", 1, "msg", (), None)
    record_info = logging.LogRecord("x", logging.INFO, "f.py", 1, "msg", (), None)

    assert f.filter(record_warn) is False
    assert f.filter(record_info) is False
