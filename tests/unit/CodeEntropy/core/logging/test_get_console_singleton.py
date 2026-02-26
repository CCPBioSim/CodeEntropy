from CodeEntropy.core.logging import LoggingConfig


def test_get_console_returns_singleton():
    # Reset singleton to make the test independent
    LoggingConfig._console = None

    c1 = LoggingConfig.get_console()
    c2 = LoggingConfig.get_console()

    assert c1 is c2


def test_get_console_records_output_enabled():
    LoggingConfig._console = None
    c = LoggingConfig.get_console()

    # Rich Console uses 'record' attribute when recording is enabled
    assert getattr(c, "record", False) is True
