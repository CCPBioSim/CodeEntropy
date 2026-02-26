import logging


def test_configure_attaches_handlers(config):
    config.configure()

    root = logging.getLogger()
    assert config.handlers["rich"] in root.handlers
    assert config.handlers["main"] in root.handlers
    assert config.handlers["error"] in root.handlers


def test_configure_commands_logger_non_propagating_with_handler(config):
    config.configure()

    commands_logger = logging.getLogger("commands")
    assert commands_logger.propagate is False
    assert config.handlers["command"] in commands_logger.handlers


def test_configure_mdanalysis_logger_non_propagating_with_handler(config):
    config.configure()

    mda_logger = logging.getLogger("MDAnalysis")
    assert mda_logger.propagate is False
    assert config.handlers["mdanalysis"] in mda_logger.handlers
