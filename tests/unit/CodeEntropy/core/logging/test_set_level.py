import logging


def test_set_level_updates_root_and_named_loggers(config):
    config.configure()

    config.set_level(logging.DEBUG)

    root = logging.getLogger()
    assert root.level == logging.DEBUG

    assert logging.getLogger("commands").level == logging.DEBUG
    assert logging.getLogger("MDAnalysis").level == logging.DEBUG


def test_set_level_sets_filehandlers_to_log_level_and_other_handlers_to_info(config):
    config.configure()

    config.set_level(logging.DEBUG)

    root = logging.getLogger()

    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            assert h.level == logging.DEBUG
        else:
            assert h.level == logging.INFO
